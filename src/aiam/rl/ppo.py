"""PPO (Proximal Policy Optimization) trainer reusing SimplexPolicy + PortfolioEnv."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aiam.rl.env import PortfolioEnv
from aiam.rl.policy import SimplexPolicy
from aiam.rl.trainer import TrainHistory, ValueHead

__all__ = [
    "PPOConfig",
    "collect_trajectory",
    "ppo_update",
    "train_ppo",
]


@dataclass
class PPOConfig:
    episodes: int = 200
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    k_epochs: int = 4
    minibatch_size: int = 32
    value_coef: float = 0.5
    lr: float = 3e-4
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    max_steps_per_episode: int | None = None


def _compute_gae(
    rewards: list[float],
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantages and returns-to-go. Both shape (T,)."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    next_val = 0.0  # bootstrap V(s_{T+1}) = 0 at episode end
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_val - values[t].item()
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        next_val = values[t].item()
    returns_to_go = advantages + values.detach()
    return advantages, returns_to_go


def _batch_forward_ppo(
    policy: SimplexPolicy,
    value_head: ValueHead,
    features: torch.Tensor,   # (B, N, F)
    weights: torch.Tensor,    # (B, N, 1)
    actions: torch.Tensor,    # (B, N)
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-evaluate policy and value head for a minibatch. Returns (log_probs, values, entropy)."""
    B, N, _ = features.shape
    x = torch.cat([features, weights], dim=-1) if policy.use_weights else features  # (B, N, F[+1])
    h = policy.encoder(x.reshape(B * N, -1)).reshape(B, N, -1)    # (B, N, H)
    logits = policy.head(h.reshape(B * N, -1)).reshape(B, N)       # (B, N)
    alpha_base = torch.softmax(logits, dim=-1)                      # (B, N)
    alpha = (alpha_base * temperature).clamp(min=1e-3)
    dist = torch.distributions.Dirichlet(alpha)
    # Clamp + renormalize to keep actions strictly on the simplex.
    actions_safe = actions.clamp(min=1e-7)
    actions_safe = actions_safe / actions_safe.sum(dim=-1, keepdim=True)
    log_probs = dist.log_prob(actions_safe)                          # (B,)
    entropy = dist.entropy()                                         # (B,)
    values = value_head(h)                                           # (B,)
    return log_probs, values, entropy


def collect_trajectory(
    policy: SimplexPolicy,
    value_head: ValueHead,
    env: PortfolioEnv,
    temperature: float = 1.0,
    max_steps: int | None = None,
) -> dict:
    """Roll out one episode without gradients. Returns tensors for PPO update."""
    state = env.reset()
    done = False
    step = 0

    features_list, weights_list, actions_list = [], [], []
    old_log_probs_list, values_list = [], []
    rewards_list, turnovers_list = [], []

    policy.eval()
    value_head.eval()

    with torch.no_grad():
        while not done:
            feat = torch.tensor(state["features"][None], dtype=torch.float32)       # (1, N, F)
            w = torch.tensor(state["weights"][None, :, None], dtype=torch.float32)  # (1, N, 1)

            B, N, _ = feat.shape
            x = torch.cat([feat, w], dim=-1) if policy.use_weights else feat
            h = policy.encoder(x.reshape(B * N, -1)).reshape(B, N, -1)
            logits = policy.head(h.reshape(B * N, -1)).reshape(B, N)
            alpha_base = torch.softmax(logits, dim=-1).squeeze(0)       # (N,)
            alpha = (alpha_base * temperature).clamp(min=1e-3)
            dist = torch.distributions.Dirichlet(alpha)
            action_t = dist.sample()                                      # (N,)
            old_lp = dist.log_prob(action_t)                              # scalar
            value = value_head(h).squeeze(0)                              # scalar

            action_np = action_t.cpu().numpy()
            next_state, reward, done, info = env.step(action_np)

            features_list.append(state["features"])
            weights_list.append(state["weights"])
            actions_list.append(action_t)
            old_log_probs_list.append(old_lp)
            values_list.append(value)
            rewards_list.append(reward)
            turnovers_list.append(info.get("turnover", 0.0))

            state = next_state
            step += 1
            if max_steps is not None and step >= max_steps:
                break

    return {
        "features": np.stack(features_list),                   # (T, N, F)
        "weights": np.stack(weights_list),                      # (T, N)
        "actions": torch.stack(actions_list),                   # (T, N)
        "old_log_probs": torch.stack(old_log_probs_list),       # (T,)
        "values": torch.stack(values_list),                     # (T,)
        "rewards": rewards_list,
        "turnovers": turnovers_list,
    }


def ppo_update(
    policy: SimplexPolicy,
    value_head: ValueHead,
    optimizer: optim.Optimizer,
    trajectory: dict,
    config: PPOConfig,
    temperature: float = 1.0,
) -> dict:
    """K-epoch clipped PPO update from one collected trajectory. Returns metrics dict."""
    rewards = trajectory["rewards"]
    values = trajectory["values"]            # (T,)
    actions = trajectory["actions"]          # (T, N)
    old_log_probs = trajectory["old_log_probs"]  # (T,)
    T = len(rewards)

    advantages, returns_to_go = _compute_gae(rewards, values, config.gamma, config.gae_lambda)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    features_t = torch.tensor(trajectory["features"], dtype=torch.float32)          # (T, N, F)
    weights_t = torch.tensor(trajectory["weights"][:, :, None], dtype=torch.float32)  # (T, N, 1)

    policy_losses, value_losses, entropies = [], [], []
    indices = np.arange(T)

    policy.train()
    value_head.train()

    for _ in range(config.k_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, config.minibatch_size):
            mb = indices[start : start + config.minibatch_size]

            new_lp, new_vals, ent = _batch_forward_ppo(
                policy, value_head,
                features_t[mb], weights_t[mb], actions[mb], temperature,
            )

            ratio = torch.exp(new_lp - old_log_probs[mb])
            adv_mb = advantages[mb]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_vals, returns_to_go[mb])
            entropy = ent.mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()),
                config.grad_clip,
            )
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    return {
        "total_reward": float(sum(rewards)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "mean_turnover": float(np.mean(trajectory["turnovers"])) if trajectory["turnovers"] else 0.0,
        "mean_weights": actions.mean(dim=0).detach().cpu().numpy(),
    }


def train_ppo(
    policy: SimplexPolicy,
    env: PortfolioEnv,
    config: PPOConfig,
    temperature: float = 1.0,
) -> tuple[TrainHistory, ValueHead]:
    """Full PPO training run. Returns (history, value_head)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    value_head = ValueHead(hidden_dim=policy.encoder[0].out_features)
    params = list(policy.parameters()) + list(value_head.parameters())
    optimizer = optim.Adam(params, lr=config.lr)

    history = TrainHistory()
    for _ in range(config.episodes):
        traj = collect_trajectory(
            policy, value_head, env, temperature,
            max_steps=config.max_steps_per_episode,
        )
        metrics = ppo_update(policy, value_head, optimizer, traj, config, temperature)
        history.episode_rewards.append(metrics["total_reward"])
        history.mean_turnovers.append(metrics["mean_turnover"])
        history.mean_weights.append(metrics["mean_weights"])

    return history, value_head
