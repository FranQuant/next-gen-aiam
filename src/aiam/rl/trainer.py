"""REINFORCE with baseline trainer for SimplexPolicy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aiam.rl.env import PortfolioEnv
from aiam.rl.policy import SimplexPolicy


@dataclass
class TrainConfig:
    episodes: int = 200
    gamma: float = 0.95
    lr: float = 1e-3
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    max_steps_per_episode: int | None = None  # None = full rollout; set e.g. 60 for fast CPU runs


@dataclass
class TrainHistory:
    episode_rewards: list[float] = field(default_factory=list)
    mean_turnovers: list[float] = field(default_factory=list)
    mean_weights: list[np.ndarray] = field(default_factory=list)


class ValueHead(nn.Module):
    """Single linear head mapping mean-pooled hidden state to scalar value."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, hidden_dim) → (B,) scalar value per step."""
        return self.net(h.mean(dim=1)).squeeze(-1)


def _compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Monte-Carlo returns-to-go, standardized."""
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    t = torch.tensor(returns, dtype=torch.float32)
    if t.std() > 1e-8:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


def _policy_forward_with_logprob(
    policy: SimplexPolicy,
    state: dict,
    temperature: float = 1.0,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Sample action from Dirichlet and return (action, log_prob, hidden).

    Returns:
        action     : (N,) sampled weight vector
        log_prob   : scalar tensor (differentiable through alpha)
        hidden_enc : (1, N, hidden_dim) encoder output for value head
    """
    feat = torch.tensor(state["features"][None], dtype=torch.float32)   # (1, N, F)
    w = torch.tensor(state["weights"][None, :, None], dtype=torch.float32)  # (1, N, 1)

    policy.train()
    x = torch.cat([feat, w], dim=-1) if policy.use_weights else feat    # (1, N, F+1)
    B, N, _ = x.shape
    h_enc = policy.encoder(x.reshape(B * N, -1)).reshape(B, N, -1)     # (1, N, H)
    logits = policy.head(h_enc.reshape(B * N, -1)).reshape(B, N)        # (1, N)
    alpha_base = torch.softmax(logits, dim=-1).squeeze(0)               # (N,)

    alpha = (alpha_base * temperature).clamp(min=1e-3)
    dist = torch.distributions.Dirichlet(alpha)
    action_t = dist.sample()                                             # (N,) – tensor
    log_prob = dist.log_prob(action_t)                                   # scalar

    return action_t.detach().cpu().numpy(), log_prob, h_enc


def run_episode(
    policy: SimplexPolicy,
    value_head: ValueHead | None,
    env: PortfolioEnv,
    temperature: float = 1.0,
    max_steps: int | None = None,
) -> tuple[list, list, list, list, list]:
    """Roll out one episode. Returns (rewards, log_probs, hiddens, turnovers, weights).

    When max_steps is set, the episode is truncated after that many steps.
    env.reset() is called at the start; for multi-episode training the caller
    is responsible for randomising the env start if desired.
    """
    state = env.reset()
    done = False
    step = 0
    rewards, log_probs, hiddens, all_turnovers, all_weights = [], [], [], [], []

    while not done:
        action, lp, h_enc = _policy_forward_with_logprob(policy, state, temperature)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        log_probs.append(lp)
        hiddens.append(h_enc)
        all_turnovers.append(info.get("turnover", 0.0))
        all_weights.append(action)
        state = next_state
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return rewards, log_probs, hiddens, all_turnovers, all_weights


def train_step(
    policy: SimplexPolicy,
    value_head: ValueHead | None,
    optimizer: optim.Optimizer,
    env: PortfolioEnv,
    config: TrainConfig,
    temperature: float = 1.0,
) -> dict:
    """One REINFORCE update. Returns metrics dict."""
    rewards, log_probs, hiddens, turnovers, weights = run_episode(
        policy, value_head, env, temperature, max_steps=config.max_steps_per_episode
    )
    returns_t = _compute_returns(rewards, config.gamma)   # (T,)

    # Baseline: learned value or mean return.
    if value_head is not None and hiddens:
        h_cat = torch.cat(hiddens, dim=0)                 # (T, N, H)
        values = value_head(h_cat)                        # (T,)
        advantage = (returns_t - values.detach())
        value_loss = nn.functional.mse_loss(values, returns_t)
    else:
        advantage = returns_t - returns_t.mean()
        value_loss = torch.tensor(0.0)

    log_probs_t = torch.stack(log_probs)                  # (T,)

    # Entropy bonus: Dirichlet entropy (approximate via −log_prob mean).
    entropy = -log_probs_t.mean()

    policy_loss = -(log_probs_t * advantage.detach()).mean()
    loss = policy_loss + value_loss - config.entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping — clip both policy and value head together.
    params = list(policy.parameters())
    if value_head is not None:
        params += list(value_head.parameters())
    total_norm = nn.utils.clip_grad_norm_(params, config.grad_clip)

    optimizer.step()

    return {
        "total_reward": float(sum(rewards)),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "mean_weights": np.mean(weights, axis=0) if weights else np.array([]),
        "grad_norm": float(total_norm),
    }


def train(
    policy: SimplexPolicy,
    env: PortfolioEnv,
    config: TrainConfig,
    use_value_baseline: bool = True,
    temperature: float = 1.0,
) -> tuple[TrainHistory, ValueHead | None]:
    """Full training run. Returns (history, value_head)."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    value_head = ValueHead(hidden_dim=policy.encoder[0].out_features) if use_value_baseline else None

    params = list(policy.parameters())
    if value_head is not None:
        params += list(value_head.parameters())
    optimizer = optim.Adam(params, lr=config.lr)

    history = TrainHistory()
    for _ in range(config.episodes):
        metrics = train_step(policy, value_head, optimizer, env, config, temperature)
        history.episode_rewards.append(metrics["total_reward"])
        history.mean_turnovers.append(metrics["mean_turnover"])
        history.mean_weights.append(metrics["mean_weights"])

    return history, value_head
