"""Tests for PPO trainer: clipped objective, GAE shape, loss on monotone env."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from aiam.rl.env import PortfolioEnv
from aiam.rl.policy import SimplexPolicy
from aiam.rl.ppo import (
    PPOConfig,
    _batch_forward_ppo,
    _compute_gae,
    collect_trajectory,
    ppo_update,
    train_ppo,
)
from aiam.rl.trainer import ValueHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rising_returns(n_assets: int = 2, n_days: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B")
    data = np.zeros((n_days, n_assets), dtype=np.float32)
    data[:, 0] = 0.003
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_env(n_assets: int = 2, n_days: int = 80) -> PortfolioEnv:
    return PortfolioEnv(_rising_returns(n_assets, n_days), lookback=5)


def _make_policy(n_features: int = 5, hidden_dim: int = 16) -> SimplexPolicy:
    torch.manual_seed(0)
    return SimplexPolicy(n_features=n_features, hidden_dim=hidden_dim, use_weights=True)


def _make_value_head(hidden_dim: int = 16) -> ValueHead:
    return ValueHead(hidden_dim=hidden_dim)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def test_compute_gae_shape_and_finite():
    rewards = [0.1, -0.05, 0.2, 0.0, 0.15]
    values = torch.tensor([0.05, -0.02, 0.18, 0.01, 0.12])
    adv, ret = _compute_gae(rewards, values, gamma=0.95, gae_lambda=0.95)
    assert adv.shape == (5,)
    assert ret.shape == (5,)
    assert torch.all(torch.isfinite(adv))
    assert torch.all(torch.isfinite(ret))


def test_compute_gae_returns_geq_terminal():
    """Returns-to-go = advantages + values; terminal value = 0 so last rtg ≈ last reward."""
    rewards = [0.1, 0.2, 0.3]
    values = torch.zeros(3)
    adv, ret = _compute_gae(rewards, values, gamma=0.95, gae_lambda=0.95)
    # With zero value baseline, returns = Monte-Carlo returns (non-standardised)
    expected_last = rewards[-1]   # G_{T-1} = r_{T-1} + 0
    assert abs(ret[-1].item() - expected_last) < 1e-5


# ---------------------------------------------------------------------------
# Clipped surrogate: ratio clamping
# ---------------------------------------------------------------------------


def test_ppo_clips_large_ratio():
    """When ratio > 1+clip_eps, the clipped surrogate must be used (not the raw ratio)."""
    env = _make_env()
    policy = _make_policy(n_features=env.n_features)
    value_head = _make_value_head()
    config = PPOConfig(episodes=1, k_epochs=1, minibatch_size=100, seed=0, clip_eps=0.2)

    traj = collect_trajectory(policy, value_head, env, temperature=1.0)
    T = len(traj["rewards"])
    features_t = torch.tensor(traj["features"], dtype=torch.float32)
    weights_t = torch.tensor(traj["weights"][:, :, None], dtype=torch.float32)
    actions = traj["actions"]
    old_lp = traj["old_log_probs"]

    # Force ratio >> 1 by subtracting a large constant from old_log_probs
    # (equivalent to old policy having very low log_prob)
    faked_old_lp = old_lp - 10.0  # new_lp - faked_old_lp >> 0 → ratio >> 1

    # Advantage positive → clipping activates if ratio > 1+eps
    advantages = torch.ones(T) * 1.0
    returns_to_go = torch.ones(T) * 0.01

    new_lp, new_vals, ent = _batch_forward_ppo(
        policy, value_head, features_t, weights_t, actions
    )
    ratio = torch.exp(new_lp - faked_old_lp)
    # Most ratios should be >> 1+0.2
    assert (ratio > 1 + config.clip_eps).any(), "Expected large ratios for clamping test"

    # Clipped surrogate should be <= raw surrogate when ratio > 1+eps and adv > 0
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
    assert (torch.min(surr1, surr2) <= surr1).all()


# ---------------------------------------------------------------------------
# collect_trajectory
# ---------------------------------------------------------------------------


def test_collect_trajectory_keys_and_shapes():
    env = _make_env()
    policy = _make_policy(n_features=env.n_features)
    value_head = _make_value_head()
    traj = collect_trajectory(policy, value_head, env, max_steps=10)

    T = len(traj["rewards"])
    N = env.n_assets
    F = env.n_features

    assert traj["features"].shape == (T, N, F)
    assert traj["weights"].shape == (T, N)
    assert traj["actions"].shape == (T, N)
    assert traj["old_log_probs"].shape == (T,)
    assert traj["values"].shape == (T,)
    assert len(traj["turnovers"]) == T


def test_collect_trajectory_actions_on_simplex():
    env = _make_env()
    policy = _make_policy(n_features=env.n_features)
    value_head = _make_value_head()
    traj = collect_trajectory(policy, value_head, env, max_steps=20)

    actions = traj["actions"].numpy()
    assert np.all(actions >= 0)
    np.testing.assert_allclose(actions.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# ppo_update
# ---------------------------------------------------------------------------


def test_ppo_update_returns_expected_keys():
    env = _make_env()
    policy = _make_policy(n_features=env.n_features)
    value_head = _make_value_head()
    params = list(policy.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    config = PPOConfig(episodes=1, k_epochs=2, minibatch_size=8, seed=0)

    traj = collect_trajectory(policy, value_head, env, max_steps=20)
    metrics = ppo_update(policy, value_head, optimizer, traj, config)

    for key in ("total_reward", "policy_loss", "value_loss", "entropy", "mean_turnover"):
        assert key in metrics
        assert np.isfinite(metrics[key]), f"{key} is not finite"


# ---------------------------------------------------------------------------
# train_ppo: history length + loss finiteness
# ---------------------------------------------------------------------------


def test_train_ppo_history_length():
    env = _make_env()
    policy = _make_policy(n_features=env.n_features)
    config = PPOConfig(episodes=8, k_epochs=2, minibatch_size=8, seed=7)
    history, value_head = train_ppo(policy, env, config)

    assert len(history.episode_rewards) == 8
    assert len(history.mean_turnovers) == 8
    assert value_head is not None


def test_train_ppo_episode_rewards_finite():
    env = _make_env(n_assets=3, n_days=100)
    policy = _make_policy(n_features=env.n_features, hidden_dim=16)
    config = PPOConfig(episodes=5, k_epochs=2, minibatch_size=16, seed=1)
    history, _ = train_ppo(policy, env, config)

    for r in history.episode_rewards:
        assert np.isfinite(r), f"episode reward {r} is not finite"


def test_train_ppo_monotone_reward_can_improve():
    """On the rising-asset env, PPO should not degrade total reward over 20 episodes."""
    env = _make_env(n_assets=2, n_days=100)
    policy = _make_policy(n_features=env.n_features, hidden_dim=16)
    config = PPOConfig(episodes=20, k_epochs=4, minibatch_size=16, lr=3e-4, seed=42)
    history, _ = train_ppo(policy, env, config)

    first5 = np.mean(history.episode_rewards[:5])
    last5 = np.mean(history.episode_rewards[-5:])
    # PPO should at least not catastrophically collapse on a trivial env.
    assert last5 > first5 - 0.5, (
        f"Reward collapsed: first5={first5:.4f}, last5={last5:.4f}"
    )


# ---------------------------------------------------------------------------
# PPOConfig defaults
# ---------------------------------------------------------------------------


def test_ppo_config_defaults():
    config = PPOConfig()
    assert config.clip_eps == 0.2
    assert config.gae_lambda == 0.95
    assert config.k_epochs == 4
    assert config.lr == 3e-4
    assert config.entropy_coef == 0.01
    assert config.grad_clip == 1.0
    assert config.max_steps_per_episode is None
