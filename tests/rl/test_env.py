"""Tests for src/aiam/rl/env.py — PortfolioEnv contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.rl.env import COST_BPS, PortfolioEnv


# ── Fixtures ────────────────────────────────────────────────────────────────

N = 29
T = 100


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((T, N)) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=T),
        columns=[f"A{i}" for i in range(N)],
    )


@pytest.fixture
def env(returns_df) -> PortfolioEnv:
    return PortfolioEnv(returns_df)


# ── State shape ─────────────────────────────────────────────────────────────

def test_reset_returns_valid_state_shape(env):
    state = env.reset()
    assert "features" in state and "weights" in state
    assert state["features"].shape == (N, env.n_features)
    assert state["weights"].shape == (N,)
    assert np.isclose(state["weights"].sum(), 1.0)
    assert not np.any(np.isnan(state["features"]))


# ── Step returns finite values ───────────────────────────────────────────────

def test_step_equal_weights_n29_no_nans(env):
    env.reset()
    w = np.ones(N, dtype=np.float32) / N
    _, reward, _, info = env.step(w)
    assert np.isfinite(reward), "reward must be finite"
    for key in ("gross_return", "turnover", "transaction_cost", "risk_penalty", "net_return"):
        assert np.isfinite(info[key]), f"info[{key!r}] must be finite"


# ── Turnover logic ───────────────────────────────────────────────────────────

def test_buy_and_hold_has_zero_turnover(env):
    """After setting equal weights, stepping with equal weights again → turnover = 0."""
    env.reset()
    w = np.ones(N, dtype=np.float32) / N
    # Initial _weights is already equal; step → prev == new → turnover = 0.
    _, _, _, info = env.step(w)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-5)


def test_rebalance_to_different_weights_has_nonzero_turnover(env):
    """After holding equal weights, shifting all weight to one asset → turnover > 0."""
    env.reset()
    w_eq = np.ones(N, dtype=np.float32) / N
    env.step(w_eq)  # weights are now equal

    w_concentrated = np.zeros(N, dtype=np.float32)
    w_concentrated[0] = 1.0
    _, _, _, info = env.step(w_concentrated)
    assert info["turnover"] > 0.01


# ── Transaction cost is proportional to turnover ────────────────────────────

def test_transaction_cost_proportional_to_turnover(env):
    env.reset()
    w = np.zeros(N, dtype=np.float32)
    w[0] = 1.0  # large move from equal weight
    _, _, _, info = env.step(w)
    expected_tc = COST_BPS / 10_000.0 * info["turnover"]
    assert info["transaction_cost"] == pytest.approx(expected_tc, abs=1e-8)


# ── Action validation ────────────────────────────────────────────────────────

def test_action_with_wrong_sum_raises(env):
    env.reset()
    bad = np.full(N, 0.1, dtype=np.float32)  # sum ≈ 2.9 ≠ 1
    with pytest.raises(ValueError, match="sum"):
        env.step(bad)


def test_action_with_negative_values_raises(env):
    env.reset()
    bad = np.ones(N, dtype=np.float32) / N
    bad[0] = -0.5
    with pytest.raises(ValueError, match="negative"):
        env.step(bad)
