"""Tests for src/aiam/rl/strategy.py — RLStrategy PointInTimeStrategy contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.rl.agent import RLAgent
from aiam.rl.policy import SimplexPolicy
from aiam.rl.strategy import RLStrategy

LOOKBACK = 20
N_ASSETS = 5
T = 80


def _make_panel() -> Panel:
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.standard_normal((T, N_ASSETS)) * 0.01, axis=0) * 100,
        index=pd.bdate_range("2020-01-01", periods=T),
        columns=[f"A{i}" for i in range(N_ASSETS)],
    )
    return Panel({"prices": prices})


@pytest.fixture
def strategy() -> RLStrategy:
    policy = SimplexPolicy(n_features=LOOKBACK, hidden_dim=16)
    agent = RLAgent(policy, lookback=LOOKBACK)
    return RLStrategy(agent)


@pytest.fixture
def panel() -> Panel:
    return _make_panel()


def test_predict_weights_returns_series(strategy, panel):
    asof = pd.Timestamp("2020-05-01")
    weights = strategy.predict_weights(panel, asof)
    assert isinstance(weights, pd.Series)


def test_predict_weights_no_nans(strategy, panel):
    asof = pd.Timestamp("2020-05-01")
    weights = strategy.predict_weights(panel, asof)
    assert not weights.isna().any()


def test_predict_weights_sums_to_one(strategy, panel):
    asof = pd.Timestamp("2020-05-01")
    weights = strategy.predict_weights(panel, asof)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_predict_weights_non_negative(strategy, panel):
    asof = pd.Timestamp("2020-05-01")
    weights = strategy.predict_weights(panel, asof)
    assert (weights >= 0).all()


def test_predict_weights_index_subset_of_universe(strategy, panel):
    asof = pd.Timestamp("2020-05-01")
    weights = strategy.predict_weights(panel, asof)
    available = set(panel.universe_at(asof))
    for asset in weights.index:
        assert asset in available
