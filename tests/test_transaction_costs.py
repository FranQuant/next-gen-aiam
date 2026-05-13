from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.evaluation.transaction_costs import apply_costs, compute_turnover


def _ew_weights(n_days: int = 100, n_assets: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    w = 1.0 / n_assets
    return pd.DataFrame(w, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _random_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-03", periods=n)  # one day after weights
    return pd.Series(rng.normal(0.0, 0.01, n), index=dates)


def test_ew_no_rebalance_zero_turnover():
    """Constant EW weights → turnover = 0 on every day except the first NaN."""
    weights = _ew_weights()
    to = compute_turnover(weights)
    assert to.isna().sum() == 1, "Exactly the first row should be NaN"
    assert (to.dropna() == 0.0).all(), "All non-NaN turnover should be exactly 0"


def test_ew_net_equals_gross():
    """Zero turnover → net returns = gross returns (to floating-point precision)."""
    weights = _ew_weights(n_days=100)
    gross = _random_returns(n=100)
    net = apply_costs(gross, weights, cost_bps=10.0)
    pd.testing.assert_series_equal(net, gross, check_names=False)


def test_turnover_bounds():
    """Turnover is always in [0, 1] for valid (non-negative, summing-to-1) weights."""
    rng = np.random.default_rng(42)
    n, k = 200, 10
    dates = pd.bdate_range("2020-01-02", periods=n)
    raw = rng.dirichlet(np.ones(k), size=n)
    weights = pd.DataFrame(raw, index=dates, columns=[f"A{i}" for i in range(k)])
    to = compute_turnover(weights).dropna()
    assert (to >= 0).all() and (to <= 1.0 + 1e-10).all()


def test_net_sharpe_below_gross():
    """With nonzero rebalancing, net Sharpe ≤ gross Sharpe (costs drag returns)."""
    rng = np.random.default_rng(7)
    n, k = 300, 5
    dates_w = pd.bdate_range("2020-01-02", periods=n)
    dates_r = pd.bdate_range("2020-01-03", periods=n)
    raw = rng.dirichlet(np.ones(k), size=n)
    weights = pd.DataFrame(raw, index=dates_w, columns=[f"A{i}" for i in range(k)])
    gross = pd.Series(rng.normal(5e-4, 0.01, n), index=dates_r)
    net = apply_costs(gross, weights, cost_bps=10.0)
    gross_sharpe = gross.mean() / gross.std() * np.sqrt(252)
    net_sharpe = net.mean() / net.std() * np.sqrt(252)
    assert net_sharpe <= gross_sharpe + 1e-10


def test_higher_cost_lower_net():
    """Higher cost_bps → lower net Sharpe."""
    rng = np.random.default_rng(3)
    n, k = 300, 5
    dates_w = pd.bdate_range("2020-01-02", periods=n)
    dates_r = pd.bdate_range("2020-01-03", periods=n)
    raw = rng.dirichlet(np.ones(k), size=n)
    weights = pd.DataFrame(raw, index=dates_w, columns=[f"A{i}" for i in range(k)])
    gross = pd.Series(rng.normal(5e-4, 0.01, n), index=dates_r)
    net_10 = apply_costs(gross, weights, cost_bps=10.0)
    net_50 = apply_costs(gross, weights, cost_bps=50.0)

    def sharpe(s):
        return s.mean() / s.std() * np.sqrt(252)

    assert sharpe(net_50) < sharpe(net_10)
