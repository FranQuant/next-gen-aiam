from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, sample_cov
from aiam.estimators.views import equilibrium_only, momentum_views
from aiam.strategy.black_litterman import BlackLitterman


def make_panel(
    means: list[float],
    n_obs: int = 600,
    seed: int = 42,
    daily_vol: float = 0.01,
) -> Panel:
    """Synthetic Panel with business-day rows. Price starts at 100."""
    dates = pd.bdate_range("2016-01-01", periods=n_obs + 1)
    tickers = [f"T{i}" for i in range(len(means))]
    rng = np.random.default_rng(seed)

    price_matrix = np.empty((n_obs + 1, len(tickers)))
    price_matrix[0] = 100.0
    for i in range(1, n_obs + 1):
        daily_returns = np.array(means) + rng.normal(0.0, daily_vol, size=len(means))
        price_matrix[i] = price_matrix[i - 1] * (1.0 + daily_returns)

    prices = pd.DataFrame(price_matrix, index=dates, columns=tickers)
    prices.index.name = "date"
    return Panel({"prices": prices})


# ── 1. Equilibrium-only: weights valid and Sharpe sensible ──────────────────

def test_equilibrium_only_weights_valid():
    panel = make_panel(means=[0.0003, 0.0002, 0.0001, -0.0001])
    strategy = BlackLitterman(
        view_generator=equilibrium_only,
        cov_estimator=sample_cov,
        lookback=252,
        prior_weights_method="equal",
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum {weights.sum():.6f}"
    assert (weights >= -1e-8).all(), "negative weight found"
    assert (weights <= 1.0 + 1e-8).all(), "weight > 1 found"


def test_equilibrium_only_sharpe_sensible():
    """BL(equilibrium_only) horse race Sharpe should be in a reasonable range."""
    from aiam.evaluation.performance import performance_stats
    from aiam.harness.horse_race import run_horse_race

    panel = make_panel(means=[0.0004, 0.0002, 0.0001, 0.00005], n_obs=700)
    strategy = BlackLitterman(
        view_generator=equilibrium_only,
        cov_estimator=sample_cov,
        lookback=252,
        prior_weights_method="equal",
    )
    result = run_horse_race(
        panel,
        strategy,
        start=panel.dates[300],
        end=panel.dates[-1],
    )
    sharpe = result["stats"]["sharpe_ratio"]
    assert -2.0 < sharpe < 10.0, f"Sharpe={sharpe:.3f} outside sensible range"


# ── 2. View dominance: tiny Omega → mu_post ≈ Q ─────────────────────────────

def test_view_dominance_posterior_close_to_q():
    """BL posterior mean collapses to Q when view uncertainty is negligible."""
    n = 5
    cov = np.eye(n) * 0.04          # 20% annual vol, uncorrelated
    w_eq = np.ones(n) / n
    delta = 2.5
    tau = 0.05

    pi = delta * cov @ w_eq
    Q = np.array([0.20, 0.15, 0.10, 0.05, 0.00])
    P = np.eye(n)
    Omega = np.eye(n) * 1e-10       # near-zero uncertainty → views dominate

    inv_term = np.linalg.inv(tau * cov)
    M_prec = inv_term + P.T @ np.linalg.inv(Omega) @ P
    M_cov = np.linalg.inv(M_prec)
    mu_post = M_cov @ (inv_term @ pi + P.T @ np.linalg.inv(Omega) @ Q)

    np.testing.assert_allclose(mu_post, Q, rtol=1e-3, atol=1e-4)


def test_view_dominance_via_strategy():
    """Strategy using a high-confidence view generator concentrates on top-view assets."""
    def _certain_views(returns, asof):
        n = returns.shape[1]
        # Strong view: T0 earns 30% annual, T1–T3 earn 0%
        Q = np.zeros(n)
        Q[0] = 0.30
        P = np.eye(n)
        Omega = np.diag(np.full(n, 1e-8))  # near-zero uncertainty
        return P, Q, Omega

    panel = make_panel(means=[0.0003, 0.0001, 0.0001, 0.0001])
    strategy = BlackLitterman(
        view_generator=_certain_views,
        cov_estimator=sample_cov,
        lookback=252,
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6
    # T0 should receive the dominant allocation
    assert weights["T0"] > 0.50, f"T0 weight={weights['T0']:.4f}, expected >0.50"


# ── 3. No look-ahead: predict_weights raises after train_until ───────────────

def test_no_lookahead_raises():
    panel = make_panel(means=[0.0002, 0.0002])
    strategy = BlackLitterman(view_generator=equilibrium_only, cov_estimator=sample_cov)
    train_until = panel.dates[300]
    strategy.fit(panel, train_until=train_until)

    future_date = panel.dates[400]
    with pytest.raises(ValueError, match="Look-ahead violation"):
        strategy.predict_weights(panel, asof=future_date)
