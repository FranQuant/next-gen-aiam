import logging

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.covariance import sample_cov
from aiam.estimators.mean import sample_mean
from aiam.strategy.max_sharpe import MaximumSharpe


def make_panel(means: list[float], n_obs: int = 400, seed: int = 42) -> Panel:
    """Synthetic Panel with n_obs business-day rows.

    Each ticker's daily return = mean + N(0, 0.01). Prices start at 100.
    """
    dates = pd.bdate_range("2018-01-01", periods=n_obs + 1)
    tickers = [f"T{i}" for i in range(len(means))]
    rng = np.random.default_rng(seed)

    price_matrix = np.empty((n_obs + 1, len(tickers)))
    price_matrix[0] = 100.0
    for i in range(1, n_obs + 1):
        daily_returns = np.array(means) + rng.normal(0.0, 0.01, size=len(means))
        price_matrix[i] = price_matrix[i - 1] * (1.0 + daily_returns)

    prices = pd.DataFrame(price_matrix, index=dates, columns=tickers)
    prices.index.name = "date"
    return Panel({"prices": prices})


def test_msr_concentrates_on_dominant_positive_mean():
    # T0 has a clearly positive daily mean; T1–T3 have negative means.
    panel = make_panel(means=[0.003, -0.003, -0.003, -0.003])
    strategy = MaximumSharpe(sample_cov, sample_mean)
    asof = panel.dates[-1]

    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert weights["T0"] > 0.50, (
        f"Expected T0 weight > 0.50, got {weights['T0']:.4f}"
    )


def test_msr_all_negative_mean_fallback_and_warning(caplog):
    panel = make_panel(means=[-0.002, -0.003, -0.001])
    strategy = MaximumSharpe(sample_cov, sample_mean)
    asof = panel.dates[-1]

    with caplog.at_level(logging.WARNING, logger="aiam.strategy.max_sharpe"):
        weights = strategy.predict_weights(panel, asof=asof)

    # EW fallback: weights are equal across valid columns and sum to 1
    assert abs(weights.sum() - 1.0) < 1e-6
    pos = weights[weights > 0]
    assert len(pos) > 0
    np.testing.assert_allclose(pos.values, pos.values.mean(), rtol=1e-5)

    # Warning was emitted
    assert "MSR fallback" in caplog.text, (
        f"Expected warning not found. Log output:\n{caplog.text}"
    )
