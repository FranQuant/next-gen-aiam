"""End-to-end: LLMViewGenerator drops into BlackLitterman without changes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.covariance import sample_cov
from aiam.llm.client import MockClient
from aiam.llm.views import LLMViewGenerator
from aiam.strategy.black_litterman import BlackLitterman


def _make_panel(
    tickers=("SPY", "IEF", "GLD"),
    n_obs: int = 600,
    seed: int = 7,
) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_obs + 1)
    prices = np.empty((n_obs + 1, len(tickers)))
    prices[0] = 100.0
    for i in range(1, n_obs + 1):
        ret = rng.normal(0.0003, 0.01, size=len(tickers))
        prices[i] = prices[i - 1] * (1.0 + ret)
    df = pd.DataFrame(prices, index=dates, columns=list(tickers))
    df.index.name = "date"
    return Panel({"prices": df})


def _mock_response(tickers: list[str]) -> str:
    views = [
        {"asset": t, "expected_excess_return": 0.05 * (i + 1) * 0.1, "confidence": 0.6}
        for i, t in enumerate(tickers)
    ]
    return json.dumps({"views": views, "rationale": "mock evidence"})


# ── Drop-in integration ──────────────────────────────────────────────────────

def test_bl_with_llm_generator_weights_sum_to_one():
    panel = _make_panel()
    tickers = list(panel.universe)

    mock = MockClient([_mock_response(tickers)])
    gen = LLMViewGenerator(mock)

    strategy = BlackLitterman(
        view_generator=gen,
        cov_estimator=sample_cov,
        lookback=252,
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6, f"weights sum = {weights.sum()}"


def test_bl_with_llm_generator_long_only():
    panel = _make_panel()
    tickers = list(panel.universe)

    mock = MockClient([_mock_response(tickers)])
    gen = LLMViewGenerator(mock)

    strategy = BlackLitterman(
        view_generator=gen,
        cov_estimator=sample_cov,
        lookback=252,
        long_only=True,
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    assert (weights >= -1e-8).all(), "negative weight found"


def test_bl_fallback_on_llm_failure():
    """When LLM returns garbage, generator returns empty views → BL uses equilibrium."""
    panel = _make_panel()

    mock = MockClient(["not json at all"])
    gen = LLMViewGenerator(mock)

    strategy = BlackLitterman(
        view_generator=gen,
        cov_estimator=sample_cov,
        lookback=252,
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    # Should still produce valid weights (equilibrium fallback)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_bl_with_partial_llm_views():
    """LLM gives views on a subset of the universe — BL still works."""
    panel = _make_panel(tickers=("SPY", "IEF", "GLD"))

    resp = json.dumps({
        "views": [
            {"asset": "SPY", "expected_excess_return": 0.10, "confidence": 0.7},
        ]
    })
    mock = MockClient([resp])
    gen = LLMViewGenerator(mock)

    strategy = BlackLitterman(
        view_generator=gen,
        cov_estimator=sample_cov,
        lookback=252,
    )
    asof = panel.dates[-1]
    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-8).all()
