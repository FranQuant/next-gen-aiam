from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.factor_signals import low_vol_signal, momentum_signal, quality_signal
from aiam.strategy.factor_portfolio import FactorPortfolio, MultiFactorPortfolio


def make_returns(data: dict[str, list[float]], start: str = "2018-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=max(len(v) for v in data.values()))
    return pd.DataFrame({k: v for k, v in data.items()}, index=dates)


def make_panel(returns: pd.DataFrame) -> Panel:
    price_matrix = (1 + returns).cumprod()
    price_matrix.iloc[0] = 100.0
    prices = price_matrix * 100.0
    prices.index.name = "date"
    return Panel({"prices": prices})


# ── Signal: momentum rank order ──────────────────────────────────────────────

def test_momentum_signal_rank_order():
    """T0: strong 12m-1m trend, flat 1m → high momentum. T4: negative trend + reversal → lowest.

    Using distinct 12m-1m and 1m windows avoids the geometric-compounding artifact
    where uniform negative daily returns produce a spuriously positive signal.
    """
    lookback, skip, n = 252, 21, 300
    preamble = n - lookback  # days before the 12m window

    def make_asset(r_trend: float, r_recent: float) -> list[float]:
        return [0.0] * preamble + [r_trend] * (lookback - skip) + [r_recent] * skip

    returns = make_returns({
        "T0": make_asset(0.005, 0.000),   # strong trend, no reversal → highest
        "T1": make_asset(0.002, 0.000),   # moderate trend, no reversal
        "T2": make_asset(0.000, 0.000),   # flat → signal ≈ 0
        "T3": make_asset(-0.003, 0.000),  # negative trend, flat recent → clearly negative
        "T4": make_asset(-0.003, 0.005),  # negative trend + strong reversal → lowest
    })
    asof = returns.index[-1]
    signal = momentum_signal(returns, asof, lookback=lookback, skip=skip)

    assert signal["T0"] > signal["T1"] > signal["T2"] > signal["T3"] > signal["T4"], (
        f"Unexpected rank order: {signal.to_dict()}"
    )


def test_low_vol_signal_rank_order():
    """Low-vol assets score higher (signal = -vol)."""
    n = 200
    returns = make_returns({
        "HighVol": np.random.default_rng(0).normal(0, 0.03, n).tolist(),
        "MidVol":  np.random.default_rng(1).normal(0, 0.01, n).tolist(),
        "LowVol":  np.random.default_rng(2).normal(0, 0.003, n).tolist(),
    })
    asof = returns.index[-1]
    signal = low_vol_signal(returns, asof, lookback=126)

    assert signal["LowVol"] > signal["MidVol"] > signal["HighVol"]


def test_quality_signal_rank_order():
    """High per-asset Sharpe → high quality signal."""
    n = 800
    rng = np.random.default_rng(42)
    returns = make_returns({
        "T_high": (rng.normal(0.003, 0.005, n)).tolist(),   # high mean, low vol
        "T_mid":  (rng.normal(0.001, 0.010, n)).tolist(),
        "T_low":  (rng.normal(-0.001, 0.015, n)).tolist(),  # negative mean
    })
    asof = returns.index[-1]
    signal = quality_signal(returns, asof, lookback=756)

    assert signal["T_high"] > signal["T_mid"] > signal["T_low"]


# ── Portfolio: equal-vol assets → equal weights over selected ────────────────

def test_equal_vol_gives_equal_weights():
    """All assets have the same vol → inverse-vol weighting collapses to equal weight."""
    n = 300
    rng = np.random.default_rng(7)
    # All assets: zero mean, same daily vol = 1%
    cols = [f"T{i}" for i in range(6)]
    data = {c: rng.normal(0.0, 0.01, n).tolist() for c in cols}
    returns = make_returns(data)
    panel = make_panel(returns)
    asof = returns.index[-1]

    strategy = FactorPortfolio(
        signal_fn=momentum_signal,
        lookback=252,
        top_fraction=1 / 3,
        weighting="inverse_vol",
    )
    weights = strategy.predict_weights(panel, asof=asof)
    selected = weights[weights > 1e-8]

    # All selected assets should have equal weight (equal vol → equal inv-vol)
    np.testing.assert_allclose(
        selected.values,
        selected.values.mean(),
        rtol=0.15,  # allow 15% relative tolerance (sample vol fluctuates)
        err_msg="Expected near-equal weights for equal-vol assets",
    )


# ── Portfolio: top_fraction=1.0 → full universe, inverse-vol weighted ────────

def test_top_fraction_one_uses_full_universe():
    """top_fraction=1.0 selects all valid assets; weights should be all-positive and sum to 1."""
    n = 300
    rng = np.random.default_rng(9)
    cols = [f"T{i}" for i in range(6)]
    data = {c: rng.normal(0.001 * i, 0.01, n).tolist() for i, c in enumerate(cols)}
    returns = make_returns(data)
    panel = make_panel(returns)
    asof = returns.index[-1]

    strategy = FactorPortfolio(
        signal_fn=low_vol_signal,
        lookback=252,
        top_fraction=1.0,
        weighting="inverse_vol",
    )
    weights = strategy.predict_weights(panel, asof=asof)

    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()
    # With top_fraction=1.0, all valid assets should receive positive weight
    assert (weights > 1e-8).sum() == len(cols)


# ── MultiFactorPortfolio: weights sum to 1 and lie in [0,1] ─────────────────

def test_multi_factor_weights_valid():
    n = 800
    rng = np.random.default_rng(13)
    cols = [f"A{i}" for i in range(5)]
    data = {c: rng.normal(0.001, 0.01, n).tolist() for c in cols}
    returns = make_returns(data)
    panel = make_panel(returns)
    asof = returns.index[-1]

    mom = FactorPortfolio(signal_fn=momentum_signal, lookback=252)
    lv = FactorPortfolio(signal_fn=low_vol_signal, lookback=252)
    qu = FactorPortfolio(signal_fn=quality_signal, lookback=756)
    multi = MultiFactorPortfolio([mom, lv, qu])

    weights = multi.predict_weights(panel, asof=asof)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -1e-8).all()


# ── No look-ahead guard ───────────────────────────────────────────────────────

def test_no_lookahead_raises():
    n = 400
    returns = make_returns({"T0": [0.001] * n, "T1": [0.002] * n})
    panel = make_panel(returns)

    strategy = FactorPortfolio(signal_fn=low_vol_signal, lookback=126)
    train_until = returns.index[200]
    strategy.fit(panel, train_until=train_until)

    with pytest.raises(ValueError, match="Look-ahead violation"):
        strategy.predict_weights(panel, asof=returns.index[300])
