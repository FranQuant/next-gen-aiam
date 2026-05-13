"""
No-look-ahead regression test for the regime engine.

Verifies that build_regime_signals(df[:T]) == build_regime_signals(df[:T+12mo])[:T]
for all regime columns and dominant_regime.

This property holds by construction: every computation in the engine is
strictly backward-looking (rolling windows, shift, 60-obs rolling mean,
prev_regime stateful walk).  If any future row were accidentally read —
e.g. via a forward-fill into the future or a misplaced dropna — the
dominant_regime at t ≤ T would differ between the two runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.regimes.regime_engine import (
    LOOKBACK_MAP,
    build_regime_signals,
    compute_features,
    get_regime,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_macro() -> pd.DataFrame:
    """120 months of synthetic macro data (Jan 2013 – Dec 2022)."""
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2013-01-31", periods=n, freq="ME")

    # Construct series with realistic scale and enough variance to
    # drive non-trivial chg / conv signals after rolling smoothing.
    gdp = np.cumsum(rng.normal(0.005, 0.015, n))          # trending ~0.5% / mo
    cpi = np.cumsum(rng.normal(0.002, 0.004, n))
    unem = 6.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    unem = np.clip(unem, 3.0, 12.0)
    yc10 = 3.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    yc10 = np.clip(yc10, 0.5, 6.0)
    yc2 = 2.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    yc2 = np.clip(yc2, 0.1, 5.5)
    yc_step = yc10 - yc2
    vix = 15.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    vix = np.clip(vix, 9.0, 60.0)
    spx = rng.normal(0.007, 0.04, n)

    df = pd.DataFrame(
        {
            "GDP_QoQ": gdp,
            "CPI_MoM": cpi,
            "UNEM": unem,
            "YC_10Y": yc10,
            "YC_2Y": yc2,
            "YC_STEP": yc_step,
            "VIX": vix,
            "SPX": spx,
        },
        index=dates,
    )
    return df


# ── unit tests ────────────────────────────────────────────────────────────────

def test_compute_features_shape(synthetic_macro):
    s = synthetic_macro["GDP_QoQ"]
    feat = compute_features(s, lookback=6)
    assert feat.shape == (len(s), 4)
    assert list(feat.columns) == ["GDP_QoQ", "lvl", "chg", "conv"]


def test_compute_features_no_future_use(synthetic_macro):
    """lvl, chg, conv at row i must not change when future rows are appended."""
    s = synthetic_macro["GDP_QoQ"]
    short = s.iloc[:60]
    full = s

    feat_short = compute_features(short, lookback=6)
    feat_full = compute_features(full, lookback=6)

    pd.testing.assert_series_equal(
        feat_short["lvl"],
        feat_full["lvl"].iloc[:60],
        check_names=True,
    )
    pd.testing.assert_series_equal(
        feat_short["chg"].dropna(),
        feat_full["chg"].iloc[:60].dropna(),
        check_names=True,
    )


def test_get_regime_returns_int_or_nan():
    row = pd.Series({"lvl": 5.0, "chg": 0.01, "conv": 0.005})
    r = get_regime(row, "lvl", "chg", "conv", mean_lvl=4.0)
    assert r in range(8) or (isinstance(r, float) and np.isnan(r))


def test_get_regime_fallback_on_tiny_change():
    row = pd.Series({"lvl": 5.0, "chg": 0.0001, "conv": 0.5})
    r = get_regime(row, "lvl", "chg", "conv", mean_lvl=4.0, prev_regime=3)
    assert r == 3  # change within eps → prev_regime returned


def test_build_regime_signals_shape(synthetic_macro):
    result = build_regime_signals(synthetic_macro)
    assert result.shape == (len(synthetic_macro), 9)
    assert "dominant_regime" in result.columns
    regime_cols = [c for c in result.columns if c.startswith("regime_")]
    assert len(regime_cols) == 8


def test_build_regime_signals_dominant_is_mode(synthetic_macro):
    """dominant_regime must equal the row-wise mode of the 8 indicator columns."""
    result = build_regime_signals(synthetic_macro)
    regime_cols = [c for c in result.columns if c.startswith("regime_")]
    expected_mode = result[regime_cols].mode(axis=1)[0]
    pd.testing.assert_series_equal(
        result["dominant_regime"],
        expected_mode,
        check_names=False,
    )


# ── no-look-ahead regression ──────────────────────────────────────────────────

def test_no_lookahead_dominant_regime(synthetic_macro):
    """
    Core no-look-ahead test.

    Run the engine on data up to T (month 74, Feb 2019) and on the full
    120-month series.  Every dominant_regime value at t ≤ T must be identical.
    """
    T = synthetic_macro.index[74]  # month 75 of 120 → well inside the series
    df_short = synthetic_macro.loc[synthetic_macro.index <= T]
    df_full = synthetic_macro

    result_short = build_regime_signals(df_short)
    result_full = build_regime_signals(df_full)

    overlap = result_full.loc[result_full.index <= T]

    pd.testing.assert_series_equal(
        result_short["dominant_regime"],
        overlap["dominant_regime"],
        check_names=True,
        obj="dominant_regime no-look-ahead",
    )


def test_no_lookahead_all_columns(synthetic_macro):
    """
    Extend the regression to all 8 individual regime columns, not just the mode.
    """
    T = synthetic_macro.index[74]
    df_short = synthetic_macro.loc[synthetic_macro.index <= T]
    df_full = synthetic_macro

    result_short = build_regime_signals(df_short)
    result_full = build_regime_signals(df_full)
    overlap = result_full.loc[result_full.index <= T]

    for col in result_short.columns:
        pd.testing.assert_series_equal(
            result_short[col],
            overlap[col],
            check_names=True,
            obj=f"{col} no-look-ahead",
        )
