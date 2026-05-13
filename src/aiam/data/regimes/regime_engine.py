"""
Regime engine — pure logic, no I/O.

Lopez de Prado-style macro regime classification (8 indicators, 8 regimes).
Faithful port of paam_lab 19d weight_hrp / regime_engine logic.

Each indicator is mapped to a regime 0–7 based on:
  level  (high / low relative to rolling 5-year mean)
  change (rising / falling relative to `lookback` months ago)
  convexity (accelerating / decelerating)

The dominant regime is the mode across all 8 indicator regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_MAP: dict[str, int] = {
    "GDP_QoQ": 6,
    "VIX": 6,
    "SPX": 6,
    "CPI_MoM": 12,
    "UNEM": 12,
    "YC_10Y": 12,
    "YC_2Y": 12,
    "YC_STEP": 12,
}

IND_SHORT: dict[str, str] = {
    "GDP_QoQ": "GDP",
    "CPI_MoM": "CPI",
    "UNEM": "UNEM",
    "YC_10Y": "YC10",
    "YC_2Y": "YC2",
    "YC_STEP": "YCSTEP",
    "VIX": "VIX",
    "SPX": "SPX",
}


def compute_features(series: pd.Series, lookback: int) -> pd.DataFrame:
    name = series.name
    lvl = series.rolling(3, min_periods=1).mean()
    chg = lvl - lvl.shift(lookback)
    conv = (lvl + lvl.shift(lookback)) / 2 - lvl.shift(lookback // 2)
    return pd.DataFrame({name: series, "lvl": lvl, "chg": chg, "conv": conv})


def get_regime(
    row: pd.Series,
    col_lvl: str,
    col_chg: str,
    col_conv: str,
    mean_lvl: float,
    prev_regime=None,
    eps_chg: float = 0.001,
    eps_conv: float = 0.001,
) -> int | float:
    fallback = prev_regime if pd.notna(prev_regime) else np.nan

    lvl_high = row[col_lvl] >= mean_lvl
    chg_pos = row[col_chg] > eps_chg
    chg_neg = row[col_chg] < -eps_chg
    conv_pos = row[col_conv] > eps_conv
    conv_neg = row[col_conv] < -eps_conv

    # NaN inputs produce all-False flags → fallback
    if not (chg_pos or chg_neg) or not (conv_pos or conv_neg):
        return fallback

    if lvl_high and chg_pos and conv_pos:
        return 0
    elif lvl_high and chg_pos and conv_neg:
        return 1
    elif lvl_high and chg_neg and conv_neg:
        return 2
    elif not lvl_high and chg_pos and conv_neg:
        return 3
    elif lvl_high and chg_neg and conv_pos:
        return 4
    elif not lvl_high and chg_neg and conv_pos:
        return 5
    elif not lvl_high and chg_pos and conv_pos:
        return 6
    elif not lvl_high and chg_neg and conv_neg:
        return 7
    return fallback


def build_regime_signals(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  df_macro with 8 columns (GDP_QoQ, CPI_MoM, UNEM, YC_10Y, YC_2Y,
            YC_STEP, VIX, SPX), monthly DatetimeIndex.
    Output: DataFrame with 9 columns — regime_<IND> for each indicator +
            dominant_regime (mode across the 8 indicator regimes).

    All computations are strictly backward-looking: rolling windows, shift(),
    and the prev_regime stateful walk never read future data.
    Expected returns are merged in separately by the caller (SWITCH strategy).
    """
    # Build feature DataFrames for each indicator
    df_features: dict[str, pd.DataFrame] = {}
    for col in LOOKBACK_MAP:
        if col not in df_macro.columns:
            continue
        df_features[col] = compute_features(df_macro[col], LOOKBACK_MAP[col])

    regime_series: dict[str, pd.Series] = {}
    for col, feat in df_features.items():
        short = IND_SHORT[col]
        regimes: list[int | float] = []
        prev_regime: int | float = np.nan

        lvl_series = feat["lvl"]
        for i, (idx, row) in enumerate(feat.iterrows()):
            # 5-year rolling mean of lvl: last 60 observations ≤ current
            mean_lvl = float(lvl_series.iloc[max(0, i - 59) : i + 1].mean())
            r = get_regime(row, "lvl", "chg", "conv", mean_lvl, prev_regime)
            regimes.append(r)
            if pd.notna(r):
                prev_regime = r

        regime_series[f"regime_{short}"] = pd.Series(regimes, index=feat.index)

    df_regimes_rb = pd.DataFrame(regime_series)
    dominant_regime = df_regimes_rb.mode(axis=1)[0]
    return pd.concat(
        [df_regimes_rb, dominant_regime.rename("dominant_regime")], axis=1
    )
