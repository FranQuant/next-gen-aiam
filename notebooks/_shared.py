"""Shared infrastructure for all paper notebooks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent

FAMILY_COLORS = {
    "Classical MV":    "#1f77b4",
    "Constrained MV":  "#17becf",
    "Diversification": "#2ca02c",
    "Regime Switch":   "#9467bd",
    "TSMOM":           "#ff7f0e",
    "Black-Litterman": "#d62728",
    "Factor":          "#8c564b",
    "Long-Short":      "#bcbd22",
    "EW (benchmark)":  "#333333",
}

FAMILY_MAP = {
    "EW":                  "Classical MV",
    "GMV(sample)":         "Classical MV",
    "GMV(ledoit_wolf)":    "Classical MV",
    "GMV(oas)":            "Classical MV",
    "MSR(sample)":         "Classical MV",
    "MSR(ledoit_wolf)":    "Classical MV",
    "MDP(sample)":         "Diversification",
    "MDP(ledoit_wolf)":    "Diversification",
    "RP(sample)":          "Diversification",
    "RP(ledoit_wolf)":     "Diversification",
    "HRP(sample)":         "Diversification",
    "HRP(ledoit_wolf)":    "Diversification",
    "SWITCH(sample)":      "Regime Switch",
    "SWITCH(ledoit_wolf)": "Regime Switch",
    "TSMOM(12m)":          "TS Momentum",
    "TSMOM(6m)":           "TS Momentum",
    "BL-Eq(sample)":       "Black-Litterman",
    "BL-Eq(LW)":           "Black-Litterman",
    "BL-Mom(LW)":          "Black-Litterman",
    "BL-Rev(LW)":          "Black-Litterman",
    "FF3-Mom":             "Factor",
    "FF3-LowVol":          "Factor",
    "FF3-Quality":         "Factor",
    "FF3-Multi":           "Factor",
    "MSR_C(ledoit_wolf)":  "Constrained MV",
    "MSR_C(sample)":       "Constrained MV",
    "MVO_C(ledoit_wolf)":  "Constrained MV",
    "MVO_C(sample)":       "Constrained MV",
    "TSMOM-LS(12m)":       "Long-Short",
    "BL-Mom-LS(LW)":       "Long-Short",
    "FF3-Mom-LS":          "Long-Short",
}

def display_name(col):
    return col.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)")

DISPLAY = {col: display_name(col) for col in FAMILY_MAP}

BASE_24_COLS = [
    "EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)",
    "MSR(sample)", "MSR(ledoit_wolf)",
    "MDP(sample)", "MDP(ledoit_wolf)",
    "RP(sample)", "RP(ledoit_wolf)",
    "HRP(sample)", "HRP(ledoit_wolf)",
    "SWITCH(sample)", "SWITCH(ledoit_wolf)",
    "TSMOM(12m)", "TSMOM(6m)",
    "BL-Eq(sample)", "BL-Eq(LW)", "BL-Mom(LW)", "BL-Rev(LW)",
    "FF3-Mom", "FF3-LowVol", "FF3-Quality", "FF3-Multi",
]

CRISES = [
    (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), "GFC"),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), "COVID"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31"), "Rate Shock"),
]

COST_BPS = 10.0
TRADING_DAYS = 252
WEIGHTS_SUFFIX = "29assets_2003_2026"

FAMILY_ORDER = [
    # Classical MV (6)
    "EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)", "MSR(sample)", "MSR(ledoit_wolf)",
    # Constrained MV (4)
    "MSR_C(ledoit_wolf)", "MSR_C(sample)", "MVO_C(ledoit_wolf)", "MVO_C(sample)",
    # Diversification (6)
    "MDP(sample)", "MDP(ledoit_wolf)", "RP(sample)", "RP(ledoit_wolf)", "HRP(sample)", "HRP(ledoit_wolf)",
    # Regime Switch (2)
    "SWITCH(sample)", "SWITCH(ledoit_wolf)",
    # TS Momentum (2)
    "TSMOM(12m)", "TSMOM(6m)",
    # Black-Litterman (4)
    "BL-Eq(sample)", "BL-Eq(LW)", "BL-Mom(LW)", "BL-Rev(LW)",
    # Factor (4)
    "FF3-Mom", "FF3-LowVol", "FF3-Quality", "FF3-Multi",
    # Long-Short (3)
    "TSMOM-LS(12m)", "BL-Mom-LS(LW)", "FF3-Mom-LS",
]

def load_base_returns():
    return pd.read_parquet(ROOT / "data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet")

def load_vmp_returns():
    return pd.read_parquet(ROOT / "data/cache/portfolio_returns/31strategies_vmp_29assets_2003_2026.parquet")

def load_regime_signals():
    return pd.read_parquet(ROOT / "data/cache/regime_signals_2003_2026.parquet")

def load_weights_cache(strategy_stem):
    p = ROOT / f"data/cache/portfolio_weights/{strategy_stem}_{WEIGHTS_SUFFIX}.parquet"
    return pd.read_parquet(p) if p.exists() else None

def build_switch_v2a(base, sw_oos):
    s = sw_oos["SWITCH_v2a_train_only"].reindex(base.index)
    s.name = "SWITCH(v2a)"
    return s

def apply_vmp_overlay(base_ret, target_vol=None, lookback=21, clip=(0.25, 1.5)):
    r = base_ret.dropna()
    if target_vol is None:
        target_vol = r.std() * np.sqrt(TRADING_DAYS)
    roll_vol = r.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    exp = (target_vol / roll_vol.shift(1)).clip(*clip)
    return (r * exp).rename(f"VMP({r.name})")

def ann_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(TRADING_DAYS)

def ann_return(r):
    r = r.dropna()
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1

def ann_vol(r):
    return r.dropna().std() * np.sqrt(TRADING_DAYS)

def max_drawdown(r):
    cum = (1 + r.dropna()).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def load_panel_for_ml(horizon: int = 21) -> "pd.DataFrame":
    """Build long-form (Date, Asset) MultiIndex panel ready for ML training.

    Returns DataFrame with columns: mom_252, mom_21, vol_60, 7 asset-class one-hots,
    target_21d. Rows with NaN targets dropped. Bridge for Session 2b notebook.
    """
    from aiam.features.technical import momentum, volatility
    from aiam.features.technical import forward_returns as fwd_ret
    from aiam.features.asset_class import asset_class_one_hot

    ret = pd.read_parquet(ROOT / "data/cache/returns_29_2003_2026.parquet")
    ret.index = pd.to_datetime(ret.index)
    ret.index.name = "Date"
    ret.columns.name = "Asset"

    frames = {
        "mom_252":     momentum(ret, 252),
        "mom_21":      momentum(ret, 21),
        "vol_60":      volatility(ret, 60),
        "target_21d":  fwd_ret(ret, horizon),
    }
    panel = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    panel.index.names = ["Date", "Asset"]

    oh = asset_class_one_hot(ret.columns.tolist())
    panel = panel.join(oh, on="Asset")
    return panel.dropna(subset=["target_21d"])


def build_all_stats(base, vmp):
    records = []
    for col in base.columns:
        r = base[col]
        fam = FAMILY_MAP.get(col, "Classical MV")
        records.append({
            "strategy": col,
            "display": display_name(col),
            "family": fam,
            "is_vmp": False,
            "sharpe": ann_sharpe(r),
            "ann_ret": ann_return(r),
            "ann_vol": ann_vol(r),
            "max_dd": max_drawdown(r),
        })
        vcol = f"VMP({col})"
        if vcol in vmp.columns:
            vr = vmp[vcol]
            records.append({
                "strategy": vcol,
                "display": display_name(vcol),
                "family": fam,
                "is_vmp": True,
                "sharpe": ann_sharpe(vr),
                "ann_ret": ann_return(vr),
                "ann_vol": ann_vol(vr),
                "max_dd": max_drawdown(vr),
            })
    return pd.DataFrame(records)
