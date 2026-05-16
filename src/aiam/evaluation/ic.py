"""Cross-sectional information coefficient diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = 5,
) -> pd.Series:
    """Cross-sectional rank/Pearson correlation between signal and forward returns at each date.

    Returns a pd.Series indexed by date. NaN if fewer than `min_assets` non-null pairs.
    method: 'spearman' (default, rank-based) or 'pearson' (linear).
    """
    sig, fwd = signal.align(forward_ret, join="inner")
    mask = sig.notna() & fwd.notna()

    if method == "spearman":
        s = sig.rank(axis=1)
        f = fwd.rank(axis=1)
    else:
        s, f = sig, fwd

    ic_vals: dict[pd.Timestamp, float] = {}
    for date in s.index:
        m = mask.loc[date]
        if m.sum() < min_assets:
            ic_vals[date] = np.nan
            continue
        ic_vals[date] = s.loc[date, m].corr(f.loc[date, m])

    return pd.Series(ic_vals, name="IC")


def ic_summary(ic_series: pd.Series) -> dict:
    """Standard IC diagnostics: mean, std, t-stat, hit rate, IR, n_obs."""
    clean = ic_series.dropna()
    n = len(clean)
    mean = clean.mean()
    std = clean.std()
    return {
        "mean": mean,
        "std": std,
        "t_stat": mean / (std / np.sqrt(n)) if n > 0 and std > 0 else np.nan,
        "hit_rate": float((clean > 0).mean()),
        "ir": mean / std if std > 0 else np.nan,
        "n_obs": n,
    }
