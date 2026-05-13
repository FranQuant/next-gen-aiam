from __future__ import annotations

import numpy as np
import pandas as pd

_SQRT252 = np.sqrt(252)


def momentum_signal(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """12-1 momentum: annualized 12m return minus annualized most-recent 1m return.

    Subtracting the skip-month return removes short-term reversal bias (Jegadeesh 1990).
    """
    trailing_12m = returns.iloc[-lookback:]
    trailing_1m = returns.iloc[-skip:]
    ret_12m = ((1 + trailing_12m).prod() - 1) * (252 / max(len(trailing_12m), 1))
    ret_1m = ((1 + trailing_1m).prod() - 1) * (252 / max(len(trailing_1m), 1))
    return (ret_12m - ret_1m).rename(asof)


def low_vol_signal(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookback: int = 126,
) -> pd.Series:
    """Negative trailing-6m realized vol (annual). Higher signal = lower vol."""
    vol = returns.iloc[-lookback:].std() * _SQRT252
    return (-vol).rename(asof)


def quality_signal(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookback: int = 756,
) -> pd.Series:
    """Trailing-3y per-asset Sharpe ratio (mean/std × √252).

    Assets with high risk-adjusted return score high. Assets with near-zero
    std get NaN, which FactorPortfolio drops before ranking.
    """
    trailing = returns.iloc[-lookback:]
    std = trailing.std()
    signal = trailing.mean() / std.replace(0, np.nan) * _SQRT252
    return signal.rename(asof)
