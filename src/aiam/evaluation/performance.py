from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_stats(returns: pd.Series, rf: float = 0.0) -> dict:
    excess = returns - rf / TRADING_DAYS

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    ann_excess_mean = excess.mean() * TRADING_DAYS
    ann_excess_vol = excess.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_excess_mean / ann_excess_vol if ann_excess_vol != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    monthly = (1 + returns).resample("ME").prod() - 1
    hit_ratio = float((monthly > 0).sum() / len(monthly)) if len(monthly) > 0 else np.nan

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "hit_ratio": hit_ratio,
    }
