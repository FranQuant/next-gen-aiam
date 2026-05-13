from __future__ import annotations

import numpy as np
import pandas as pd


def regime_conditional_performance(
    portfolio_returns: dict[str, pd.Series],
    dominant_regime: pd.Series,  # monthly index
    n_regimes: int = 8,
    min_days: int = 252,
    rf: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    For each strategy × regime, compute annualized performance metrics.
    Returns dict with keys: 'sharpe', 'ann_return', 'ann_vol', 'max_drawdown', 'n_days'.
    Each value is a DataFrame indexed by strategy, columns = regime int 0..n_regimes-1.
    """
    strategies = list(portfolio_returns.keys())
    regimes = list(range(n_regimes))

    # Union of all daily trading-date indices across strategies
    all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    for s in portfolio_returns.values():
        all_dates = all_dates.union(s.index)

    # Forward-fill monthly dominant_regime onto daily dates
    combined_idx = all_dates.union(dominant_regime.index).sort_values()
    regime_daily = dominant_regime.reindex(combined_idx).ffill().reindex(all_dates)

    tables = {
        k: pd.DataFrame(np.nan, index=strategies, columns=regimes, dtype=float)
        for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days")
    }

    for name, returns in portfolio_returns.items():
        reg_aligned = regime_daily.reindex(returns.index)

        for r in regimes:
            subset = returns[reg_aligned == r]
            n = len(subset)
            tables["n_days"].loc[name, r] = float(n)

            if n < 3:
                continue

            ann_ret = subset.mean() * trading_days_per_year
            ann_v = subset.std() * np.sqrt(trading_days_per_year)
            tables["ann_return"].loc[name, r] = ann_ret
            tables["ann_vol"].loc[name, r] = ann_v
            tables["sharpe"].loc[name, r] = (ann_ret - rf) / ann_v if ann_v > 0 else np.nan

            cum = (1 + subset).cumprod()
            peak = cum.cummax()
            tables["max_drawdown"].loc[name, r] = float(((cum - peak) / peak).min())

    return tables
