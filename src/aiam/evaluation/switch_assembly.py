from __future__ import annotations

import pandas as pd


def assemble_switch_returns(
    portfolio_returns: pd.DataFrame,   # wide: dates × strategy_names
    dominant_regime: pd.Series,        # monthly index
    rule: dict[int, str],              # regime int → strategy name (must be column in portfolio_returns)
    default_strategy: str,
) -> pd.Series:
    """
    Compute SWITCH portfolio returns by dispatching cached series day-by-day.
    - Forward-fill dominant_regime to the daily index of portfolio_returns
    - For each date t: look up regime r, dispatch to rule[r] (or default_strategy if missing)
    - Return the resulting daily portfolio return series
    """
    daily_idx = portfolio_returns.index

    # portfolio_returns is indexed by the return date t1; rebalancing happened at
    # the previous business-day t = t1 - BDay(1).  The horse-race loop uses
    # pd.bdate_range (Mon-Fri, no holiday exclusion), so market holidays such as
    # New Year's observance are included as rebalancing days even though they have
    # no price data.  Using BDay(1) rather than the previous *return* date correctly
    # identifies these holiday rebalancing dates, which is also why Saturday month-end
    # regime labels are handled correctly: Friday's panel slice can't see Saturday.
    reb_dates = daily_idx - pd.offsets.BDay(1)
    combined_idx = reb_dates.union(dominant_regime.index).sort_values()
    regime_at_reb = dominant_regime.reindex(combined_idx).ffill().reindex(reb_dates)
    regime_daily = pd.Series(regime_at_reb.values, index=daily_idx)

    # Build a per-date strategy name Series, defaulting to default_strategy
    strategy_name = pd.Series(default_strategy, index=daily_idx)
    for regime_val, strat_name in rule.items():
        mask = regime_daily == regime_val
        strategy_name[mask] = strat_name

    # Vectorized lookup grouped by strategy name
    result = pd.Series(dtype=float, index=daily_idx)
    for name in strategy_name.unique():
        mask = strategy_name == name
        result[mask] = portfolio_returns.loc[mask, name].values

    return result
