from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per date: 0.5 * sum(|w[t] - w[t-1]|).

    Returns a daily Series of turnover ratios in [0, 1].
    Ignores price drift between rebalances (uses raw weight changes) —
    standard simplification; slight overestimate of true turnover.
    The first row is NaN (no prior observation to diff against).
    """
    diff = weights.fillna(0.0).diff()
    result = (diff.abs().sum(axis=1) / 2).rename("turnover")
    result.iloc[0] = np.nan
    return result


def apply_costs(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """net_return[t] = gross_returns[t] - turnover[t] * (cost_bps / 10000).

    The weights index (decision dates) leads the returns index (next-day return
    dates) by one business day.  turnover is shifted forward one position so that
    the cost of rebalancing on date t is deducted from the return earned on t+1.
    """
    turnover = compute_turnover(weights)
    cost_rate = cost_bps / 10_000
    # Positional shift: cost incurred at decision date t → deducted at return date t+1
    cost_at_return = turnover.shift(1).reindex(gross_returns.index, fill_value=0.0).fillna(0.0)
    return (gross_returns - cost_at_return * cost_rate).rename(gross_returns.name)
