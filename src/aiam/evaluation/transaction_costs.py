from __future__ import annotations

import numpy as np
import pandas as pd

# Per-asset round-trip cost in bps.
# Source: Hilpisch (2026) Ch 11.3 representative practitioner costs.
STRATIFIED_COST_BPS: dict[str, float] = {
    # US large-cap single stocks
    "AAPL.US": 5.0, "MSFT.US": 5.0, "GOOGL.US": 5.0, "NVDA.US": 5.0,
    "JPM.US":  5.0, "JNJ.US":  5.0, "XOM.US":  5.0, "WMT.US":  5.0,
    # US sector and broad ETFs
    "XLK.US": 3.0, "XLF.US": 3.0, "XLE.US": 3.0, "XLV.US": 3.0,
    "XLP.US": 3.0, "XLU.US": 3.0, "SPY.US": 3.0, "IWM.US": 3.0,
    # International equity ETFs
    "EFA.US": 5.0, "EEM.US": 5.0, "FXI.US": 5.0,
    # Fixed-income ETFs
    "SHY.US": 2.0, "IEF.US": 2.0, "TLT.US": 2.0, "AGG.US": 2.0, "HYG.US": 2.0,
    # Commodities and FX
    "GLD.US": 5.0, "SLV.US": 5.0, "DBC.US": 5.0, "USO.US": 5.0,
    "EURUSD.FOREX": 5.0,
    # Crypto
    "BTC-USD.CC": 30.0,
}


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


def apply_stratified_costs(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    cost_vector: dict[str, float] = STRATIFIED_COST_BPS,
) -> pd.Series:
    """net_return[t] = gross_returns[t] - sum_i( |Δw_i| / 2 * c_i / 10000 ).

    Per-asset one-way turnover weighted by per-asset cost in bps.
    Shift aligns decision-date rebalancing cost to next-day return date.
    """
    rates = pd.Series(cost_vector) / 10_000
    w = weights.fillna(0.0)
    aligned_rates = rates.reindex(w.columns, fill_value=0.0)
    per_asset_to = w.diff().abs() / 2
    per_asset_to.iloc[0] = np.nan
    daily_cost = per_asset_to.mul(aligned_rates, axis=1).sum(axis=1)
    cost_at_return = daily_cost.shift(1).reindex(gross_returns.index, fill_value=0.0).fillna(0.0)
    return (gross_returns - cost_at_return).rename(gross_returns.name)
