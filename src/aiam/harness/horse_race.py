from __future__ import annotations

import pandas as pd

from aiam.data.panel import Panel
from aiam.evaluation.performance import performance_stats
from aiam.strategy.base import Strategy


def run_horse_race(
    panel: Panel,
    strategy: Strategy,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Filter to weekdays so weekend calendar rows don't inject artificial zeros
    # for US equities or double-count crypto/FX weekend moves.
    all_returns = panel.data["prices"].pct_change()
    returns = all_returns[all_returns.index.dayofweek < 5]

    trading_days = pd.bdate_range(start=start, end=end)

    weights_records: dict[pd.Timestamp, pd.Series] = {}
    portfolio_returns: dict[pd.Timestamp, float] = {}

    for t in trading_days:
        weights_t = strategy.predict_weights(panel, asof=t)

        # next weekday in the returns index (one-day lag, no look-ahead)
        next_days = returns.index[returns.index > t]
        if next_days.empty:
            break
        t1 = next_days[0]

        ret_t1 = returns.loc[t1]
        # align weights to available return columns; missing = 0
        aligned = weights_t.reindex(ret_t1.index, fill_value=0.0)
        portfolio_returns[t1] = float(aligned.dot(ret_t1.fillna(0.0)))
        weights_records[t] = weights_t

    weights_df = pd.DataFrame(weights_records).T
    port_series = pd.Series(portfolio_returns, name="portfolio_return")

    stats = performance_stats(port_series.dropna())

    return {"weights": weights_df, "portfolio_returns": port_series, "stats": stats}
