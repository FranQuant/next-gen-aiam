from __future__ import annotations

from pathlib import Path

import pandas as pd

from aiam.data.panel import Panel
from aiam.evaluation.performance import performance_stats
from aiam.strategy.base import Strategy

_WEIGHTS_DIR = Path("data/cache/portfolio_weights")
_PRICES_CACHE = Path("data/cache/prices_30.parquet")


def _weights_path(strategy_name: str, suffix: str = "2008_2026") -> Path:
    safe = strategy_name.replace("(", "_").replace(")", "").replace("/", "_")
    return _WEIGHTS_DIR / f"{safe}_{suffix}.parquet"


def run_horse_race(
    panel: Panel,
    strategy: Strategy,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    save_weights: bool = False,
    strategy_name: str | None = None,
    weights_suffix: str = "2008_2026",
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

    if save_weights and strategy_name is not None:
        path = _weights_path(strategy_name, suffix=weights_suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights_df.to_parquet(path)

    return {"weights": weights_df, "portfolio_returns": port_series, "stats": stats}


def _load_or_run_weights(
    panel: Panel,
    strategy: Strategy,
    strategy_name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return cached weights DataFrame, re-running the harness if the cache is
    missing or older than the prices cache (mtime-check invalidation)."""
    path = _weights_path(strategy_name)
    if path.exists() and path.stat().st_mtime > _PRICES_CACHE.stat().st_mtime:
        return pd.read_parquet(path)
    result = run_horse_race(
        panel, strategy, start=start, end=end,
        save_weights=True, strategy_name=strategy_name,
    )
    return result["weights"]
