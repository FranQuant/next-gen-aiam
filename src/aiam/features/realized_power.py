"""Daily realized power from 5-minute intraday bars."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def compute_realized_power_5min(
    intraday_panel: pd.DataFrame,
    tickers: Optional[Sequence[str]] = None,
    market_tz: str = "America/New_York",
    regular_hours_only: bool = True,
) -> pd.DataFrame:
    """Compute daily realized power from 5-minute intraday bars.

    RP_t = sum_k r_{t,k}^2  where r are within-day 5-min log returns.

    Returns wide-format DataFrame indexed by trading date (tz-naive) with one column per asset.
    """
    df_flat = intraday_panel.reset_index()

    if tickers is not None:
        df_flat = df_flat[df_flat["asset"].isin(tickers)]

    results: dict[str, pd.Series] = {}
    for asset, asset_df in df_flat.groupby("asset"):
        asset_df = asset_df.sort_values("timestamp")
        ts = pd.to_datetime(asset_df["timestamp"], utc=True)
        close = pd.Series(asset_df["close"].values, index=ts.values).tz_localize("UTC")
        close = close.tz_convert(market_tz).dropna()

        if regular_hours_only:
            close = close.between_time("09:30", "16:00")

        if close.empty:
            continue

        rp_by_date: dict = {}
        for trade_date, group in close.groupby(close.index.normalize().date):
            group = group.sort_index()
            if len(group) < 2:
                rp_by_date[trade_date] = 0.0
                continue
            log_rets = np.log(group.values[1:] / group.values[:-1])
            rp_by_date[trade_date] = float(np.sum(log_rets ** 2))

        results[asset] = pd.Series(rp_by_date)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def compute_intraday_features(
    intraday_panel: pd.DataFrame,
    tickers: Optional[Sequence[str]] = None,
    market_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Compute full intraday feature set as long-format MultiIndex (date, asset) DataFrame."""
    rp = compute_realized_power_5min(intraday_panel, tickers=tickers, market_tz=market_tz)
    rp_long = rp.stack()
    rp_long.index.names = ["date", "asset"]
    return rp_long.rename("realized_power_5min").to_frame()
