"""
Fetch full OHLCV for the 29-asset universe from EODHD and build the panel.

Outputs
-------
data/cache/prices_29_ohlcv_2003_2026.parquet   MultiIndex (date, ticker) × 6 OHLCV cols
data/cache/prices_29.parquet                    Wide adj_close panel (dates × tickers)
data/raw/prices_29_ohlcv_2003_2026.csv[.gz]    CSV export

Run:
    source .venv/bin/activate
    python scripts/build_ohlcv_29.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from dotenv import load_dotenv
from eodhd import APIClient

load_dotenv()

from aiam.data.universe import UNIVERSE_29

FETCH_START = "2003-01-01"
FETCH_END = "2026-04-30"
CACHE_DIR = Path("data/cache")
RAW_DIR = Path("data/raw")
OHLCV_PARQUET = CACHE_DIR / "prices_29_ohlcv_2003_2026.parquet"
WIDE_PARQUET = CACHE_DIR / "prices_29.parquet"
PER_TICKER_DIR = CACHE_DIR / "ohlcv_per_ticker"
CSV_SIZE_LIMIT_MB = 80

# EURUSD is FX — forward-fill across NYSE holidays
FX_TICKERS = {"EURUSD.FOREX"}


def _eodhd_symbol(ticker: str) -> str:
    """Convert internal ticker to EODHD API symbol."""
    if ticker.endswith(".FOREX"):
        base = ticker.replace(".FOREX", "")
        return f"{base}.FOREX"
    return ticker


def _fetch_ticker(client: APIClient, ticker: str) -> pd.DataFrame:
    """Fetch OHLCV from EODHD; return DataFrame with columns open/high/low/close/adj_close/volume."""
    cache_path = PER_TICKER_DIR / f"{ticker.replace('/', '_')}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    symbol = _eodhd_symbol(ticker)
    raw = client.get_eod_historical_stock_market_data(
        symbol=symbol,
        from_date=FETCH_START,
        to_date=FETCH_END,
        period="d",
    )
    df = pd.DataFrame(raw)
    if df.empty:
        print(f"  [warn] empty response for {ticker}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Normalise column names (EODHD uses 'adjusted_close')
    rename_map = {"adjusted_close": "adj_close"}
    df = df.rename(columns=rename_map)
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_parquet(cache_path)
    return df


def build_nyse_index() -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=FETCH_START, end_date=FETCH_END)
    dr = mcal.date_range(sched, frequency="1D")
    # Strip timezone and time component — keep date-only tz-naive index
    dates = dr.normalize().tz_convert(None).rename("date")
    return dates


def align_to_calendar(
    df: pd.DataFrame, cal: pd.DatetimeIndex, ticker: str
) -> pd.DataFrame:
    """Reindex to NYSE calendar; forward-fill only for FX tickers."""
    df = df.reindex(cal)
    if ticker in FX_TICKERS:
        df = df.ffill()
    return df


def main() -> None:
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        raise EnvironmentError("EODHD_API_KEY not set")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PER_TICKER_DIR.mkdir(parents=True, exist_ok=True)

    client = APIClient(api_key)
    nyse_cal = build_nyse_index()
    print(f"NYSE calendar: {nyse_cal[0].date()} → {nyse_cal[-1].date()}  ({len(nyse_cal)} trading days)")

    all_frames: dict[str, pd.DataFrame] = {}
    for ticker in UNIVERSE_29:
        print(f"  [{UNIVERSE_29.index(ticker)+1:2d}/29] {ticker}...", end="", flush=True)
        try:
            df = _fetch_ticker(client, ticker)
            df = align_to_calendar(df, nyse_cal, ticker)
            all_frames[ticker] = df
            n_valid = df["adj_close"].notna().sum() if "adj_close" in df.columns else 0
            print(f"  {n_valid} valid rows")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_frames[ticker] = pd.DataFrame(index=nyse_cal, columns=["open", "high", "low", "close", "adj_close", "volume"])

    # Build MultiIndex panel
    print("\nBuilding MultiIndex panel...")
    panels = []
    for ticker, df in all_frames.items():
        df = df.copy()
        df.index.name = "date"
        df["ticker"] = ticker
        df = df.reset_index().set_index(["date", "ticker"])
        panels.append(df)

    panel_df = pd.concat(panels).sort_index()
    panel_df.to_parquet(OHLCV_PARQUET)
    print(f"Saved MultiIndex panel → {OHLCV_PARQUET}  shape={panel_df.shape}")

    # Build wide adj_close panel (what the strategy harness consumes)
    print("Building wide adj_close panel...")
    wide = pd.DataFrame({
        ticker: all_frames[ticker]["adj_close"] if "adj_close" in all_frames[ticker].columns
        else pd.Series(dtype=float)
        for ticker in UNIVERSE_29
    }, index=nyse_cal)
    wide.index.name = "date"
    wide.to_parquet(WIDE_PARQUET)
    print(f"Saved wide panel → {WIDE_PARQUET}  shape={wide.shape}")

    # CSV export
    ohlcv_flat = panel_df.reset_index()
    csv_path = RAW_DIR / "prices_29_ohlcv_2003_2026.csv"
    gz_path = RAW_DIR / "prices_29_ohlcv_2003_2026.csv.gz"

    ohlcv_flat.to_csv(csv_path, index=False)
    size_mb = csv_path.stat().st_size / 1_048_576
    if size_mb > CSV_SIZE_LIMIT_MB:
        ohlcv_flat.to_csv(gz_path, index=False, compression="gzip")
        csv_path.unlink()
        print(f"Saved compressed CSV → {gz_path}  ({size_mb:.1f} MB raw, compressed)")
    else:
        print(f"Saved CSV → {csv_path}  ({size_mb:.1f} MB)")

    # Summary
    print("\n── Per-ticker summary ──")
    print(f"{'ticker':<20}  {'inception':<12}  {'last':<12}  {'pct_valid':>9}")
    n_trading = len(nyse_cal)
    for ticker in UNIVERSE_29:
        if "adj_close" not in all_frames[ticker].columns:
            continue
        s = all_frames[ticker]["adj_close"]
        valid = s.dropna()
        pct = len(valid) / n_trading * 100 if n_trading > 0 else 0
        inception = valid.index.min().date() if len(valid) > 0 else "N/A"
        last = valid.index.max().date() if len(valid) > 0 else "N/A"
        print(f"{ticker:<20}  {str(inception):<12}  {str(last):<12}  {pct:8.1f}%")

    print(f"\nTotal panel rows: {len(panel_df):,}")
    print(f"Wide panel shape: {wide.shape}")


if __name__ == "__main__":
    main()
