"""
Fetch 30-ticker universe from EODHD (2007-01-01 → today), align to a
NYSE-business-day index, forward-fill BTC and FX over US equity holidays,
and cache to data/cache/prices_30.parquet.

Calendar: pandas_market_calendars NYSE (preferred) or pd.bdate_range fallback.
Reads EODHD_API_KEY from environment (populated via .env by load_dotenv).
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from eodhd import APIClient

load_dotenv()

TICKERS = [
    # Large-cap equities
    "AAPL.US", "MSFT.US", "GOOGL.US", "NVDA.US",
    "JPM.US", "JNJ.US", "XOM.US", "WMT.US",
    # Sector ETFs
    "XLK.US", "XLF.US", "XLE.US", "XLV.US", "XLP.US", "XLU.US",
    # Broad equity ETFs
    "SPY.US", "IWM.US",
    # International equity ETFs
    "EFA.US", "EEM.US", "FXI.US",
    # Fixed income ETFs
    "SHY.US", "IEF.US", "TLT.US", "AGG.US", "HYG.US",
    # Commodities
    "GLD.US", "SLV.US", "DBC.US", "USO.US",
    # FX & crypto (trade on days US equity is closed; will be forward-filled)
    "EURUSD.FOREX", "BTC-USD.CC",
]

# Tickers that trade outside NYSE hours — forward-fill over US holidays
ALWAYS_ON = {"EURUSD.FOREX", "BTC-USD.CC"}

OUTPUT = Path(__file__).parent.parent / "data" / "cache" / "prices_30.parquet"

START = "2007-01-01"
END = date.today().isoformat()


def build_nyse_index(start: str, end: str) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.valid_days(start_date=start, end_date=end)
        idx = pd.DatetimeIndex(schedule).tz_localize(None).normalize()
        print(f"[calendar] Using NYSE calendar via pandas_market_calendars ({len(idx)} days)")
        return idx
    except ImportError:
        idx = pd.bdate_range(start=start, end=end)
        print(f"[calendar] pandas_market_calendars not found; falling back to pd.bdate_range "
              f"({len(idx)} days, misses US holidays)")
        return idx


def fetch_all(client: APIClient, tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        print(f"  Fetching {ticker} …", flush=True)
        raw = client.get_eod_historical_stock_market_data(
            symbol=ticker,
            from_date=start,
            to_date=end,
            period="d",
        )
        df = pd.DataFrame(raw)
        if df.empty or "adjusted_close" not in df.columns:
            print(f"  WARNING: no data for {ticker}")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        series[ticker] = df["adjusted_close"]
    return series


def main() -> None:
    client = APIClient(os.environ["EODHD_API_KEY"])

    print(f"Fetching {len(TICKERS)} tickers from {START} to {END} …")
    series = fetch_all(client, TICKERS, START, END)

    nyse_idx = build_nyse_index(START, END)

    # Reindex every ticker onto the NYSE calendar.
    # US equities: NaN on days they don't trade (none expected after reindex to NYSE days).
    # BTC/FX: forward-fill over any NYSE holiday where they still had a price.
    panels: dict[str, pd.Series] = {}
    for ticker, s in series.items():
        reindexed = s.reindex(nyse_idx)
        if ticker in ALWAYS_ON:
            reindexed = reindexed.ffill()
        panels[ticker] = reindexed

    prices = pd.DataFrame(panels)
    prices.index.name = "date"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(OUTPUT)

    # ── diagnostics ──────────────────────────────────────────────────────────
    print(f"\nShape: {prices.shape}")
    print(f"Index frequency: {prices.index.inferred_freq!r}")
    print(f"Index range: {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\nPer-ticker first non-NaN date:")
    first_valid = prices.apply(lambda s: s.first_valid_index())
    for ticker, dt in first_valid.sort_values().items():
        dt_str = dt.date() if dt is not None and not pd.isnull(dt) else "ALL NaN"
        print(f"  {ticker:22s}  {dt_str}")

    print("\nNaN count per column:")
    nan_counts = prices.isna().sum()
    for ticker, n in nan_counts.items():
        print(f"  {ticker:22s}  {n}")

    # Spot-check: confirm no weekend rows
    weekend_rows = prices[prices.index.dayofweek >= 5]
    if weekend_rows.empty:
        print("\n[OK] Index contains no weekend rows — confirmed NYSE business-day index.")
    else:
        print(f"\n[WARN] {len(weekend_rows)} weekend rows found — calendar alignment may be off.")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
