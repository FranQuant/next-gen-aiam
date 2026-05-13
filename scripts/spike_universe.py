"""
Spike: fetch 30-ticker universe from EODHD (last 10 years), pivot to wide
adjusted_close panel, and cache to parquet.
Reads EODHD_API_KEY from environment (populated via .env by load_dotenv).
"""

import os
from datetime import date, timedelta
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
    # FX & crypto
    "EURUSD.FOREX", "BTC-USD.CC",
]
OUTPUT = Path(__file__).parent.parent / "data" / "cache" / "prices_30.parquet"

end = date.today()
start = end - timedelta(days=10 * 365)

client = APIClient(os.environ["EODHD_API_KEY"])

series = {}
for ticker in TICKERS:
    print(f"Fetching {ticker} …", flush=True)
    raw = client.get_eod_historical_stock_market_data(
        symbol=ticker,
        from_date=start.isoformat(),
        to_date=end.isoformat(),
        period="d",
    )
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    series[ticker] = df["adjusted_close"]

prices = pd.DataFrame(series)
prices.index.name = "date"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
prices.to_parquet(OUTPUT)

# ── diagnostics ────────────────────────────────────────────────────────────────
print(f"\nShape: {prices.shape}")
print(f"\nColumns: {prices.columns.tolist()}")
print(f"\nDtypes:\n{prices.dtypes}")

print("\nPer-ticker date range (non-NaN rows):")
for col in prices.columns:
    s = prices[col].dropna()
    print(f"  {col:20s}  {s.index.min().date()} → {s.index.max().date()}  ({len(s)} rows)")

print("\nNaN count per column:")
print(prices.isna().sum().to_string())

print("\nFirst 3 rows:")
print(prices.head(3).to_string())

print("\nLast 3 rows:")
print(prices.tail(3).to_string())

print(f"\nSaved to {OUTPUT}")
