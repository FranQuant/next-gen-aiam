"""
Spike: fetch AAPL.US daily OHLCV for the last 5 years via EODHD and cache to parquet.
Reads EODHD_API_KEY from the environment.
"""

import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

import pandas as pd
from eodhd import APIClient

API_KEY = os.environ["EODHD_API_KEY"]
TICKER = "AAPL.US"
OUTPUT = Path(__file__).parent.parent / "data" / "cache" / "AAPL_US.parquet"

end = date.today()
start = end - timedelta(days=5 * 365)

client = APIClient(API_KEY)
raw = client.get_eod_historical_stock_market_data(
    symbol=TICKER,
    from_date=start.isoformat(),
    to_date=end.isoformat(),
    period="d",
)

df = pd.DataFrame(raw)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT)

print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print(f"\nDate range: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")
print(f"\nSaved to {OUTPUT}")
