"""
Spike: fetch CPIAUCSL from FRED (2000-01-01 → today) and cache to parquet.
Reads FRED_API_KEY from the environment.
"""

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

import pandas as pd
from fredapi import Fred

API_KEY = os.environ["FRED_API_KEY"]
SERIES_ID = "CPIAUCSL"
OUTPUT = Path(__file__).parent.parent / "data" / "cache" / "CPIAUCSL.parquet"

fred = Fred(api_key=API_KEY)
series = fred.get_series(SERIES_ID, observation_start="2000-01-01", observation_end=date.today().isoformat())
info = fred.get_series_info(SERIES_ID)

df = series.rename(SERIES_ID).to_frame()
df.index.name = "date"
df.index = pd.to_datetime(df.index)
df = df.sort_index()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT)

print(df.head())
print("\nDtypes:\n", df.dtypes)
print(f"\nFrequency: {info.get('frequency_short', 'unknown')} ({info.get('frequency', '')})")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")
print(f"\nSaved to {OUTPUT}")
