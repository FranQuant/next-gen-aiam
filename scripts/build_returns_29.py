"""Compute simple returns from adj_close and save to parquet.

Output: data/cache/returns_29_2003_2026.parquet
  Wide format: date index × 29 ticker columns
  First valid return per ticker = first day after inception; pre-inception = NaN.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICES_CACHE = Path("data/cache/prices_29.parquet")
OUT = Path("data/cache/returns_29_2003_2026.parquet")


def main() -> None:
    prices = pd.read_parquet(PRICES_CACHE)
    returns = prices.pct_change()
    # Keep only weekdays (strip any holiday rows that EODHD may inject)
    returns = returns[returns.index.dayofweek < 5]
    returns.to_parquet(OUT)
    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index.min().date()} → {returns.index.max().date()}")
    print(f"Non-NaN counts per ticker:")
    for col in returns.columns:
        n = returns[col].notna().sum()
        print(f"  {col:<20}  {n:5d}")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
