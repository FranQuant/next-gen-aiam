"""
ETL: fetch macro + index data, compute regime signals, save to parquet.

Sources
-------
FRED (via fredapi):  GDPC1, CPIAUCSL, UNRATE, GS10, GS2
EODHD (via eodhd):   VIX.INDX (or fallback: realised vol), GSPC.INDX (or SPY.US)

Output
------
data/cache/regime_signals.parquet
  index : monthly DatetimeIndex (MonthEnd)
  cols  : regime_GDP, regime_CPI, regime_UNEM, regime_YC10, regime_YC2,
          regime_YCSTEP, regime_VIX, regime_SPX, dominant_regime

Run:
    source .venv/bin/activate
    python scripts/build_regime_signals.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from fredapi import Fred
from eodhd import APIClient

from aiam.data.regimes.regime_engine import build_regime_signals

# ── config ───────────────────────────────────────────────────────────────────
FETCH_START = "2000-01-01"
FETCH_END = date.today().isoformat()
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "macro"
OUT_PATH = Path(__file__).parent.parent / "data" / "cache" / "regime_signals.parquet"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _load_or_fetch_fred(fred: Fred, series_id: str) -> pd.Series:
    cache = _cache_path(series_id)
    if cache.exists():
        print(f"  [cache] {series_id}")
        df = pd.read_parquet(cache)
        return df[series_id]

    print(f"  [fetch] {series_id} from FRED …")
    s = fred.get_series(series_id, observation_start=FETCH_START, observation_end=FETCH_END)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().rename(series_id)
    s.to_frame().to_parquet(cache)
    return s


def _load_or_fetch_eodhd(client: APIClient, ticker: str, name: str) -> pd.DataFrame:
    cache = _cache_path(name)
    if cache.exists():
        print(f"  [cache] {name} ({ticker})")
        return pd.read_parquet(cache)

    print(f"  [fetch] {ticker} from EODHD …")
    raw = client.get_eod_historical_stock_market_data(
        symbol=ticker,
        from_date=FETCH_START,
        to_date=FETCH_END,
        period="d",
    )
    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError(f"EODHD returned empty data for {ticker}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(cache)
    return df


def _fetch_eodhd_series(
    client: APIClient,
    primary_ticker: str,
    fallback_ticker: str,
    name: str,
    col: str = "adjusted_close",
) -> pd.Series:
    try:
        df = _load_or_fetch_eodhd(client, primary_ticker, name)
    except Exception as e:
        print(f"  [warn] {primary_ticker} failed ({e}); trying {fallback_ticker}")
        df = _load_or_fetch_eodhd(client, fallback_ticker, f"{name}_fallback")
    return df[col].rename(name)


# ── fetch ─────────────────────────────────────────────────────────────────────

def fetch_all() -> dict[str, pd.Series]:
    fred_key = os.environ.get("FRED_API_KEY")
    eodhd_key = os.environ.get("EODHD_API_KEY")
    if not fred_key:
        raise EnvironmentError("FRED_API_KEY not set")
    if not eodhd_key:
        raise EnvironmentError("EODHD_API_KEY not set")

    fred = Fred(api_key=fred_key)
    client = APIClient(eodhd_key)

    print("Fetching FRED series …")
    gdpc1 = _load_or_fetch_fred(fred, "GDPC1")
    cpi = _load_or_fetch_fred(fred, "CPIAUCSL")
    unrate = _load_or_fetch_fred(fred, "UNRATE")
    gs10 = _load_or_fetch_fred(fred, "GS10")
    gs2 = _load_or_fetch_fred(fred, "GS2")

    print("Fetching EODHD series …")
    vix_raw = _fetch_eodhd_series(client, "VIX.INDX", "VIX.INDX", "VIX", col="close")
    spx_raw = _fetch_eodhd_series(client, "GSPC.INDX", "SPY.US", "SPX")

    return {
        "gdpc1": gdpc1,
        "cpi": cpi,
        "unrate": unrate,
        "gs10": gs10,
        "gs2": gs2,
        "vix_raw": vix_raw,
        "spx_raw": spx_raw,
    }


# ── resample ──────────────────────────────────────────────────────────────────

def build_df_macro(series: dict[str, pd.Series]) -> pd.DataFrame:
    # GDP: quarterly → monthly (ffill), then QoQ = pct_change(3)
    gdp_m = series["gdpc1"].resample("ME").last().ffill()
    gdp_qoq = gdp_m.pct_change(3).rename("GDP_QoQ")

    # CPI: monthly MoM
    cpi_m = series["cpi"].resample("ME").last()
    cpi_mom = cpi_m.pct_change(1).rename("CPI_MoM")

    # Unemployment: monthly level
    unem = series["unrate"].resample("ME").last().rename("UNEM")

    # Yield curve: monthly levels
    yc10 = series["gs10"].resample("ME").last().rename("YC_10Y")
    yc2 = series["gs2"].resample("ME").last().rename("YC_2Y")
    yc_step = (yc10 - yc2).rename("YC_STEP")

    # VIX: monthly mean of daily closes
    vix = series["vix_raw"].resample("ME").mean().rename("VIX")

    # SPX: monthly last, then MoM return
    spx_m = series["spx_raw"].resample("ME").last()
    spx = spx_m.pct_change(1).rename("SPX")

    df = pd.concat(
        [gdp_qoq, cpi_mom, unem, yc10, yc2, yc_step, vix, spx], axis=1
    )
    df = df.dropna(how="all").sort_index()
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    series = fetch_all()

    print("\nResampling to monthly …")
    df_macro = build_df_macro(series)
    print(f"  df_macro shape: {df_macro.shape}")
    print(f"  date range:     {df_macro.index.min().date()} → {df_macro.index.max().date()}")

    print("\nBuilding regime signals …")
    df_signals = build_regime_signals(df_macro)
    print(f"  regime_signals shape: {df_signals.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_signals.to_parquet(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    print("\nLast 12 months of dominant_regime:")
    print(df_signals["dominant_regime"].iloc[-12:].to_string())


if __name__ == "__main__":
    main()
