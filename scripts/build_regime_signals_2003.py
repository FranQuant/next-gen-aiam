"""
Re-run regime signals on the extended 2003-2026 sample.

Fetches FRED macro from 2000-01-01 (for warmup) through today.
Saves output to data/cache/regime_signals_2003_2026.parquet.
Compares regime day counts to the old 2008-2026 sample.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from fredapi import Fred
from eodhd import APIClient

from aiam.data.regimes.regime_engine import build_regime_signals

FETCH_START = "2000-01-01"
FETCH_END = date.today().isoformat()
CACHE_DIR = Path("data/cache/macro")
OLD_REGIME_PATH = Path("data/cache/regime_signals.parquet")
OUT_PATH = Path("data/cache/regime_signals_2003_2026.parquet")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _fred(fred: Fred, sid: str) -> pd.Series:
    p = _cache(sid)
    if p.exists():
        return pd.read_parquet(p)[sid]
    s = fred.get_series(sid, observation_start=FETCH_START, observation_end=FETCH_END)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().rename(sid)
    s.to_frame().to_parquet(p)
    return s


def _eodhd(client: APIClient, ticker: str, col: str = "adjusted_close") -> pd.Series:
    name = ticker.replace(".", "_")
    p = _cache(name)
    if p.exists():
        df = pd.read_parquet(p)
        return df[col] if col in df.columns else df["close"]
    raw = client.get_eod_historical_stock_market_data(
        symbol=ticker, from_date=FETCH_START, to_date=FETCH_END, period="d"
    )
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(p)
    return df[col] if col in df.columns else df["close"]


def build_df_macro(series: dict[str, pd.Series]) -> pd.DataFrame:
    gdp_m = series["gdpc1"].resample("ME").last().ffill()
    gdp_qoq = gdp_m.pct_change(3).rename("GDP_QoQ")
    cpi_m = series["cpi"].resample("ME").last()
    cpi_mom = cpi_m.pct_change(1).rename("CPI_MoM")
    unem = series["unrate"].resample("ME").last().rename("UNEM")
    yc10 = series["gs10"].resample("ME").last().rename("YC_10Y")
    yc2 = series["gs2"].resample("ME").last().rename("YC_2Y")
    yc_step = (yc10 - yc2).rename("YC_STEP")
    vix = series["vix_raw"].resample("ME").mean().rename("VIX")
    spx_m = series["spx_raw"].resample("ME").last()
    spx = spx_m.pct_change(1).rename("SPX")
    df = pd.concat([gdp_qoq, cpi_mom, unem, yc10, yc2, yc_step, vix, spx], axis=1)
    return df.dropna(how="all").sort_index()


def main() -> None:
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    client = APIClient(os.environ["EODHD_API_KEY"])

    print("Fetching FRED series...")
    series = {
        "gdpc1": _fred(fred, "GDPC1"),
        "cpi": _fred(fred, "CPIAUCSL"),
        "unrate": _fred(fred, "UNRATE"),
        "gs10": _fred(fred, "GS10"),
        "gs2": _fred(fred, "GS2"),
        "vix_raw": _eodhd(client, "VIX.INDX", col="close"),
        "spx_raw": _eodhd(client, "GSPC.INDX"),
    }

    print("Resampling to monthly...")
    df_macro = build_df_macro(series)
    print(f"  df_macro: {df_macro.shape}  {df_macro.index.min().date()} → {df_macro.index.max().date()}")

    print("Building regime signals...")
    df_signals = build_regime_signals(df_macro)
    df_signals.to_parquet(OUT_PATH)
    print(f"Saved → {OUT_PATH}")

    # Trim to 2003-2026 for reporting
    sig_2003 = df_signals[df_signals.index >= "2003-01-01"]
    sig_2003 = sig_2003[sig_2003.index <= "2026-04-30"]

    print("\n── Regime day counts (2003-2026, monthly) ──")
    counts_new = sig_2003["dominant_regime"].value_counts().sort_index()
    print(counts_new.to_string())

    if OLD_REGIME_PATH.exists():
        old = pd.read_parquet(OLD_REGIME_PATH)
        print("\n── Regime day counts (2008-2026, old sample) ──")
        counts_old = old["dominant_regime"].value_counts().sort_index()
        print(counts_old.to_string())

        print("\n── Distribution comparison (% of months) ──")
        total_new = counts_new.sum()
        total_old = counts_old.sum()
        regimes = sorted(set(counts_new.index) | set(counts_old.index))
        print(f"{'regime':<8}  {'new 2003-26':>12}  {'old 2008-26':>12}")
        for r in regimes:
            pn = counts_new.get(r, 0) / total_new * 100
            po = counts_old.get(r, 0) / total_old * 100
            print(f"{r:<8}  {pn:11.1f}%  {po:11.1f}%")


if __name__ == "__main__":
    main()
