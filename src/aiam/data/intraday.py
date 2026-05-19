"""Fetch and cache EODHD intraday OHLCV bars."""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import requests

log = logging.getLogger(__name__)

_EODHD_INTRADAY_URL = "https://eodhd.com/api/intraday/{ticker}"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_NO_RETRY_CODES = {401, 402, 403}


def _chunk_date_ranges(start: date, end: date, months: int) -> list[tuple[date, date]]:
    """Split [start, end] into chunks of `months` months."""
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        y = chunk_start.year + (chunk_start.month - 1 + months) // 12
        m = (chunk_start.month - 1 + months) % 12 + 1
        next_start = date(y, m, 1)
        chunk_end = min(next_start - timedelta(days=1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = next_start
    return chunks


def _fetch_chunk(ticker: str, start: date, end: date, interval: str, api_key: str) -> list[dict]:
    """Fetch one chunk from EODHD intraday endpoint."""
    url = _EODHD_INTRADAY_URL.format(ticker=ticker)
    ts_from = int(datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    ts_to = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    params = {"api_token": api_key, "interval": interval, "from": ts_from, "to": ts_to, "fmt": "json"}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else []
            if resp.status_code in _NO_RETRY_CODES:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            log.warning("HTTP %d for %s %s→%s (attempt %d)", resp.status_code, ticker, start, end, attempt + 1)
        except requests.HTTPError:
            raise
        except requests.RequestException as e:
            log.warning("Request error for %s %s→%s (attempt %d): %s", ticker, start, end, attempt + 1, e)
            if attempt == _MAX_RETRIES - 1:
                raise
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

    log.error("Failed %s chunk %s→%s after %d attempts", ticker, start, end, _MAX_RETRIES)
    return []


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert EODHD intraday bar list to UTC-indexed DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.index.name = "timestamp"
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def fetch_intraday_bars(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str = "5m",
    api_key: Optional[str] = None,
    chunk_months: int = 3,
    sleep_between_chunks: float = 0.5,
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars from EODHD for a single ticker.

    Returns DataFrame indexed by UTC timestamp with columns [open, high, low, close, volume].
    Chunks the date range to stay under per-request bar limits; retries on transient errors.
    """
    if api_key is None:
        api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise EnvironmentError("EODHD_API_KEY not set")

    chunks = _chunk_date_ranges(start_date, end_date, chunk_months)
    all_bars: list[pd.DataFrame] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        log.info("Fetching %s chunk %d/%d: %s → %s", ticker, i + 1, len(chunks), chunk_start, chunk_end)
        try:
            bars = _fetch_chunk(ticker, chunk_start, chunk_end, interval, api_key)
            if bars:
                all_bars.append(_bars_to_df(bars))
        except requests.HTTPError:
            raise
        except Exception as e:
            log.error("Skipping chunk %s→%s for %s: %s", chunk_start, chunk_end, ticker, e)
        if i < len(chunks) - 1:
            time.sleep(sleep_between_chunks)

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = pd.concat(all_bars).sort_index()
    return result[~result.index.duplicated(keep="first")]


def fetch_intraday_panel(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    interval: str = "5m",
    api_key: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Fetch intraday bars for multiple tickers; return long-format MultiIndex (timestamp, asset) panel."""
    frames = []
    for ticker in tickers:
        log.info("Fetching intraday bars for %s...", ticker)
        df = fetch_intraday_bars(ticker, start_date, end_date, interval, api_key, **kwargs)
        df = df.assign(asset=ticker).reset_index().set_index(["timestamp", "asset"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(frames).sort_index()


def load_cached_intraday(
    cache_path: Path,
    tickers: Optional[Sequence[str]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Load cached intraday panel from parquet, optionally filtering by ticker and date."""
    df = pd.read_parquet(Path(cache_path))
    if tickers is not None:
        df = df[df.index.get_level_values("asset").isin(tickers)]
    if start is not None:
        ts_start = pd.Timestamp(start, tz="UTC")
        df = df[df.index.get_level_values("timestamp") >= ts_start]
    if end is not None:
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df.index.get_level_values("timestamp") <= ts_end]
    return df


def fetch_or_load_intraday(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    cache_path: Path,
    interval: str = "5m",
    force_refetch: bool = False,
    api_key: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load from cache if present, otherwise fetch from EODHD and cache the result."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refetch:
        log.info("Loading cached intraday panel from %s", cache_path)
        return pd.read_parquet(cache_path)
    panel = fetch_intraday_panel(tickers, start_date, end_date, interval, api_key, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path)
    log.info("Saved intraday panel → %s  shape=%s", cache_path, panel.shape)
    return panel
