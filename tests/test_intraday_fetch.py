from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from aiam.data.intraday import (
    _bars_to_df,
    _chunk_date_ranges,
    fetch_intraday_bars,
    fetch_intraday_panel,
    fetch_or_load_intraday,
    load_cached_intraday,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(ts: int, price: float = 100.0, volume: int | None = 1000) -> dict:
    return {
        "timestamp": ts,
        "gmtoffset": 0,
        "datetime": "2021-01-04 14:30:00",
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
        "volume": volume,
    }


def _mock_resp(data, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.text = str(data)
    return resp


def _synthetic_panel(n_bars: int = 10, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY.US", "IEF.US"]
    ts_start = pd.Timestamp("2021-01-04 14:30:00", tz="UTC")
    timestamps = [ts_start + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]
    records = []
    for ticker in tickers:
        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp": ts,
                "asset": ticker,
                "open": 100.0 + i * 0.1,
                "high": 100.5 + i * 0.1,
                "low": 99.5 + i * 0.1,
                "close": 100.0 + i * 0.1,
                "volume": float(1000 + i),
            })
    return pd.DataFrame(records).set_index(["timestamp", "asset"])


# ── _chunk_date_ranges ────────────────────────────────────────────────────────

def test_chunk_date_ranges_single():
    chunks = _chunk_date_ranges(date(2021, 1, 1), date(2021, 2, 28), months=3)
    assert len(chunks) == 1
    assert chunks[0] == (date(2021, 1, 1), date(2021, 2, 28))


def test_chunk_date_ranges_multiple():
    chunks = _chunk_date_ranges(date(2021, 1, 1), date(2021, 12, 31), months=3)
    assert len(chunks) == 4
    assert chunks[0] == (date(2021, 1, 1), date(2021, 3, 31))
    assert chunks[-1] == (date(2021, 10, 1), date(2021, 12, 31))


def test_chunk_date_ranges_cover_full_period():
    start, end = date(2021, 1, 1), date(2023, 6, 30)
    chunks = _chunk_date_ranges(start, end, months=1)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # No gaps between consecutive chunks
    for (_, a_end), (b_start, _) in zip(chunks[:-1], chunks[1:]):
        from datetime import timedelta
        assert b_start == a_end + timedelta(days=1)


# ── _bars_to_df ───────────────────────────────────────────────────────────────

def test_bars_to_df_shape():
    bars = [_bar(1609772400 + i * 300) for i in range(5)]
    df = _bars_to_df(bars)
    assert df.shape == (5, 5)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_bars_to_df_utc_index():
    bars = [_bar(1609772400)]
    df = _bars_to_df(bars)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_bars_to_df_volume_null():
    bars = [_bar(1609772400, volume=None)]
    df = _bars_to_df(bars)
    assert pd.isna(df["volume"].iloc[0])
    assert not pd.isna(df["close"].iloc[0])


def test_bars_to_df_empty():
    df = _bars_to_df([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ── fetch_intraday_bars ───────────────────────────────────────────────────────

def test_fetch_intraday_bars_success():
    bars = [_bar(1609772400 + i * 300) for i in range(10)]
    with patch("aiam.data.intraday.requests.get", return_value=_mock_resp(bars)):
        df = fetch_intraday_bars("SPY.US", date(2021, 1, 4), date(2021, 3, 31), api_key="test")
    assert not df.empty
    assert "close" in df.columns
    assert df.index.tz is not None


def test_fetch_intraday_bars_deduplication():
    # Two chunks with overlapping bars at boundaries
    ts_base = 1609772400
    bars_a = [_bar(ts_base + i * 300) for i in range(5)]
    bars_b = [_bar(ts_base + i * 300) for i in range(3, 8)]  # overlap at 3,4

    responses = [_mock_resp(bars_a), _mock_resp(bars_b)]
    with patch("aiam.data.intraday.requests.get", side_effect=responses), \
         patch("aiam.data.intraday.time.sleep"):
        df = fetch_intraday_bars("SPY.US", date(2021, 1, 1), date(2021, 6, 30),
                                 chunk_months=3, api_key="test")
    # Unique timestamps: 0..7 = 8 rows
    assert not df.index.duplicated().any()
    assert len(df) == 8


def test_fetch_intraday_bars_http_401_raises():
    with patch("aiam.data.intraday.requests.get", return_value=_mock_resp([], status=401)):
        with pytest.raises(requests.HTTPError):
            fetch_intraday_bars("SPY.US", date(2021, 1, 1), date(2021, 3, 31), api_key="bad_key")


def test_fetch_intraday_bars_empty_response():
    with patch("aiam.data.intraday.requests.get", return_value=_mock_resp([])):
        df = fetch_intraday_bars("SPY.US", date(2021, 1, 1), date(2021, 3, 31), api_key="test")
    assert df.empty


# ── fetch_intraday_panel ──────────────────────────────────────────────────────

def test_fetch_intraday_panel_multi_ticker():
    bars = [_bar(1609772400 + i * 300) for i in range(5)]
    with patch("aiam.data.intraday.requests.get", return_value=_mock_resp(bars)), \
         patch("aiam.data.intraday.time.sleep"):
        panel = fetch_intraday_panel(["SPY.US", "IEF.US"], date(2021, 1, 1), date(2021, 3, 31),
                                     api_key="test")
    assert isinstance(panel.index, pd.MultiIndex)
    assert "asset" in panel.index.names
    assert set(panel.index.get_level_values("asset").unique()) == {"SPY.US", "IEF.US"}


# ── load_cached_intraday ──────────────────────────────────────────────────────

def test_load_cached_intraday_basic():
    panel = _synthetic_panel(10)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        panel.to_parquet(path)
        loaded = load_cached_intraday(path)
        assert loaded.shape == panel.shape
    finally:
        path.unlink(missing_ok=True)


def test_load_cached_intraday_ticker_filter():
    panel = _synthetic_panel(10, tickers=["SPY.US", "IEF.US"])
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        panel.to_parquet(path)
        loaded = load_cached_intraday(path, tickers=["SPY.US"])
        assert set(loaded.index.get_level_values("asset").unique()) == {"SPY.US"}
    finally:
        path.unlink(missing_ok=True)


# ── fetch_or_load_intraday ────────────────────────────────────────────────────

def test_fetch_or_load_uses_cache_no_http():
    panel = _synthetic_panel(10)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        panel.to_parquet(path)
        with patch("aiam.data.intraday.requests.get") as mock_get:
            result = fetch_or_load_intraday(
                ["SPY.US", "IEF.US"], date(2021, 1, 1), date(2021, 3, 31),
                cache_path=path, api_key="test",
            )
        mock_get.assert_not_called()
        assert result.shape == panel.shape
    finally:
        path.unlink(missing_ok=True)


def test_fetch_or_load_fetches_when_missing():
    bars = [_bar(1609772400 + i * 300) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "missing.parquet"
        with patch("aiam.data.intraday.requests.get", return_value=_mock_resp(bars)), \
             patch("aiam.data.intraday.time.sleep"):
            result = fetch_or_load_intraday(
                ["SPY.US"], date(2021, 1, 1), date(2021, 3, 31),
                cache_path=path, api_key="test",
            )
        assert path.exists()
        assert not result.empty
