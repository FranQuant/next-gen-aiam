from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.features.realized_power import compute_intraday_features, compute_realized_power_5min


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_panel(
    n_days: int = 2,
    bars_per_day: int = 79,  # 09:30–16:00 at 5-min intervals (inclusive)
    tickers: list[str] | None = None,
    prices: dict | None = None,  # ticker → array of prices (n_days * bars_per_day)
    extra_premarket: bool = False,
) -> pd.DataFrame:
    """Build a synthetic MultiIndex (timestamp, asset) intraday panel in UTC."""
    tickers = tickers or ["SPY.US", "IEF.US"]
    n_bars = n_days * bars_per_day
    # NYSE regular open = 09:30 ET = 14:30 UTC
    records = []
    for ticker in tickers:
        if prices and ticker in prices:
            close_arr = prices[ticker]
        else:
            rng = np.random.default_rng(hash(ticker) % (2**32))
            close_arr = 100.0 * np.exp(np.cumsum(rng.standard_normal(n_bars) * 0.001))

        bar_idx = 0
        for day in range(n_days):
            day_base = pd.Timestamp(f"2021-01-{4 + day:02d}", tz="UTC")
            open_utc = day_base + pd.Timedelta(hours=14, minutes=30)
            for b in range(bars_per_day):
                ts = open_utc + pd.Timedelta(minutes=5 * b)
                records.append({
                    "timestamp": ts,
                    "asset": ticker,
                    "open": close_arr[bar_idx],
                    "high": close_arr[bar_idx] * 1.001,
                    "low": close_arr[bar_idx] * 0.999,
                    "close": close_arr[bar_idx],
                    "volume": 1000.0,
                })
                bar_idx += 1

        if extra_premarket:
            # Add a pre-market bar at 08:00 ET = 13:00 UTC
            for day in range(n_days):
                day_base = pd.Timestamp(f"2021-01-{4 + day:02d}", tz="UTC")
                premarket_ts = day_base + pd.Timedelta(hours=13)
                records.append({
                    "timestamp": premarket_ts,
                    "asset": ticker,
                    "open": 99.0, "high": 99.1, "low": 98.9, "close": 99.0,
                    "volume": 100.0,
                })

    return pd.DataFrame(records).set_index(["timestamp", "asset"])


# ── Shape / structure ─────────────────────────────────────────────────────────

def test_rp_output_shape():
    panel = _make_panel(n_days=3, tickers=["SPY.US", "IEF.US"])
    rp = compute_realized_power_5min(panel)
    assert rp.shape == (3, 2)
    assert set(rp.columns) == {"SPY.US", "IEF.US"}


def test_rp_index_is_datetimeindex():
    panel = _make_panel(n_days=2)
    rp = compute_realized_power_5min(panel)
    assert isinstance(rp.index, pd.DatetimeIndex)


def test_rp_ticker_filter():
    panel = _make_panel(n_days=2, tickers=["SPY.US", "IEF.US"])
    rp = compute_realized_power_5min(panel, tickers=["SPY.US"])
    assert list(rp.columns) == ["SPY.US"]
    assert rp.shape == (2, 1)


# ── Correctness ───────────────────────────────────────────────────────────────

def test_rp_known_value():
    """Manually construct close prices with known squared log returns."""
    # 3 bars per day: prices [100, 101, 103]. Log returns: log(101/100), log(103/101)
    close_prices = np.array([100.0, 101.0, 103.0, 100.0, 101.0, 103.0])  # 2 days
    panel = _make_panel(n_days=2, bars_per_day=3, tickers=["T"],
                        prices={"T": close_prices})
    rp = compute_realized_power_5min(panel, tickers=["T"])

    r1 = np.log(101 / 100)
    r2 = np.log(103 / 101)
    expected_day = r1 ** 2 + r2 ** 2
    assert np.allclose(rp["T"].values, expected_day, atol=1e-12)


def test_rp_nonnegative():
    panel = _make_panel(n_days=5, tickers=["SPY.US", "IEF.US"])
    rp = compute_realized_power_5min(panel)
    assert (rp >= 0).all().all()


def test_rp_single_bar_day_is_zero():
    """A day with only 1 bar yields RP = 0 (no within-day returns)."""
    records = [{
        "timestamp": pd.Timestamp("2021-01-04 14:30:00", tz="UTC"),
        "asset": "T",
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0,
    }]
    panel = pd.DataFrame(records).set_index(["timestamp", "asset"])
    rp = compute_realized_power_5min(panel, tickers=["T"])
    assert rp["T"].iloc[0] == 0.0


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_rp_regular_hours_excludes_premarket():
    """Pre-market bars must be excluded when regular_hours_only=True."""
    panel = _make_panel(n_days=1, bars_per_day=79, tickers=["T"], extra_premarket=True)

    rp_filtered = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=True)
    rp_unfiltered = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=False)

    # Pre-market price of 99 vs regular open ~100 creates an artificial large return
    # The filtered RP should be smaller than (or equal to) unfiltered
    assert rp_filtered["T"].values[0] <= rp_unfiltered["T"].values[0]


def test_rp_regular_hours_false_includes_more_bars():
    panel = _make_panel(n_days=1, bars_per_day=79, tickers=["T"], extra_premarket=True)
    # Unfiltered should sum at least as many squared returns
    rp_on = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=True)
    rp_off = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=False)
    assert rp_off["T"].values[0] >= rp_on["T"].values[0]


# ── Timezone ──────────────────────────────────────────────────────────────────

def test_rp_timezone_date_grouping():
    """Bars after 20:00 UTC (15:00 ET for 16:00 close) should be filtered by regular hours."""
    # Bar at 21:30 UTC = 16:30 ET — post-market, should be excluded
    records = [
        {"timestamp": pd.Timestamp("2021-01-04 14:30:00", tz="UTC"),
         "asset": "T", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0},
        {"timestamp": pd.Timestamp("2021-01-04 15:00:00", tz="UTC"),
         "asset": "T", "open": 101.0, "high": 101.0, "low": 101.0, "close": 101.0, "volume": 0.0},
        # Post-market bar: 21:30 UTC = 16:30 ET
        {"timestamp": pd.Timestamp("2021-01-04 21:30:00", tz="UTC"),
         "asset": "T", "open": 105.0, "high": 105.0, "low": 105.0, "close": 105.0, "volume": 0.0},
    ]
    panel = pd.DataFrame(records).set_index(["timestamp", "asset"])

    rp_filtered = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=True)
    rp_unfiltered = compute_realized_power_5min(panel, tickers=["T"], regular_hours_only=False)

    # Filtered: only 2 bars → 1 return → log(101/100)^2
    r1 = np.log(101.0 / 100.0)
    assert np.allclose(rp_filtered["T"].values[0], r1 ** 2, atol=1e-12)

    # Unfiltered: 3 bars → 2 returns including the post-market jump
    assert rp_unfiltered["T"].values[0] > rp_filtered["T"].values[0]


# ── compute_intraday_features ─────────────────────────────────────────────────

def test_compute_intraday_features_long_format():
    panel = _make_panel(n_days=2, tickers=["SPY.US", "IEF.US"])
    features = compute_intraday_features(panel)
    assert isinstance(features.index, pd.MultiIndex)
    assert "realized_power_5min" in features.columns
    # 2 days × 2 tickers = 4 rows
    assert len(features) == 4
