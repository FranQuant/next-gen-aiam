from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.features.technical import (
    SignalEngine,
    atr,
    bollinger,
    forward_returns,
    gap,
    momentum,
    rsi,
    volatility,
    volume_signal,
    zscore,
)

N_ROWS = 60
N_ASSETS = 4
TICKERS = [f"A{i}" for i in range(N_ASSETS)]
DATES = pd.bdate_range("2020-01-01", periods=N_ROWS)


def _ret_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((N_ROWS, N_ASSETS)) * 0.01, index=DATES, columns=TICKERS)


def _price_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.standard_normal((N_ROWS, N_ASSETS)) * 0.01, index=DATES, columns=TICKERS)
    return (1 + r).cumprod() * 100


def _ohlcv() -> dict:
    rng = np.random.default_rng(7)
    close = _price_df()
    noise = rng.uniform(0.001, 0.005, (N_ROWS, N_ASSETS))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close.shift(1).bfill() * (1 + rng.uniform(-0.002, 0.002, (N_ROWS, N_ASSETS)))
    vol = pd.DataFrame(
        rng.integers(1000, 10000, (N_ROWS, N_ASSETS)).astype(float), index=DATES, columns=TICKERS
    )
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


# ── Shape / dtype ─────────────────────────────────────────────────────────────

def test_momentum_shape():
    r = _ret_df()
    out = momentum(r, lookback=10)
    assert out.shape == r.shape
    assert list(out.columns) == TICKERS
    assert out.index.equals(r.index)


def test_momentum_dtype():
    assert momentum(_ret_df(), 10).dtypes.unique()[0] == np.float64


def test_volatility_shape():
    assert volatility(_ret_df(), 10).shape == (N_ROWS, N_ASSETS)


def test_forward_returns_shape_and_trailing_nan():
    r = _ret_df()
    h = 5
    out = forward_returns(r, horizon=h)
    assert out.shape == r.shape
    assert out.iloc[-h:].isna().all().all()


def test_zscore_shape():
    assert zscore(_ret_df(), window=10).shape == (N_ROWS, N_ASSETS)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_momentum_all_nan_column():
    r = _ret_df().copy()
    r["A0"] = np.nan
    assert momentum(r, lookback=10)["A0"].isna().all()


def test_momentum_lookback_gt_len():
    r = _ret_df().iloc[:5]
    assert momentum(r, lookback=20).isna().all().all()


def test_volatility_lookback_gt_len():
    r = _ret_df().iloc[:3]
    assert volatility(r, lookback=20).isna().all().all()


def test_single_row_momentum():
    r = _ret_df().iloc[:1]
    out = momentum(r, lookback=1)
    # rolling(1).sum() on a single row is valid
    assert out.shape == (1, N_ASSETS)
    assert out.notna().all().all()


# ── Semantic correctness ──────────────────────────────────────────────────────

def test_momentum_monotone_up():
    prices = pd.DataFrame({"A": np.linspace(100, 200, N_ROWS)}, index=DATES)
    rets = prices.pct_change().dropna()
    mom = momentum(rets, lookback=10).dropna()
    assert (mom > 0).all().all()


def test_momentum_monotone_down():
    prices = pd.DataFrame({"A": np.linspace(200, 100, N_ROWS)}, index=DATES)
    rets = prices.pct_change().dropna()
    mom = momentum(rets, lookback=10).dropna()
    assert (mom < 0).all().all()


def test_volatility_zero_for_constant_returns():
    rets = pd.DataFrame(np.zeros((N_ROWS, N_ASSETS)), index=DATES, columns=TICKERS)
    vol = volatility(rets, lookback=10).dropna()
    assert (vol.abs() < 1e-10).all().all()


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_shape():
    assert rsi(_price_df(), lookback=14).shape == (N_ROWS, N_ASSETS)


def test_rsi_bounds():
    valid = rsi(_price_df(), lookback=14).dropna()
    assert (valid >= 0).all().all()
    assert (valid <= 100).all().all()


def test_rsi_constant_prices_no_crash():
    prices = pd.DataFrame(np.ones((N_ROWS, 2)) * 100.0, index=DATES[:N_ROWS], columns=["X", "Y"])
    out = rsi(prices, lookback=5)
    assert out.shape == prices.shape
    valid = out.dropna()
    if not valid.empty:
        assert (valid >= 0).all().all() and (valid <= 100).all().all()


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_shape():
    o = _ohlcv()
    assert atr(o, lookback=14).shape == (N_ROWS, N_ASSETS)


def test_atr_positive():
    o = _ohlcv()
    valid = atr(o, lookback=14).dropna()
    assert (valid > 0).all().all()


# ── Bollinger ─────────────────────────────────────────────────────────────────

def test_bollinger_keys():
    assert set(bollinger(_price_df(), window=20).keys()) == {"middle", "upper", "lower", "pct"}


def test_bollinger_shapes():
    p = _price_df()
    for v in bollinger(p, window=20).values():
        assert v.shape == p.shape


def test_bollinger_upper_gt_lower():
    b = bollinger(_price_df(), window=20)
    diff = (b["upper"] - b["lower"]).dropna()
    assert (diff > 0).all().all()


# ── Gap / volume ──────────────────────────────────────────────────────────────

def test_gap_shape():
    o = _ohlcv()
    assert gap(o).shape == o["open"].shape


def test_volume_signal_shape():
    o = _ohlcv()
    assert volume_signal(o["volume"], lookback=10).shape == o["volume"].shape


# ── SignalEngine ──────────────────────────────────────────────────────────────

def test_signal_engine_no_ohlcv_raises():
    engine = SignalEngine(returns=_ret_df())
    with pytest.raises(RuntimeError):
        engine.rsi()


def test_signal_engine_no_ohlcv_atr_raises():
    engine = SignalEngine(returns=_ret_df())
    with pytest.raises(RuntimeError):
        engine.atr()


def test_signal_engine_momentum():
    engine = SignalEngine(returns=_ret_df())
    out = engine.momentum(lookback=10)
    assert out.shape == (N_ROWS, N_ASSETS)


def test_signal_engine_with_ohlcv_rsi():
    engine = SignalEngine(returns=_ret_df(), ohlcv=_ohlcv())
    out = engine.rsi(lookback=14)
    assert out.shape == (N_ROWS, N_ASSETS)


def test_signal_engine_ic():
    engine = SignalEngine(returns=_ret_df())
    sig = engine.momentum(lookback=10)
    ic_series = engine.ic(sig, horizon=5)
    assert isinstance(ic_series, pd.Series)
    assert len(ic_series) == N_ROWS
