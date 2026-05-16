"""Signal computation mirroring Hilpisch §19.2 SignalEngine API."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Module-level functions ────────────────────────────────────────────────────

def momentum(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Sum of log returns over the trailing `lookback` window."""
    return returns.rolling(lookback).sum()


def volatility(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Annualized rolling standard deviation (× √252) over `lookback`."""
    return returns.rolling(lookback).std() * np.sqrt(252)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """h-period forward cumulative return: shift(-h) + rolling(h).sum(). NaN at the right edge by construction."""
    return returns.shift(-horizon).rolling(horizon).sum()


def zscore(series: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score: (x − rolling mean) / rolling std over `window`."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()


def rsi(prices: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Wilder's RSI over `lookback`. Values in [0, 100]; NaN where price is flat."""
    delta = prices.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    alpha = 1.0 / lookback
    rs = up.ewm(alpha=alpha, adjust=False).mean() / dn.ewm(alpha=alpha, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def atr(ohlc: dict, lookback: int = 14) -> pd.DataFrame:
    """Average True Range. `ohlc` dict must have 'high', 'low', 'close' DataFrames."""
    h, lo, c = ohlc["high"], ohlc["low"], ohlc["close"]
    cp = c.shift(1)
    tr = pd.DataFrame(
        np.maximum(np.maximum((h - lo).values, (h - cp).abs().values), (lo - cp).abs().values),
        index=h.index,
        columns=h.columns,
    )
    return tr.ewm(alpha=1.0 / lookback, adjust=False).mean()


def bollinger(prices: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> dict:
    """Bollinger bands: dict with 'middle', 'upper', 'lower', 'pct' (position in band)."""
    roll = prices.rolling(window)
    middle = roll.mean()
    std = roll.std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct = (prices - lower) / (upper - lower)
    return {"middle": middle, "upper": upper, "lower": lower, "pct": pct}


def gap(ohlc: dict) -> pd.DataFrame:
    """Overnight gap: (open / prev_close) − 1."""
    return ohlc["open"] / ohlc["close"].shift(1) - 1


def volume_signal(volume: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Volume relative to rolling mean: volume / volume.rolling(lookback).mean() − 1."""
    return volume / volume.rolling(lookback).mean() - 1


# ── SignalEngine class ────────────────────────────────────────────────────────

class SignalEngine:
    """Hilpisch §19.2-style signal engine. Hold the panel data; expose feature methods.

    Usage:
        engine = SignalEngine(returns=panel.slice(asof, 'returns', lookback=512), ohlcv=ohlcv)
        mom_252 = engine.momentum(252)
        ic_series = engine.ic(mom_252, horizon=21)
    """

    def __init__(self, returns: pd.DataFrame, ohlcv: dict | None = None) -> None:
        self.returns = returns
        self.ohlcv = ohlcv  # {'open', 'high', 'low', 'close', 'volume'}

    def _require_ohlcv(self) -> None:
        if self.ohlcv is None:
            raise RuntimeError("This feature requires ohlcv — pass ohlcv= to SignalEngine()")

    def momentum(self, lookback: int) -> pd.DataFrame:
        return momentum(self.returns, lookback)

    def volatility(self, lookback: int) -> pd.DataFrame:
        return volatility(self.returns, lookback)

    def forward_returns(self, horizon: int) -> pd.DataFrame:
        return forward_returns(self.returns, horizon)

    def zscore(self, series: pd.DataFrame, window: int) -> pd.DataFrame:
        return zscore(series, window)

    def rsi(self, lookback: int = 14) -> pd.DataFrame:
        self._require_ohlcv()
        return rsi(self.ohlcv["close"], lookback)

    def atr(self, lookback: int = 14) -> pd.DataFrame:
        self._require_ohlcv()
        return atr(self.ohlcv, lookback)

    def bollinger(self, window: int = 20, num_std: float = 2.0) -> dict:
        self._require_ohlcv()
        return bollinger(self.ohlcv["close"], window, num_std)

    def gap(self) -> pd.DataFrame:
        self._require_ohlcv()
        return gap(self.ohlcv)

    def volume_signal(self, lookback: int = 21) -> pd.DataFrame:
        self._require_ohlcv()
        return volume_signal(self.ohlcv["volume"], lookback)

    def ic(self, signal: pd.DataFrame, horizon: int, method: str = "spearman") -> pd.Series:
        """Cross-sectional IC between `signal` and h-period forward returns from self.returns."""
        from aiam.evaluation.ic import information_coefficient
        fwd = forward_returns(self.returns, horizon)
        return information_coefficient(signal, fwd, method=method)
