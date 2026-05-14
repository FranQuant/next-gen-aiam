from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

_SQRT252 = np.sqrt(252)


class TSMOM(PointInTimeStrategy):
    """Time-series momentum — Moskowitz, Ooi, Pedersen (2012).

    Long-only adaptation: assets with a negative momentum signal are held flat
    (weight = 0) rather than shorted.  When all signals are flat (bear market
    scenario), falls back to equal weight rather than returning an all-zero
    portfolio — this keeps the harness invariant that weights sum to 1.
    """

    def __init__(
        self,
        signal_lookback: int = 252,
        vol_lookback: int = 63,
        target_per_asset_vol: float = 0.10,
        long_only: bool = True,
    ) -> None:
        self.signal_lookback = signal_lookback
        self.vol_lookback = vol_lookback
        self.target_per_asset_vol = target_per_asset_vol
        self.long_only = long_only

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        total_lookback = self.signal_lookback + self.vol_lookback + 1

        # Request 2× to survive weekend/holiday rows; then filter to weekdays
        # and take the exact slice, mirroring the GMV/MDP/etc. pattern.
        prices = panel.slice(asof, kind="prices", lookback=total_lookback * 2)
        prices = prices[prices.index.dayofweek < 5]
        prices = prices.iloc[-total_lookback:]

        returns = prices.pct_change().iloc[1:]

        # Drop assets with > 10% NaN in the return window
        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)
        prices = prices[valid_cols]

        # Warm-up guard: if the panel predates the full signal window, return EW.
        # This only fires in the first ~signal_lookback days of a run that starts
        # near the beginning of the price data (e.g. 2008-01-01 with prices from 2007).
        if len(prices) < self.signal_lookback:
            n = len(valid_cols) or 1
            return pd.Series(1.0 / n, index=valid_cols, name=asof)

        # Momentum signal: sign of total return over signal_lookback bars.
        # prices.iloc[-signal_lookback] is the price (signal_lookback - 1) periods
        # before the most recent observation — consistent with the MOP-2012 spec as
        # written in the instruction.
        raw_signal = (
            prices.iloc[-1].values / prices.iloc[-self.signal_lookback].values - 1
        )
        # Flat for assets whose history is too short (NaN or inf from division)
        signal = np.nan_to_num(np.sign(raw_signal), nan=0.0, posinf=0.0, neginf=0.0)

        if self.long_only:
            signal = np.where(signal < 0, 0.0, signal)

        # Inverse-vol scaling: each asset is sized to target_per_asset_vol annualized
        recent_vol = returns.iloc[-self.vol_lookback :].std() * _SQRT252
        recent_vol = recent_vol.clip(lower=1e-6)
        raw_weights = signal * (self.target_per_asset_vol / recent_vol.values)

        # Normalize by gross (abs sum) — works for both long-only (gross = net)
        # and long-short (gross = 1, net ≈ 0).
        gross = np.abs(raw_weights).sum()
        if gross < 1e-12:
            n = len(valid_cols)
            weights = np.full(n, 1.0 / n)
        else:
            weights = raw_weights / gross

        return pd.Series(weights, index=valid_cols, name=asof)
