from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.factor_signals import momentum_signal
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)

_SQRT252 = np.sqrt(252)


class FactorPortfolio(PointInTimeStrategy):
    def __init__(
        self,
        signal_fn: Callable,
        lookback: int = 756,
        top_fraction: float = 1 / 3,
        weighting: str = "inverse_vol",
    ) -> None:
        self.signal_fn = signal_fn
        self.lookback = lookback
        self.top_fraction = top_fraction
        self.weighting = weighting

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        prices = panel.slice(asof, kind="prices", lookback=self.lookback * 2)
        prices = prices[prices.index.dayofweek < 5]
        prices = prices.iloc[-(self.lookback + 1):]
        returns = prices.pct_change().iloc[1:]

        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)

        universe = panel.universe_at(asof)

        def _ew_fallback() -> pd.Series:
            cols = valid_cols if valid_cols else list(universe)
            k = len(cols)
            return pd.Series(1.0 / k, index=cols).reindex(universe, fill_value=0.0).rename(asof)

        if not valid_cols:
            return _ew_fallback()

        signal = self.signal_fn(returns, asof)
        signal = signal.dropna()

        if signal.empty:
            logger.warning("FactorPortfolio fallback to EW at asof=%s: all signals NaN", asof.date())
            return _ew_fallback()

        k = max(1, int(np.ceil(len(signal) * self.top_fraction)))
        top_assets = signal.nlargest(k).index.tolist()

        selected = returns[top_assets]

        if self.weighting == "inverse_vol":
            vols = selected.std() * _SQRT252
            inv_vol = 1.0 / vols.clip(lower=1e-8)
            weights = inv_vol / inv_vol.sum()
        else:
            weights = pd.Series(1.0 / len(top_assets), index=top_assets)

        return weights.reindex(universe, fill_value=0.0).rename(asof)


class FF3MomLongShort(PointInTimeStrategy):
    """12-1 momentum long-short: long top tercile, short bottom tercile.

    Equal-weight within each leg; gross exposure = 1.0, net ≈ 0.
    Assumes zero borrow cost and unlimited short availability.
    """

    def __init__(self, lookback: int = 756, skip: int = 21, top_fraction: float = 1 / 3) -> None:
        self.lookback = lookback
        self.skip = skip
        self.top_fraction = top_fraction

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        prices = panel.slice(asof, kind="prices", lookback=self.lookback * 2)
        prices = prices[prices.index.dayofweek < 5].iloc[-(self.lookback + 1):]
        returns = prices.pct_change().iloc[1:]

        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)

        universe = panel.universe_at(asof)
        n = len(valid_cols)
        if n < 3:
            return pd.Series(1.0 / max(n, 1), index=valid_cols).reindex(universe, fill_value=0.0).rename(asof)

        sig = momentum_signal(returns, asof, lookback=min(self.lookback, 252), skip=self.skip).dropna()
        k = max(1, int(np.ceil(len(sig) * self.top_fraction)))
        longs = sig.nlargest(k).index
        shorts = sig.nsmallest(k).index

        weights = pd.Series(0.0, index=valid_cols)
        weights[longs] += 0.5 / len(longs)
        weights[shorts] -= 0.5 / len(shorts)

        return weights.reindex(universe, fill_value=0.0).rename(asof)


class MultiFactorPortfolio(PointInTimeStrategy):
    """Equal-weight combination of N factor sub-strategies, rebalanced daily."""

    def __init__(self, strategies: list[FactorPortfolio]) -> None:
        self.strategies = strategies

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        universe = panel.universe_at(asof)
        all_w = pd.DataFrame(
            {i: s._predict_weights(panel, asof) for i, s in enumerate(self.strategies)}
        ).reindex(universe, fill_value=0.0)
        combined = all_w.mean(axis=1)
        total = combined.sum()
        if total > 1e-12:
            combined = combined / total
        return combined.rename(asof)
