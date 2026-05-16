"""SignalTilt: tilt a base weight vector by a cross-sectional signal z-score."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy
from aiam.strategy.equal_weight import EqualWeight


def momentum_signal_fn(returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional momentum: cumulative return over the available window."""
    return returns.sum()


class SignalTilt(PointInTimeStrategy):
    """Tilt base weights by cross-sectional signal z-score with long-only renormalization.

    weights = clip(base_weights + tilt_strength × zscore_cross_sectional(signal), 0)
    then renormalized to sum = 1. signal_fn maps a returns DataFrame to a per-asset pd.Series.
    """

    def __init__(
        self,
        signal_fn: Callable[[pd.DataFrame], pd.Series],
        tilt_strength: float = 0.5,
        base: PointInTimeStrategy | None = None,
        signal_lookback: int = 252,
    ) -> None:
        self.signal_fn = signal_fn
        self.tilt_strength = tilt_strength
        self.base = base if base is not None else EqualWeight()
        self.signal_lookback = signal_lookback

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        base_w = self.base.predict_weights(panel, asof)
        returns = panel.slice(asof, "returns", lookback=self.signal_lookback)
        signal = self.signal_fn(returns)
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[-1]
        zs = ((signal - signal.mean()) / signal.std()).fillna(0)
        w = base_w.add(self.tilt_strength * zs, fill_value=0).clip(lower=0)
        return (w / w.sum()).rename(asof)
