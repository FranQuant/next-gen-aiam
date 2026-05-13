from __future__ import annotations

import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy


class EqualWeight(PointInTimeStrategy):
    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        tickers = panel.universe_at(asof)
        w = 1.0 / len(tickers)
        return pd.Series(w, index=tickers, name=asof)
