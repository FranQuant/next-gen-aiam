from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from aiam.data.panel import Panel


class Strategy(ABC):
    def fit(self, panel: Panel, train_until: pd.Timestamp) -> None:
        pass

    @abstractmethod
    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series: ...


class PointInTimeStrategy(Strategy, ABC):
    _train_until: pd.Timestamp | None = None

    def fit(self, panel: Panel, train_until: pd.Timestamp) -> None:
        self._train_until = pd.Timestamp(train_until)

    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        asof = pd.Timestamp(asof)
        if self._train_until is not None and asof > self._train_until:
            raise ValueError(
                f"Look-ahead violation: asof={asof} > train_until={self._train_until}"
            )
        return self._predict_weights(panel, asof)

    @abstractmethod
    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series: ...
