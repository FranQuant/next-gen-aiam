from __future__ import annotations

import logging

import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class SwitchingStrategy(PointInTimeStrategy):
    """Regime-conditional strategy-of-strategies.

    Dispatches to a child PointInTimeStrategy based on the current dominant
    regime read from the panel.  The child's public predict_weights is called
    (not _predict_weights) so every child independently enforces its own
    asof ≤ train_until contract.

    Regime data is monthly; slicing kind="regimes" with no lookback returns all
    rows ≤ asof, so .iloc[-1] of dominant_regime is the most-recent month —
    implicitly forward-filling the regime across intra-month daily rebalances.
    """

    def __init__(
        self,
        switching_rule: dict[int, PointInTimeStrategy],
        default_strategy: PointInTimeStrategy,
        regime_kind: str = "regimes",
        regime_col: str = "dominant_regime",
    ) -> None:
        self.switching_rule = switching_rule
        self.default_strategy = default_strategy
        self.regime_kind = regime_kind
        self.regime_col = regime_col

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        regime_df = panel.slice(asof, kind=self.regime_kind)
        regime_series = regime_df[self.regime_col].dropna()

        if regime_series.empty:
            logger.warning("SwitchingStrategy: no regime data at asof=%s; using default", asof.date())
            return self.default_strategy.predict_weights(panel, asof)

        current_regime = int(regime_series.iloc[-1])
        strategy = self.switching_rule.get(current_regime, self.default_strategy)
        return strategy.predict_weights(panel, asof)
