"""RLStrategy: thin PointInTimeStrategy wrapper around a frozen RLAgent."""
from __future__ import annotations

import pandas as pd

from aiam.data.panel import Panel
from aiam.rl.agent import RLAgent
from aiam.strategy.base import PointInTimeStrategy


class RLStrategy(PointInTimeStrategy):
    """Delegates predict_weights to an RLAgent (trained or untrained).

    When the agent is untrained the policy returns weights from random
    initialization — useful as a contract sanity check before Session 4b.
    """

    def __init__(self, agent: RLAgent) -> None:
        self.agent = agent

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        return self.agent.predict_weights(panel, asof)
