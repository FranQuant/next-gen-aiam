"""RLAgent: wraps SimplexPolicy and satisfies PointInTimeStrategy.predict_weights contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.dl.workflow import set_global_seed
from aiam.rl import SEED
from aiam.rl.policy import SimplexPolicy


class RLAgent:
    """Holds a SimplexPolicy and translates panel data into policy calls.

    .predict_weights(panel, date) satisfies the PointInTimeStrategy contract:
    builds (N, F) feature state from trailing returns, runs policy.act, returns
    a pd.Series of weights indexed by asset name.

    optimizer is a placeholder for the Session 4b training loop.
    """

    def __init__(
        self,
        policy: SimplexPolicy,
        lookback: int = 20,
        seed: int = SEED,
    ) -> None:
        self.policy = policy
        self.lookback = lookback
        self.seed = seed
        self.optimizer: object | None = None  # set in Session 4b training

    def predict_weights(self, panel: Panel, date: pd.Timestamp) -> pd.Series:
        """Build state from panel at `date`, run policy, return per-asset weights."""
        date = pd.Timestamp(date)
        assets = panel.universe_at(date)
        n = len(assets)

        # Trailing return features: (N, lookback)
        prices = panel.slice(date, "prices", lookback=self.lookback + 1)
        rets = prices[assets].pct_change().dropna()
        if len(rets) >= self.lookback:
            feat = rets.iloc[-self.lookback:].values.T.astype(np.float32)  # (N, lookback)
        else:
            feat = np.zeros((n, self.lookback), dtype=np.float32)

        # Approximate current weights with equal weight (stateless contract).
        current_w = np.ones(n, dtype=np.float32) / n

        state = {"features": feat, "weights": current_w}
        set_global_seed(self.seed)
        weights = self.policy.act(state)  # (N,)
        return pd.Series(weights, index=assets, name=date)
