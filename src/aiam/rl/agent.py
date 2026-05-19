"""RLAgent: wraps SimplexPolicy and satisfies PointInTimeStrategy.predict_weights contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from aiam.data.panel import Panel
from aiam.dl.workflow import set_global_seed
from aiam.rl import SEED
from aiam.rl.policy import SimplexPolicy
from aiam.rl.trainer import TrainConfig, TrainHistory, train


class RLAgent:
    """Holds a SimplexPolicy and translates panel data into policy calls.

    .predict_weights(panel, date) satisfies the PointInTimeStrategy contract:
    builds (N, F) feature state from trailing returns, runs policy.act, returns
    a pd.Series of weights indexed by asset name.

    .fit(env, config) runs REINFORCE+baseline training and stores history.
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
        self.history: TrainHistory | None = None
        self._value_head = None

    def fit(
        self,
        env,  # PortfolioEnv — avoid circular import by not type-annotating
        config: TrainConfig | None = None,
        use_value_baseline: bool = True,
        temperature: float = 1.0,
    ) -> TrainHistory:
        """Train policy via REINFORCE with optional value baseline.

        Mutates self.policy in-place. Returns training history.
        """
        if config is None:
            config = TrainConfig(seed=self.seed)
        history, value_head = train(
            self.policy, env, config,
            use_value_baseline=use_value_baseline,
            temperature=temperature,
        )
        self.history = history
        self._value_head = value_head
        return history

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location="cpu"))
        self.policy.eval()

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
