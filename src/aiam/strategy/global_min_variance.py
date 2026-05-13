from __future__ import annotations

from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy


class GlobalMinVariance(PointInTimeStrategy):
    def __init__(self, cov_estimator: Callable, lookback: int = 252) -> None:
        self.cov_estimator = cov_estimator
        self.lookback = lookback

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        # Pull enough calendar rows to cover self.lookback business days, then
        # filter to weekdays so weekend NaN on US equities doesn't dominate.
        prices = panel.slice(asof, kind="prices", lookback=self.lookback * 2)
        prices = prices[prices.index.dayofweek < 5]   # Mon–Fri only
        prices = prices.iloc[-(self.lookback + 1):]
        returns = prices.pct_change().iloc[1:]

        # drop columns with > 10% NaN in the window
        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)

        cov = self.cov_estimator(returns)
        n = len(valid_cols)

        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if w.value is None:
            # fallback to equal weight if solver fails
            weights = np.ones(n) / n
        else:
            weights = np.maximum(w.value, 0.0)
            weights /= weights.sum()

        return pd.Series(weights, index=valid_cols, name=asof)
