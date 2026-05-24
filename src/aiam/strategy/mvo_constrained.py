from __future__ import annotations

import logging
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class MVOConstrained(PointInTimeStrategy):
    """Global Minimum Variance with per-asset weight upper bound.

    MVO (mean-variance optimization) in minimum-variance form with a
    hard upper bound and a soft minimum-holding-size lower bound (per-asset
    bounds [5%, 40%]).  The ub hard constraint prevents degenerate
    corner solutions (e.g., SHY concentration).  Positions below lb are
    zeroed post-optimization and weights renormalized.
    """

    def __init__(
        self,
        cov_estimator: Callable[[pd.DataFrame], np.ndarray],
        lookback: int = 252,
        bounds: tuple[float, float] = (0.05, 0.40),
    ) -> None:
        self.cov_estimator = cov_estimator
        self.lookback = lookback
        self.lb, self.ub = bounds

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        prices = panel.slice(asof, kind="prices", lookback=self.lookback * 2)
        prices = prices[prices.index.dayofweek < 5]
        prices = prices.iloc[-(self.lookback + 1):]
        returns = prices.pct_change().iloc[1:]

        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)

        universe = panel.universe_at(asof)
        n = len(valid_cols)

        if len(returns) < self.lookback:
            weights = np.ones(n) / n
            return pd.Series(weights, index=valid_cols, name=asof).reindex(universe, fill_value=0.0)

        cov = self.cov_estimator(returns)

        ub_eff = min(self.ub, 1.0)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        constraints = [cp.sum(w) == 1, w >= 0, w <= ub_eff]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status != "optimal":
            logger.warning(
                "%s solver status non-optimal: %s at asof=%s",
                type(self).__name__, prob.status, asof,
            )

        if w.value is None:
            weights = np.ones(n) / n
        else:
            weights = np.maximum(w.value, 0.0)
            # Soft lb: zero out positions below threshold, renormalize
            weights[weights < self.lb] = 0.0
            total = weights.sum()
            if total < 1e-8:
                weights = np.ones(n) / n
            else:
                weights /= total

        return (
            pd.Series(weights, index=valid_cols, name=asof)
            .reindex(universe, fill_value=0.0)
        )
