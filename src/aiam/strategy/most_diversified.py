from __future__ import annotations

import logging
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class MostDiversified(PointInTimeStrategy):
    def __init__(self, cov_estimator: Callable, lookback: int = 252) -> None:
        self.cov_estimator = cov_estimator
        self.lookback = lookback

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
            n = len(cols)
            return pd.Series(1.0 / n, index=cols).reindex(universe, fill_value=0.0).rename(asof)

        if len(returns) < self.lookback:
            return _ew_fallback()

        cov = self.cov_estimator(returns)
        sigma = np.sqrt(np.diag(cov))

        # drop near-zero-vol columns (degenerate assets cause unbounded QP)
        keep = sigma > 1e-8
        if not keep.all():
            cols_arr = np.array(valid_cols)
            valid_cols = list(cols_arr[keep])
            cov = cov[np.ix_(keep, keep)]
            sigma = sigma[keep]

        n = len(valid_cols)
        if n == 0:
            logger.warning("MDP fallback to EW at asof=%s: no non-degenerate columns", asof.date())
            return _ew_fallback()

        # Convex reformulation of max DR: minimize w'Σw s.t. w'σ=1, w>=0, then normalize.
        # Structurally identical to MSR with σ replacing (μ - rf).
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        constraints = [sigma @ w == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status != "optimal":
            logger.warning(
                "%s solver non-optimal: %s at asof=%s",
                type(self).__name__, prob.status, asof.date(),
            )

        if w.value is None or prob.status != "optimal":
            return _ew_fallback()

        weights = np.maximum(w.value, 0.0)
        total = weights.sum()
        if total < 1e-12:
            return _ew_fallback()
        weights /= total

        return pd.Series(weights, index=valid_cols).reindex(universe, fill_value=0.0).rename(asof)
