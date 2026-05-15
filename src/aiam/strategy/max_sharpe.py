from __future__ import annotations

import logging
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class MaximumSharpe(PointInTimeStrategy):
    def __init__(
        self,
        cov_estimator: Callable[[pd.DataFrame], np.ndarray],
        mean_estimator: Callable[[pd.DataFrame], np.ndarray],
        lookback: int = 252,
        rf: float = 0.0,
    ) -> None:
        self.cov_estimator = cov_estimator
        self.mean_estimator = mean_estimator
        self.lookback = lookback
        self.rf = rf

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
            n = len(valid_cols)
            w = pd.Series(1.0 / n, index=valid_cols)
            return w.reindex(universe, fill_value=0.0).rename(asof)

        if len(returns) < self.lookback:
            return _ew_fallback()

        cov = self.cov_estimator(returns)
        mu = self.mean_estimator(returns)
        mu_ex = mu - self.rf

        if (mu_ex > 0).sum() == 0:
            logger.warning(
                "MSR fallback to EW at asof=%s: no positive excess return "
                "across %d valid columns",
                asof.date(),
                len(valid_cols),
            )
            return _ew_fallback()

        n = len(valid_cols)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        constraints = [mu_ex @ w == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status != "optimal":
            logger.warning(
                "%s solver status non-optimal: %s at asof=%s",
                type(self).__name__, prob.status, asof,
            )

        if w.value is None or prob.status != "optimal":
            logger.warning(
                "MSR solver fallback to EW at asof=%s: status=%s, %d valid columns",
                asof.date(),
                prob.status,
                len(valid_cols),
            )
            return _ew_fallback()

        weights = np.maximum(w.value, 0.0)
        weights /= weights.sum()

        w_series = pd.Series(weights, index=valid_cols)
        return w_series.reindex(universe, fill_value=0.0).rename(asof)
