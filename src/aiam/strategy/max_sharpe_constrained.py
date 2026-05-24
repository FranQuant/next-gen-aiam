from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class MaximumSharpeConstrained(PointInTimeStrategy):
    """MSR with per-asset weight bounds [lb, ub].

    Solved via scipy SLSQP (direct Sharpe maximization) rather than the
    Dantzig-Wolfe CVXPY parametrization, which doesn't support mixed
    lb/ub constraints cleanly on large universes.  Positions below lb are
    zeroed and weights renormalized (minimum-holding-size interpretation
    for lb; ub is a hard constraint).
    """

    def __init__(
        self,
        cov_estimator: Callable[[pd.DataFrame], np.ndarray],
        mean_estimator: Callable[[pd.DataFrame], np.ndarray],
        lookback: int = 252,
        rf: float = 0.0,
        bounds: tuple[float, float] = (0.05, 0.40),
    ) -> None:
        self.cov_estimator = cov_estimator
        self.mean_estimator = mean_estimator
        self.lookback = lookback
        self.rf = rf
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

        def _ew_fallback() -> pd.Series:
            w = pd.Series(1.0 / n, index=valid_cols)
            return w.reindex(universe, fill_value=0.0).rename(asof)

        if len(returns) < self.lookback:
            return _ew_fallback()

        cov = self.cov_estimator(returns)
        mu = self.mean_estimator(returns)
        mu_ex = mu - self.rf

        if (mu_ex > 0).sum() == 0:
            return _ew_fallback()

        def neg_sharpe(w):
            port_ret = np.dot(mu_ex, w)
            port_var = np.dot(w, np.dot(cov, w))
            port_vol = np.sqrt(max(port_var, 1e-14))
            return -port_ret / port_vol

        # Upper bound only in optimizer (lb * n > 1 would be infeasible as hard constraint)
        ub_eff = min(self.ub, 1.0)
        bounds_list = [(0.0, ub_eff)] * n
        x0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=bounds_list, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 1000})

        if not res.success and res.fun > 0:
            logger.warning("MSR_C SLSQP failed at asof=%s: %s", asof.date(), res.message)
            return _ew_fallback()

        weights = np.maximum(res.x, 0.0)

        # Soft lb: zero out positions below threshold, renormalize
        weights[weights < self.lb] = 0.0
        total = weights.sum()
        if total < 1e-8:
            return _ew_fallback()
        weights /= total

        w_series = pd.Series(weights, index=valid_cols)
        return w_series.reindex(universe, fill_value=0.0).rename(asof)
