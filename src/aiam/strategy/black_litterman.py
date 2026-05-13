from __future__ import annotations

import logging
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class BlackLitterman(PointInTimeStrategy):
    def __init__(
        self,
        view_generator: Callable,
        cov_estimator: Callable,
        lookback: int = 252,
        tau: float = 0.05,
        delta: float = 2.5,
        prior_weights_method: str = "equal",
    ) -> None:
        self.view_generator = view_generator
        self.cov_estimator = cov_estimator
        self.lookback = lookback
        self.tau = tau
        self.delta = delta
        self.prior_weights_method = prior_weights_method

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
            cols = valid_cols if valid_cols else list(universe)
            k = len(cols)
            return pd.Series(1.0 / k, index=cols).reindex(universe, fill_value=0.0).rename(asof)

        # Annualized covariance — pi must be in annual units to match view Q.
        cov = self.cov_estimator(returns) * 252

        if self.prior_weights_method == "inverse_vol":
            vols = np.sqrt(np.diag(cov))
            inv_vol = 1.0 / np.maximum(vols, 1e-8)
            w_market = inv_vol / inv_vol.sum()
        else:
            w_market = np.ones(n) / n

        pi = self.delta * cov @ w_market

        P, Q, Omega = self.view_generator(returns, asof)

        if len(P) == 0:
            mu_post = pi
            cov_post = cov
        else:
            inv_term = np.linalg.inv(self.tau * cov)
            # M_prec is the precision of the posterior mean (see He & Litterman 1999)
            M_prec = inv_term + P.T @ np.linalg.inv(Omega) @ P
            M_cov = np.linalg.inv(M_prec)
            mu_post = M_cov @ (inv_term @ pi + P.T @ np.linalg.inv(Omega) @ Q)
            # Combined covariance: asset cov + posterior mean uncertainty
            cov_post = cov + M_cov

        mu_ex = mu_post  # rf = 0

        if (mu_ex > 0).sum() == 0:
            logger.warning(
                "BL fallback to EW at asof=%s: no positive posterior expected return "
                "across %d valid columns",
                asof.date(),
                n,
            )
            return _ew_fallback()

        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_post)))
        constraints = [mu_ex @ w == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status != "optimal":
            logger.warning(
                "BL solver non-optimal: %s at asof=%s", prob.status, asof.date()
            )

        if w.value is None or prob.status != "optimal":
            return _ew_fallback()

        weights = np.maximum(w.value, 0.0)
        weights /= weights.sum()

        return pd.Series(weights, index=valid_cols).reindex(universe, fill_value=0.0).rename(asof)
