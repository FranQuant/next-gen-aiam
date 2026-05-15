from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy

logger = logging.getLogger(__name__)


class RiskParity(PointInTimeStrategy):
    """Equal Risk Contribution (ERC) via least-squares SLSQP.

    Ref: Roncalli 2013; Qian 2005; Hilpisch §8.5 (weight_rp / paam_lab 19d).
    """

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

        if len(returns) < self.lookback:
            n = len(valid_cols)
            return pd.Series(np.ones(n) / n, index=valid_cols, name=asof)

        # Covariance is not annualised: scaling cov by any constant k leaves
        # the ERC optimal weights unchanged (both port_vol and mrc scale by k).
        cov = self.cov_estimator(returns)

        n = len(valid_cols)
        w0 = np.ones(n) / n
        bounds = [(1e-6, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        def risk_parity_obj(w: np.ndarray) -> float:
            port_vol = np.sqrt(w @ cov @ w)
            mrc = cov @ w / (port_vol + 1e-8)
            rc = w * mrc
            target = port_vol / n
            return float(np.sum((rc - target) ** 2))

        result = minimize(
            risk_parity_obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if not result.success:
            logger.warning(
                "RiskParity solver failed at asof=%s: %s", asof.date(), result.message
            )

        w = result.x if result.success else w0
        w = w / w.sum()

        return pd.Series(w, index=valid_cols, name=asof)
