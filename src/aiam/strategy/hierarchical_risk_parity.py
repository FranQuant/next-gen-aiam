from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from aiam.data.panel import Panel
from aiam.strategy.base import PointInTimeStrategy


class HierarchicalRiskParity(PointInTimeStrategy):
    """HRP via Ward linkage + recursive bisection (Lopez de Prado 2016, SSRN 2708678).

    Deterministic — no solver. Correlation is derived from cov_estimator output
    rather than a separate returns.corr() call, so shrinkage carries through.
    """

    def __init__(self, cov_estimator: Callable, lookback: int = 252) -> None:
        self.cov_estimator = cov_estimator
        self.lookback = lookback

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        prices = panel.slice(asof, kind="prices", lookback=self.lookback + 1)
        returns = prices.pct_change().iloc[1:]

        thresh = 0.10 * len(returns)
        valid_cols = [c for c in returns.columns if returns[c].isna().sum() <= thresh]
        returns = returns[valid_cols].fillna(0.0)

        if len(returns) < self.lookback:
            n = len(valid_cols)
            return pd.Series(np.ones(n) / n, index=valid_cols, name=asof)

        cov = self.cov_estimator(returns)

        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        np.clip(corr, -1.0, 1.0, out=corr)

        n = corr.shape[0]
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0)
        dist_sq = squareform(dist, checks=False)
        link = linkage(dist_sq, method="ward")
        order = leaves_list(link)

        weights = np.ones(n)
        clusters = [list(order)]
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]

                def cluster_var(idx: list[int]) -> float:
                    w = np.zeros(n)
                    w[idx] = 1.0 / len(idx)
                    return float(w @ cov @ w)

                v_l, v_r = cluster_var(left), cluster_var(right)
                alpha = 1 - v_l / (v_l + v_r + 1e-8)
                weights[left] *= alpha
                weights[right] *= 1 - alpha
                new_clusters.extend([left, right])
            clusters = [c for c in new_clusters if len(c) > 1]

        w = weights / weights.sum()
        return pd.Series(w, index=valid_cols, name=asof)
