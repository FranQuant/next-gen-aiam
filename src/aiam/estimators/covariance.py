from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


def oas_cov(returns: pd.DataFrame) -> np.ndarray:
    oas = OAS()
    oas.fit(returns.values)
    return oas.covariance_
