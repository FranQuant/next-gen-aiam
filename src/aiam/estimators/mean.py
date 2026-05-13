from __future__ import annotations

import numpy as np
import pandas as pd


def sample_mean(returns: pd.DataFrame) -> np.ndarray:
    return returns.mean().values
