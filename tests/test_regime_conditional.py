import numpy as np
import pandas as pd

from aiam.evaluation.regime_conditional import regime_conditional_performance


def test_regime_conditional_known_sharpe():
    """Per-regime Sharpe on a synthetic example with deterministic regime labels."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=120)

    # First 60 trading days = regime 0, last 60 = regime 1
    regime = pd.Series([0] * 60 + [1] * 60, index=dates)

    r0 = rng.normal(0.002, 0.01, 60)
    r1 = rng.normal(-0.001, 0.015, 60)
    returns = pd.Series(np.concatenate([r0, r1]), index=dates)

    result = regime_conditional_performance(
        {"strat": returns}, regime, n_regimes=2, min_days=3
    )
    sharpe = result["sharpe"]

    def expected_sharpe(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        return float(s.mean() * 252 / (s.std() * np.sqrt(252)))

    assert abs(sharpe.loc["strat", 0] - expected_sharpe(r0)) < 1e-9
    assert abs(sharpe.loc["strat", 1] - expected_sharpe(r1)) < 1e-9
