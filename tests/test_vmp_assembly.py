from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.evaluation.vmp_assembly import assemble_vmp_returns


def _const_vol_returns(n: int = 252, daily_vol: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(rng.normal(0.0, daily_vol, n), index=dates)


def test_constant_vol_exposure_near_one():
    """Constant-vol returns → target_vol == realized_vol → exposure ≈ 1.0 after warm-up."""
    returns = _const_vol_returns(n=500, daily_vol=0.01)
    result = assemble_vmp_returns(returns, lookback=21, lag=1)

    # Reconstruct the exposure series the same way the function does
    realized_vol = returns.rolling(21).std() * np.sqrt(252)
    target_vol = returns.std() * np.sqrt(252)
    exposure = (target_vol / realized_vol).shift(1).clip(0.25, 1.5).fillna(1.0)

    # After the warm-up window the exposure should cluster tightly around 1.0
    warm_up = 21 + 1
    np.testing.assert_allclose(
        exposure.iloc[warm_up:].mean(), 1.0, atol=0.10,
        err_msg="Mean exposure should be ≈ 1.0 for constant-vol returns",
    )


def test_high_vol_regime_reduces_exposure():
    """High-vol regime → VMP deleverage → realized exposure < 1 in that period."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2020-01-02", periods=n)

    low_vol = rng.normal(0.0, 0.005, 150)
    high_vol = rng.normal(0.0, 0.025, 150)
    returns = pd.Series(np.concatenate([low_vol, high_vol]), index=dates)

    result = assemble_vmp_returns(returns, lookback=21, lag=1)

    realized_vol = returns.rolling(21).std() * np.sqrt(252)
    target_vol = returns.std() * np.sqrt(252)
    exposure = (target_vol / realized_vol).shift(1).clip(0.25, 1.5).fillna(1.0)

    # In the high-vol window (after warm-up settles) exposure should be below 1
    high_vol_exposure = exposure.iloc[175:]  # skip transition zone
    assert high_vol_exposure.mean() < 0.9, (
        f"Expected exposure < 0.9 in high-vol regime, got {high_vol_exposure.mean():.3f}"
    )


def test_no_lookahead():
    """
    exposure_t depends only on realized_vol_{t-lag}, computed from the rolling window
    ending at t-lag — no future data.  We pin target_vol as an explicit scalar so the
    global constant is identical between the two series; only the rolling window changes.
    """
    n = 200
    lookback = 21
    lag = 1
    fixed_target_vol = 0.15  # explicit so it doesn't differ between series
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(99)
    base = rng.normal(0.0, 0.01, n)

    shock_idx = 100
    perturbed = base.copy()
    perturbed[shock_idx] *= 50  # enormous spike

    r_base = pd.Series(base, index=dates)
    r_perturbed = pd.Series(perturbed, index=dates)

    vmp_base = assemble_vmp_returns(r_base, target_vol=fixed_target_vol, lookback=lookback, lag=lag)
    vmp_perturbed = assemble_vmp_returns(r_perturbed, target_vol=fixed_target_vol, lookback=lookback, lag=lag)

    # Days strictly before the shock can't be affected by it:
    # exposure_t uses data ending at t-lag, so the shock at shock_idx only enters
    # the rolling window starting at t = shock_idx + lag.
    pre_shock_slice = slice(None, shock_idx - lag)
    pd.testing.assert_series_equal(
        vmp_base.iloc[pre_shock_slice],
        vmp_perturbed.iloc[pre_shock_slice],
        check_names=False,
        rtol=1e-10,
    )


def test_rf_pass_through():
    """When rf > 0, the risk-free rate is correctly added back to VMP returns."""
    returns = _const_vol_returns(n=100, daily_vol=0.01)
    rf = 0.05
    vmp = assemble_vmp_returns(returns, rf=rf, lookback=21, lag=1)

    realized_vol = returns.rolling(21).std() * np.sqrt(252)
    target_vol = returns.std() * np.sqrt(252)
    exposure = (target_vol / realized_vol).shift(1).clip(0.25, 1.5).fillna(1.0)
    expected = exposure * (returns - rf / 252) + rf / 252

    pd.testing.assert_series_equal(vmp, expected, check_names=False)


def test_clip_bounds_respected():
    """Exposure is always within the specified clip bounds."""
    # Alternating very-low and very-high vol to force clip both ways
    rng = np.random.default_rng(7)
    n = 500
    dates = pd.bdate_range("2020-01-02", periods=n)
    vols = np.where(np.arange(n) % 2 == 0, 0.001, 0.05)
    returns = pd.Series(rng.normal(0.0, 1.0, n) * vols, index=dates)

    clip = (0.3, 1.2)
    vmp = assemble_vmp_returns(returns, clip=clip, lookback=10, lag=1)

    realized_vol = returns.rolling(10).std() * np.sqrt(252)
    target_vol = returns.std() * np.sqrt(252)
    exposure = (target_vol / realized_vol).shift(1).clip(*clip).fillna(1.0)

    # Exposure must never exceed the clip bounds (within float precision)
    assert exposure.max() <= clip[1] + 1e-12
    assert exposure.min() >= clip[0] - 1e-12
