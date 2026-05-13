from __future__ import annotations

import numpy as np
import pandas as pd


def equilibrium_only(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No views — BL collapses to MSR on equilibrium-implied returns."""
    n = returns.shape[1]
    return (np.zeros((0, n)), np.array([]), np.zeros((0, 0)))


def momentum_views(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    signal_lookback: int = 252,
    view_uncertainty_scaler: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One view per asset: expected return = annualized trailing return.

    Omega diagonal = view_uncertainty_scaler × sample daily variance.
    Small Omega relative to annualized cov → views are highly confident.
    """
    n = returns.shape[1]
    trailing = returns.iloc[-signal_lookback:]
    view_returns = ((1 + trailing).prod() - 1) * (252 / signal_lookback)
    P = np.eye(n)
    Q = view_returns.values
    Omega = np.diag(trailing.var().values * view_uncertainty_scaler)
    return (P, Q, Omega)


def mean_reversion_views(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    long_lookback: int = 1260,
    view_uncertainty_scaler: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """View: expected return = -1 × annualized long-run mean (~5y).

    Sign-flip expresses reversion: assets with high historical returns get
    lower expected returns going forward.
    """
    long_trailing = returns.iloc[-long_lookback:]
    view_returns = -long_trailing.mean().values * 252
    n = returns.shape[1]
    P = np.eye(n)
    Q = view_returns
    Omega = np.diag(long_trailing.var().values * view_uncertainty_scaler)
    return (P, Q, Omega)
