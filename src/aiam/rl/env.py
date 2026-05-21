"""N-asset portfolio environment for RL training and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS: float = 10.0
LAMBDA_RISK: float = 0.02
_VOL_LOOKBACK: int = 21


class PortfolioEnv:
    """Discrete-time portfolio environment.

    State  : {'features': (N, F), 'weights': (N,)}
    Action : (N,) numpy array on the simplex (validated at step time)
    Reward : gross_return − transaction_cost − risk_penalty

    Features default to rolling `lookback`-day return windows for each asset.
    Pass a pre-computed (T, N, F) array via `features` to override.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        features: np.ndarray | None = None,
        lookback: int = 20,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        lambda_risk: float = LAMBDA_RISK,
    ) -> None:
        if start is not None or end is not None:
            mask = pd.Series(True, index=returns.index)
            if start is not None:
                mask &= returns.index >= pd.Timestamp(start)
            if end is not None:
                mask &= returns.index <= pd.Timestamp(end)
            returns = returns.loc[mask]

        self.assets: list[str] = returns.columns.tolist()
        self.dates: pd.DatetimeIndex = returns.index
        self.n_assets: int = len(self.assets)
        self._returns: np.ndarray = returns.values.astype(np.float32)  # (T, N)

        if features is not None:
            if features.shape[:2] != (len(returns), self.n_assets):
                raise ValueError(
                    f"features shape {features.shape} inconsistent with returns shape {self._returns.shape}"
                )
            self._features: np.ndarray = features.astype(np.float32)
        else:
            self._features = _rolling_return_features(self._returns, lookback)

        self.n_features: int = self._features.shape[2]

        self._vol_proxy: np.ndarray = _rolling_vol(self._returns, _VOL_LOOKBACK)
        self._lambda_risk: float = lambda_risk

        # First valid index: need both feature history and vol history.
        self._start_idx: int = max(lookback, _VOL_LOOKBACK)

        self._t: int = self._start_idx
        self._weights: np.ndarray = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset to initial state. Returns state dict."""
        self._t = self._start_idx
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return self._get_state()

    def state(self) -> dict:
        """Return current state without advancing the clock."""
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Apply action, advance one step, return (next_state, reward, done, info).

        Raises ValueError if action has negative values or does not sum to ~1.
        """
        action = np.asarray(action, dtype=np.float32)
        if np.any(action < -1e-6):
            raise ValueError(f"action contains negative values (min={action.min():.6f})")
        if abs(float(action.sum()) - 1.0) > 1e-4:
            raise ValueError(f"action must sum to 1.0, got {action.sum():.6f}")

        # Clip tiny negatives from floating-point noise, then renormalize.
        action = np.clip(action, 0.0, None)
        w_t: np.ndarray = action / action.sum()

        prev_w = self._weights

        if self._t + 1 >= len(self._returns):
            self._weights = w_t
            info = dict(gross_return=0.0, turnover=0.0, transaction_cost=0.0,
                        risk_penalty=0.0, net_return=0.0)
            return self._get_state(), 0.0, True, info

        r_next: np.ndarray = self._returns[self._t + 1]

        gross_return = float(w_t @ r_next)
        turnover = float(np.abs(w_t - prev_w).sum())
        transaction_cost = COST_BPS / 10_000.0 * turnover
        risk_penalty = self._lambda_risk * float(w_t @ self._vol_proxy[self._t])
        net_return = gross_return - transaction_cost - risk_penalty

        self._weights = w_t
        self._t += 1
        done = self._t + 1 >= len(self._returns)

        info = {
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "risk_penalty": risk_penalty,
            "net_return": net_return,
            "date": self.dates[self._t],
        }
        return self._get_state(), net_return, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        return {
            "features": self._features[self._t].copy(),
            "weights": self._weights.copy(),
        }


def _rolling_return_features(returns: np.ndarray, lookback: int) -> np.ndarray:
    """Build (T, N, lookback) feature array: each row is the last `lookback` returns."""
    T, N = returns.shape
    feats = np.zeros((T, N, lookback), dtype=np.float32)
    for t in range(lookback - 1, T):
        feats[t] = returns[t - lookback + 1 : t + 1].T
    return feats


def _rolling_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling std of returns: (T, N), zero-filled for early dates."""
    T, N = returns.shape
    vol = np.zeros((T, N), dtype=np.float32)
    for t in range(window, T):
        vol[t] = returns[t - window : t].std(axis=0)
    return vol
