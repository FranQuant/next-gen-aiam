"""Raw lagged-return and realized-power feature extraction (JPM Cheng & Wu 2024 spec)."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def get_raw_feature_cols(lookback: int, include_rp: bool) -> list[str]:
    """Return ordered list of feature column names for the given raw-history spec."""
    cols = [f"r_lag_{k:03d}" for k in range(lookback)]
    if include_rp:
        cols += [f"rp_lag_{k:03d}" for k in range(lookback)]
    return cols


def extract_raw_history(
    returns_wide: pd.DataFrame,
    realized_power_wide: Optional[pd.DataFrame] = None,
    lookback: int = 252,
) -> pd.DataFrame:
    """Extract raw lagged history as a feature panel.

    For each (date, asset) row, produces `lookback` return lags and optionally
    `lookback` realized-power lags, following Cheng & Wu (J.P. Morgan, 2024).

    Column naming: 'r_lag_000' = return on day t (current day), 'r_lag_001' = day t-1, ...
    If RP included: 'rp_lag_000', 'rp_lag_001', ... appended after all 'r_lag_*'.

    Returns a long-format DataFrame with MultiIndex (date, asset).
    Rows where any lag column is NaN (early dates without full lookback) are dropped.
    """
    if realized_power_wide is not None:
        if not returns_wide.index.equals(realized_power_wide.index):
            raise ValueError(
                "returns_wide and realized_power_wide must have identical date indices"
            )

    r_shifts = {f"r_lag_{k:03d}": returns_wide.shift(k) for k in range(lookback)}
    all_shifted = pd.concat(r_shifts, axis=1)  # MultiIndex cols: (lag_name, asset)

    if realized_power_wide is not None:
        rp_shifts = {f"rp_lag_{k:03d}": realized_power_wide.shift(k) for k in range(lookback)}
        all_shifted = pd.concat([all_shifted, pd.concat(rp_shifts, axis=1)], axis=1)

    all_shifted.index.name = "date"
    result = all_shifted.stack(level=1)  # (date, asset) MultiIndex; cols = lag names
    result.index.names = ["date", "asset"]
    return result.dropna()
