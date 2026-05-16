"""ML workflow helpers: chronological splits, standardization, leakage check, walk-forward predict."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def chronological_splits(
    panel_dates: pd.Index,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    validation_share: float = 0.15,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """Split dates chronologically into train / validation / test.

    Validation is the last `validation_share` fraction of the pre-test window, contiguous and
    adjacent to test. Returns (train_dates, validation_dates, test_dates). NO shuffling.

    For the paper-locked split: train_end=2022-12-31, test_start=2023-01-01, validation_share=0.15:
    - train:      2003-01-02 to ~2019-09-30  (~17 years)
    - validation: ~2019-10-01 to 2022-12-31  (~3.25 years, last 15% of pre-test)
    - test:       2023-01-01 to end           (~3.25 years)
    """
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    pre_test = panel_dates[panel_dates <= train_end_ts]
    test = panel_dates[panel_dates >= test_start_ts]
    n_val = int(len(pre_test) * validation_share)
    train = pre_test[:-n_val] if n_val > 0 else pre_test
    val = pre_test[-n_val:] if n_val > 0 else pre_test[:0]
    return train, val, test


def fit_standardizer(
    X_train: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Compute per-feature mean and std on training data ONLY.

    Returns (center, scale) Series indexed by feature_cols. Zeros in scale replaced by 1.
    """
    center = X_train[feature_cols].mean()
    scale = X_train[feature_cols].std().replace(0.0, 1.0)
    return center, scale


def apply_standardizer(
    X: pd.DataFrame, center: pd.Series, scale: pd.Series, feature_cols: list[str]
) -> pd.DataFrame:
    """Apply (X - center) / scale to feature_cols. Other columns pass through unchanged."""
    result = X.copy()
    result[feature_cols] = (X[feature_cols] - center) / scale
    return result


def leakage_check_forward_returns(
    returns: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int,
    asset: str,
    asof: pd.Timestamp,
) -> bool:
    """Verify forward_returns[asof, asset] == returns.shift(-horizon).rolling(horizon).sum()[asof, asset].

    Returns True on match within 1e-10. Catches off-by-one errors in target alignment.
    """
    expected = returns[asset].shift(-horizon).rolling(horizon).sum().loc[asof]
    actual = forward_returns[asset].loc[asof]
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    return abs(float(expected) - float(actual)) < 1e-10


def predict_walk_forward(
    model_fn: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    train_dates: pd.Index,
    test_dates: pd.Index,
    feature_cols: list[str],
) -> pd.Series:
    """Walk-forward predictions: single fit on train_dates, predict on test_dates.

    Handles MultiIndex (Date, Asset) and flat DatetimeIndex. Returns Series indexed by
    test observations. model_fn signature: (X_train_arr, y_train_arr, X_test_arr) -> y_pred_arr.
    """
    def _date_mask(index: pd.Index, dates: pd.Index) -> pd.Series:
        level = index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index
        return level.isin(dates)

    X_tr = X.loc[_date_mask(X.index, train_dates), feature_cols]
    X_te = X.loc[_date_mask(X.index, test_dates), feature_cols]
    y_tr = y.loc[_date_mask(y.index, train_dates)]

    center, scale = fit_standardizer(X_tr, feature_cols)
    X_tr_s = apply_standardizer(X_tr, center, scale, feature_cols)
    X_te_s = apply_standardizer(X_te, center, scale, feature_cols)
    preds = model_fn(X_tr_s.values, y_tr.values, X_te_s.values)
    return pd.Series(preds, index=X_te.index)


def cross_sectional_score(predictions: pd.Series, date: pd.Timestamp) -> pd.Series:
    """Extract the cross-section of predictions at `date`, indexed by asset.

    Returns empty Series when no prediction exists for `date`.
    """
    try:
        return predictions.xs(date, level=0)
    except KeyError:
        return pd.Series(dtype=float)
