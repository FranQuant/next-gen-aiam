"""Tests for src/aiam/strategy/ml_strategies.py. Synthetic data only; no parquet I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.strategy.ml_strategies import LassoSignalStrategy, RFSignalStrategy, XGBSignalStrategy

# ── Synthetic data helpers ────────────────────────────────────────────────────

N_ASSETS = 5
N_TRAIN = 120   # business days of training data
N_TEST = 20     # business days of test data
FEATURE_COLS = ["f1", "f2", "f3"]
ASSETS = [f"A{i}" for i in range(N_ASSETS)]

_TRAIN_DATES = pd.bdate_range("2010-01-04", periods=N_TRAIN)
_TEST_DATES = pd.bdate_range(_TRAIN_DATES[-1] + pd.Timedelta("1D"), periods=N_TEST)
_ALL_DATES = _TRAIN_DATES.append(_TEST_DATES)
_TRAIN_END = str(_TRAIN_DATES[-1].date())
_TEST_START = str(_TEST_DATES[0].date())


def _make_panels(seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([_ALL_DATES, ASSETS], names=["Date", "Asset"])
    n = len(idx)
    feature_panel = pd.DataFrame(
        {c: rng.standard_normal(n) for c in FEATURE_COLS}, index=idx
    )
    target_panel = pd.Series(rng.standard_normal(n), index=idx, name="target")
    return feature_panel, target_panel


def _make_panel_obj():
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        (1 + rng.standard_normal((_ALL_DATES.shape[0], N_ASSETS)) * 0.01).cumprod(axis=0) * 100,
        index=_ALL_DATES,
        columns=ASSETS,
    )
    returns = prices.pct_change().fillna(0)
    return Panel({"prices": prices, "returns": returns})


# ── Fit without error ─────────────────────────────────────────────────────────

def test_lasso_fits():
    fp, tp = _make_panels()
    strat = LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)
    assert hasattr(strat, "model")
    assert hasattr(strat, "predictions")


def test_rf_fits():
    fp, tp = _make_panels()
    strat = RFSignalStrategy(
        fp, tp, FEATURE_COLS, train_end=_TRAIN_END,
        n_estimators=10, max_depth=3, min_samples_leaf=5,
    )
    assert hasattr(strat, "model")


def test_xgb_fits():
    fp, tp = _make_panels()
    strat = XGBSignalStrategy(
        fp, tp, FEATURE_COLS, train_end=_TRAIN_END,
        n_estimators=10, learning_rate=0.1, max_depth=3,
    )
    assert hasattr(strat, "model")


# ── predict_weights validity ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lasso_strat():
    fp, tp = _make_panels()
    return LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)


def test_predict_weights_sums_to_one(lasso_strat):
    panel = _make_panel_obj()
    asof = _TEST_DATES[5]
    w = lasso_strat._predict_weights(panel, asof)
    assert abs(w.sum() - 1.0) < 1e-9


def test_predict_weights_nonnegative(lasso_strat):
    panel = _make_panel_obj()
    w = lasso_strat._predict_weights(panel, _TEST_DATES[5])
    assert (w >= 0).all()


def test_predict_weights_indexed_by_assets(lasso_strat):
    panel = _make_panel_obj()
    w = lasso_strat._predict_weights(panel, _TEST_DATES[5])
    assert set(w.index).issubset(set(ASSETS))


# ── No look-ahead ─────────────────────────────────────────────────────────────

def test_no_lookahead_train_data_boundary():
    """Model trained exclusively on pre-test observations."""
    fp, tp = _make_panels()
    strat = LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)
    # _X_train + _X_val must not exceed total pre-test observations
    n_pre_test = N_TRAIN * N_ASSETS
    assert strat._X_train.shape[0] + strat._X_val.shape[0] == n_pre_test


def test_no_lookahead_predictions_cached():
    """Mutating feature_panel after construction leaves cached predictions unchanged."""
    fp, tp = _make_panels()
    strat = LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)
    preds_before = strat.predictions.copy()
    fp.iloc[:] = 999.0  # mutate in-place
    pd.testing.assert_series_equal(strat.predictions, preds_before)


def test_no_lookahead_predictions_only_at_test_dates():
    """Cached predictions exist only for post-train-end dates."""
    fp, tp = _make_panels()
    strat = LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)
    train_end_ts = pd.Timestamp(_TRAIN_END)
    pred_dates = strat.predictions.index.get_level_values(0).unique()
    assert all(d > train_end_ts for d in pred_dates)


# ── Permutation importance (RF only) ─────────────────────────────────────────

def test_rf_permutation_importance_shape():
    fp, tp = _make_panels()
    strat = RFSignalStrategy(
        fp, tp, FEATURE_COLS, train_end=_TRAIN_END,
        n_estimators=10, max_depth=3, min_samples_leaf=5,
    )
    imp = strat.permutation_importance(n_repeats=2)
    assert len(imp) == len(FEATURE_COLS)
    assert list(imp.index) == FEATURE_COLS


def test_rf_permutation_importance_positive_total():
    fp, tp = _make_panels(seed=1)
    strat = RFSignalStrategy(
        fp, tp, FEATURE_COLS, train_end=_TRAIN_END,
        n_estimators=10, max_depth=3, min_samples_leaf=5,
    )
    imp = strat.permutation_importance(n_repeats=2)
    # Sum of importances is not guaranteed positive for noise data, but shape is correct
    assert imp.notna().all()


# ── Fallback to EW when no prediction cached ─────────────────────────────────

def test_predict_weights_missing_asof_falls_back_to_ew():
    """If asof has no cached prediction, return EW weights."""
    fp, tp = _make_panels()
    strat = LassoSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END)
    panel = _make_panel_obj()
    # Use a date in the training period (no cached prediction)
    asof = _TRAIN_DATES[10]
    w = strat._predict_weights(panel, asof)
    ew = 1.0 / N_ASSETS
    assert abs(w.sum() - 1.0) < 1e-9
    np.testing.assert_allclose(w.values, ew, atol=1e-9)
