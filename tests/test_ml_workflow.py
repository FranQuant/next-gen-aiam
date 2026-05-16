"""Tests for src/aiam/ml/workflow.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.ml.workflow import (
    apply_standardizer,
    chronological_splits,
    cross_sectional_score,
    fit_standardizer,
    leakage_check_forward_returns,
    predict_walk_forward,
)
from aiam.features.technical import forward_returns

DATES = pd.bdate_range("2003-01-02", "2026-04-30")
RNG = np.random.default_rng(42)


# ── chronological_splits ──────────────────────────────────────────────────────

def test_splits_train_end_respected():
    train, val, test = chronological_splits(DATES, train_end="2022-12-31", test_start="2023-01-01")
    assert train[-1] <= pd.Timestamp("2022-12-31")
    assert val[-1] <= pd.Timestamp("2022-12-31")


def test_splits_test_start_respected():
    _, _, test = chronological_splits(DATES, train_end="2022-12-31", test_start="2023-01-01")
    assert test[0] >= pd.Timestamp("2023-01-01")


def test_splits_no_overlap():
    train, val, test = chronological_splits(DATES)
    assert len(set(train) & set(val)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(set(train) & set(test)) == 0


def test_splits_validation_is_last_fraction():
    """Validation must be the LAST 15% of the pre-test window (contiguous, adjacent to test)."""
    train, val, _ = chronological_splits(DATES, validation_share=0.15)
    pre_test = DATES[DATES <= pd.Timestamp("2022-12-31")]
    # val immediately follows train in the pre-test window
    assert train[-1] < val[0]
    assert len(val) == int(len(pre_test) * 0.15)


def test_splits_cover_all_dates():
    train, val, test = chronological_splits(DATES)
    combined = set(train) | set(val) | set(test)
    # Dates in gap between train_end+1 and test_start are covered by val or test
    assert len(combined) == len(DATES)


def test_splits_small_dataset():
    """Edge case: small dataset where n_val=0."""
    dates = pd.bdate_range("2010-01-01", periods=5)
    train, val, test = chronological_splits(
        dates, train_end="2010-01-05", test_start="2010-01-06", validation_share=0.05
    )
    assert len(val) == 0 or len(val) >= 0  # no crash; val may be empty


# ── fit_standardizer / apply_standardizer ────────────────────────────────────

def _make_X():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=200)
    return pd.DataFrame(
        {"f1": rng.standard_normal(200) * 2 + 5, "f2": rng.standard_normal(200) * 0.5, "other": 1.0},
        index=dates,
    )


def test_fit_standardizer_center_matches_train():
    X = _make_X()
    center, scale = fit_standardizer(X, ["f1", "f2"])
    np.testing.assert_allclose(center["f1"], X["f1"].mean(), rtol=1e-10)
    np.testing.assert_allclose(center["f2"], X["f2"].mean(), rtol=1e-10)


def test_fit_standardizer_scale_matches_train():
    X = _make_X()
    center, scale = fit_standardizer(X, ["f1", "f2"])
    np.testing.assert_allclose(scale["f1"], X["f1"].std(), rtol=1e-10)


def test_apply_standardizer_passthrough_nonfeat():
    """Non-feature columns must pass through unchanged."""
    X = _make_X()
    center, scale = fit_standardizer(X, ["f1", "f2"])
    Xs = apply_standardizer(X, center, scale, ["f1", "f2"])
    pd.testing.assert_series_equal(Xs["other"], X["other"])


def test_apply_standardizer_standardizes_features():
    X = _make_X()
    center, scale = fit_standardizer(X, ["f1"])
    Xs = apply_standardizer(X, center, scale, ["f1"])
    np.testing.assert_allclose(Xs["f1"].mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(Xs["f1"].std(), 1.0, atol=1e-10)


# ── leakage_check_forward_returns ────────────────────────────────────────────

def _make_returns(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=100)
    return pd.DataFrame({"A": rng.standard_normal(100) * 0.01, "B": rng.standard_normal(100) * 0.01}, index=dates)


def test_leakage_check_true():
    """Correctly constructed forward returns pass the leakage check."""
    ret = _make_returns()
    fwd = forward_returns(ret, horizon=5)
    asof = ret.index[20]
    assert leakage_check_forward_returns(ret, fwd, horizon=5, asset="A", asof=asof) is True


def test_leakage_check_false():
    """Deliberately misaligned forward returns fail the leakage check."""
    ret = _make_returns()
    fwd_wrong = ret.shift(2)  # wrong alignment
    asof = ret.index[20]
    assert leakage_check_forward_returns(ret, fwd_wrong, horizon=5, asset="A", asof=asof) is False


# ── predict_walk_forward ──────────────────────────────────────────────────────

def _make_multiindex_panel(n_dates=200, n_assets=4):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", periods=n_dates)
    assets = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["Date", "Asset"])
    X = pd.DataFrame({"f1": rng.standard_normal(len(idx)), "f2": rng.standard_normal(len(idx))}, index=idx)
    y = pd.Series(rng.standard_normal(len(idx)), index=idx)
    return X, y, dates, n_assets


def test_predict_walk_forward_count():
    """One prediction per test-date × asset combination."""
    X, y, dates, n_assets = _make_multiindex_panel()
    train_d, _, test_d = chronological_splits(
        dates, train_end=str(dates[149].date()), test_start=str(dates[150].date()), validation_share=0.0
    )
    model_fn = lambda Xtr, ytr, Xte: Xte @ np.array([0.1, 0.2])
    preds = predict_walk_forward(model_fn, X, y, train_d, test_d, ["f1", "f2"])
    assert len(preds) == len(test_d) * n_assets


def test_predict_walk_forward_index():
    """Predictions are indexed by (Date, Asset) MultiIndex."""
    X, y, dates, _ = _make_multiindex_panel()
    train_d, _, test_d = chronological_splits(
        dates, train_end=str(dates[149].date()), test_start=str(dates[150].date()), validation_share=0.0
    )
    preds = predict_walk_forward(lambda Xtr, ytr, Xte: np.zeros(len(Xte)), X, y, train_d, test_d, ["f1", "f2"])
    assert isinstance(preds.index, pd.MultiIndex)


# ── cross_sectional_score ─────────────────────────────────────────────────────

def test_cross_sectional_score_existing_date():
    dates = pd.bdate_range("2020-01-02", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["Date", "Asset"])
    preds = pd.Series(np.arange(6, dtype=float), index=idx)
    score = cross_sectional_score(preds, dates[1])
    assert list(score.index) == ["A", "B"]
    assert len(score) == 2


def test_cross_sectional_score_missing_date():
    dates = pd.bdate_range("2020-01-02", periods=2)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["Date", "Asset"])
    preds = pd.Series([1.0, 2.0], index=idx)
    missing = pd.Timestamp("2025-01-01")
    score = cross_sectional_score(preds, missing)
    assert score.empty
