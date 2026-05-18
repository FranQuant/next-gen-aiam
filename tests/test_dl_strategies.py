"""Tests for src/aiam/strategy/dl_strategies.py — strategy wiring, multi-seed, no-leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.strategy.dl_strategies import (
    EnsembleDLSignalStrategy,
    MLPSignalStrategy,
    LSTMSignalStrategy,
    TransformerSignalStrategy,
)

# ── Synthetic data helpers ────────────────────────────────────────────────────

N_ASSETS = 4
FEATURE_COLS = ["f1", "f2", "f3"]
ASSETS = [f"A{i}" for i in range(N_ASSETS)]

_TRAIN_DATES = pd.bdate_range("2010-01-04", periods=120)
_TEST_DATES = pd.bdate_range(_TRAIN_DATES[-1] + pd.Timedelta("1D"), periods=15)
_ALL_DATES = _TRAIN_DATES.append(_TEST_DATES)
_TRAIN_END = str(_TRAIN_DATES[-1].date())

_TINY_SEEDS = (0, 1)


def _make_panels(seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([_ALL_DATES, ASSETS], names=["Date", "Asset"])
    n = len(idx)
    fp = pd.DataFrame({c: rng.standard_normal(n) for c in FEATURE_COLS}, index=idx)
    tp = pd.Series(rng.standard_normal(n), index=idx, name="target")
    return fp, tp


def _make_panel_obj():
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        (1 + rng.standard_normal((_ALL_DATES.shape[0], N_ASSETS)) * 0.01).cumprod(axis=0) * 100,
        index=_ALL_DATES,
        columns=ASSETS,
    )
    returns = prices.pct_change().fillna(0)
    return Panel({"prices": prices, "returns": returns})


_TINY_MLP_HP = dict(hidden_dims=(4,), max_epochs=3, patience=3, seeds=_TINY_SEEDS)
_TINY_SEQ_HP = dict(hidden_dim=4, max_epochs=3, patience=3, seeds=_TINY_SEEDS)
_TINY_TF_HP = dict(d_model=4, nhead=2, num_layers=1, max_epochs=3, patience=3, seeds=_TINY_SEEDS)


@pytest.fixture(scope="module")
def mlp_strat():
    fp, tp = _make_panels()
    return MLPSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END, **_TINY_MLP_HP)


@pytest.fixture(scope="module")
def lstm_strat():
    fp, tp = _make_panels()
    return LSTMSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END, **_TINY_SEQ_HP)


@pytest.fixture(scope="module")
def tf_strat():
    fp, tp = _make_panels()
    return TransformerSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END, **_TINY_TF_HP)


# ── Fit without error ─────────────────────────────────────────────────────────

def test_mlp_fits(mlp_strat):
    assert hasattr(mlp_strat, "_seed_ensemble")
    assert hasattr(mlp_strat, "predictions")


def test_lstm_fits(lstm_strat):
    assert hasattr(lstm_strat, "_seed_ensemble")
    assert len(lstm_strat._seed_ensemble.fits) == len(_TINY_SEEDS)


def test_transformer_fits(tf_strat):
    assert hasattr(tf_strat, "_seed_ensemble")


# ── Multi-seed ensemble has correct seed count ────────────────────────────────

def test_mlp_seed_count(mlp_strat):
    assert len(mlp_strat._seed_ensemble.fits) == len(_TINY_SEEDS)


# ── predict_weights validity ──────────────────────────────────────────────────

def test_mlp_predict_weights_sums_to_one(mlp_strat):
    panel = _make_panel_obj()
    w = mlp_strat._predict_weights(panel, _TEST_DATES[3])
    assert abs(w.sum() - 1.0) < 1e-9


def test_mlp_predict_weights_nonnegative(mlp_strat):
    panel = _make_panel_obj()
    w = mlp_strat._predict_weights(panel, _TEST_DATES[3])
    assert (w >= 0).all()


def test_lstm_predict_weights_sums_to_one(lstm_strat):
    panel = _make_panel_obj()
    # Use a test date that has enough lookback history (all _TEST_DATES should qualify)
    w = lstm_strat._predict_weights(panel, _TEST_DATES[3])
    assert abs(w.sum() - 1.0) < 1e-9


# ── No look-ahead ─────────────────────────────────────────────────────────────

def test_mlp_predictions_only_at_test_dates(mlp_strat):
    train_end_ts = pd.Timestamp(_TRAIN_END)
    pred_dates = mlp_strat.predictions.index.get_level_values(0).unique()
    assert all(d > train_end_ts for d in pred_dates)


def test_lstm_predictions_only_at_test_dates(lstm_strat):
    train_end_ts = pd.Timestamp(_TRAIN_END)
    pred_dates = lstm_strat.predictions.index.get_level_values(0).unique()
    assert all(d > train_end_ts for d in pred_dates)


def test_mlp_no_lookahead_train_size():
    """Training data covers pre-test dates only."""
    fp, tp = _make_panels()
    strat = MLPSignalStrategy(fp, tp, FEATURE_COLS, train_end=_TRAIN_END, **_TINY_MLP_HP)
    assert strat._X_train.shape[0] + strat._X_val.shape[0] == strat._n_pre_test_obs


# ── EnsembleDLSignalStrategy ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ensemble_strat(mlp_strat, lstm_strat):
    return EnsembleDLSignalStrategy([mlp_strat, lstm_strat])


def test_ensemble_predictions_is_mean(ensemble_strat, mlp_strat, lstm_strat):
    common_idx = mlp_strat.predictions.index.intersection(lstm_strat.predictions.index)
    expected = (mlp_strat.predictions.loc[common_idx] + lstm_strat.predictions.loc[common_idx]) / 2
    pd.testing.assert_series_equal(
        ensemble_strat.predictions.loc[common_idx].sort_index(),
        expected.sort_index().rename("pred"),
        rtol=1e-5,
    )


def test_ensemble_predict_weights_sums_to_one(ensemble_strat):
    panel = _make_panel_obj()
    w = ensemble_strat._predict_weights(panel, _TEST_DATES[3])
    assert abs(w.sum() - 1.0) < 1e-9


def test_ensemble_empty_strategies_raises():
    with pytest.raises(ValueError, match="non-empty"):
        EnsembleDLSignalStrategy([])


def test_ensemble_weighted_single_model(mlp_strat, lstm_strat):
    """weights=[1, 0] gives predictions equal to the first strategy at common indices."""
    ens = EnsembleDLSignalStrategy([mlp_strat, lstm_strat], weights=[1.0, 0.0])
    common = mlp_strat.predictions.index.intersection(lstm_strat.predictions.index)
    pd.testing.assert_series_equal(
        ens.predictions.loc[common].sort_index(),
        mlp_strat.predictions.loc[common].sort_index().rename("pred"),
        rtol=1e-5,
    )


def test_ensemble_equal_weights_is_default(mlp_strat, lstm_strat):
    """No weights → stored weights are all 1/n."""
    ens = EnsembleDLSignalStrategy([mlp_strat, lstm_strat])
    assert all(abs(w - 0.5) < 1e-9 for w in ens.weights)


def test_ensemble_weights_bad_sum_raises(mlp_strat, lstm_strat):
    """Weights not summing to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        EnsembleDLSignalStrategy([mlp_strat, lstm_strat], weights=[0.6, 0.6])
