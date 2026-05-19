"""Tests for src/aiam/strategy/dl_policy_strategies.py — weights shape, sum, non-negativity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.strategy.dl_policy_strategies import (
    DirectWeightLSTMStrategy,
    DirectWeightMLPStrategy,
    DirectWeightShrinkageStrategy,
    DirectWeightTransformerStrategy,
)

N_ASSETS = 4
FEATURE_COLS = ["f1", "f2", "f3"]
ASSETS = [f"A{i}" for i in range(N_ASSETS)]

_TRAIN_DATES = pd.bdate_range("2010-01-04", periods=120)
_TEST_DATES = pd.bdate_range(_TRAIN_DATES[-1] + pd.Timedelta("1D"), periods=10)
_ALL_DATES = _TRAIN_DATES.append(_TEST_DATES)
_TRAIN_END = str(_TRAIN_DATES[-1].date())

_TINY_SEEDS = (0, 1)


def _make_panels(seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([_ALL_DATES, ASSETS], names=["Date", "Asset"])
    n = len(idx)
    fp = pd.DataFrame({c: rng.standard_normal(n) for c in FEATURE_COLS}, index=idx)
    tp = pd.Series(rng.standard_normal(n) * 0.01, index=idx, name="target")
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
    return DirectWeightMLPStrategy(fp, tp, FEATURE_COLS, ASSETS, train_end=_TRAIN_END, **_TINY_MLP_HP)


@pytest.fixture(scope="module")
def lstm_strat():
    fp, tp = _make_panels()
    return DirectWeightLSTMStrategy(fp, tp, FEATURE_COLS, ASSETS, train_end=_TRAIN_END, **_TINY_SEQ_HP)


@pytest.fixture(scope="module")
def tf_strat():
    fp, tp = _make_panels()
    return DirectWeightTransformerStrategy(fp, tp, FEATURE_COLS, ASSETS, train_end=_TRAIN_END, **_TINY_TF_HP)


@pytest.fixture(scope="module")
def shrinkage_strat():
    fp, tp = _make_panels()
    bw = np.ones(N_ASSETS) / N_ASSETS
    return DirectWeightShrinkageStrategy(
        fp, tp, FEATURE_COLS, ASSETS, train_end=_TRAIN_END,
        benchmark_w=bw, **_TINY_SEQ_HP,
    )


# ── MLP strategy ─────────────────────────────────────────────────────────────

def test_mlp_fits(mlp_strat):
    assert hasattr(mlp_strat, "_seed_ensemble")
    assert hasattr(mlp_strat, "_weight_cache")


def test_mlp_weights_shape(mlp_strat):
    panel = _make_panel_obj()
    date = _TEST_DATES[0]
    w = mlp_strat.predict_weights(panel, date)
    assert w.shape == (N_ASSETS,), f"Unexpected shape: {w.shape}"


def test_mlp_weights_sum_to_one(mlp_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = mlp_strat.predict_weights(panel, date)
        assert abs(w.sum() - 1.0) < 1e-5, f"Weights sum to {w.sum():.6f} on {date}"


def test_mlp_weights_nonnegative(mlp_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = mlp_strat.predict_weights(panel, date)
        assert (w >= -1e-9).all(), f"Negative weight found on {date}: {w.min()}"


# ── LSTM strategy ─────────────────────────────────────────────────────────────

def test_lstm_weights_sum_to_one(lstm_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = lstm_strat.predict_weights(panel, date)
        assert abs(w.sum() - 1.0) < 1e-5


def test_lstm_weights_nonnegative(lstm_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = lstm_strat.predict_weights(panel, date)
        assert (w >= -1e-9).all()


# ── Transformer strategy ──────────────────────────────────────────────────────

def test_transformer_weights_sum_to_one(tf_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = tf_strat.predict_weights(panel, date)
        assert abs(w.sum() - 1.0) < 1e-5


def test_transformer_weights_nonnegative(tf_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = tf_strat.predict_weights(panel, date)
        assert (w >= -1e-9).all()


# ── Shrinkage strategy ────────────────────────────────────────────────────────

def test_shrinkage_fits(shrinkage_strat):
    assert hasattr(shrinkage_strat, "_seed_ensemble")


def test_shrinkage_weights_sum_to_one(shrinkage_strat):
    panel = _make_panel_obj()
    for date in _TEST_DATES[:3]:
        w = shrinkage_strat.predict_weights(panel, date)
        assert abs(w.sum() - 1.0) < 1e-5


def test_shrinkage_uses_benchmark(shrinkage_strat):
    """Shrinkage strategy stores benchmark weights and uses them in inference."""
    assert hasattr(shrinkage_strat, "_benchmark_w_arr")
    bw = shrinkage_strat._benchmark_w_arr
    assert len(bw) == N_ASSETS
    assert abs(bw.sum() - 1.0) < 1e-5
