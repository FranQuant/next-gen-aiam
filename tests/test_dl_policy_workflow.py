"""Tests for src/aiam/dl/policy_workflow.py — training, ensemble, window building."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.dl.losses import sharpe_loss, crra_loss
from aiam.dl.policies import MLPPolicy, LSTMPolicy
from aiam.dl.policy_workflow import (
    DirectWeightFitResult,
    DirectWeightSeedEnsemble,
    build_policy_sequence_windows,
    fit_direct_weight_policy,
    fit_direct_weight_seed_ensemble,
)

N_ASSETS = 4
N_FEATURES = 3
LOOKBACK = 8


def _toy_tabular(n_train: int = 64, n_val: int = 16, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, N_FEATURES)).astype("float32")
    X_va = rng.standard_normal((n_val, N_FEATURES)).astype("float32")
    y_tr = (rng.standard_normal((n_train, N_ASSETS)) * 0.01).astype("float32")
    y_va = (rng.standard_normal((n_val, N_ASSETS)) * 0.01).astype("float32")
    return X_tr, y_tr, X_va, y_va


def _toy_seq(n_train: int = 64, n_val: int = 16, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, LOOKBACK, N_FEATURES)).astype("float32")
    X_va = rng.standard_normal((n_val, LOOKBACK, N_FEATURES)).astype("float32")
    y_tr = (rng.standard_normal((n_train, N_ASSETS)) * 0.01).astype("float32")
    y_va = (rng.standard_normal((n_val, N_ASSETS)) * 0.01).astype("float32")
    return X_tr, y_tr, X_va, y_va


# ── fit_direct_weight_policy ──────────────────────────────────────────────────

def test_fit_returns_result_object():
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    result = fit_direct_weight_policy(
        MLPPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss,
        seed=0, max_epochs=3, patience=3,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    assert isinstance(result, DirectWeightFitResult)
    assert isinstance(result.history, pd.DataFrame)
    assert "epoch" in result.history.columns and "val_loss" in result.history.columns
    assert isinstance(result.summary, dict)
    assert "best_epoch" in result.summary and "seed" in result.summary


def test_fit_model_output_shape():
    import torch
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    result = fit_direct_weight_policy(
        MLPPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss,
        seed=0, max_epochs=2, patience=2,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    result.model.eval()
    x = torch.randn(4, N_FEATURES)
    with torch.no_grad():
        out = result.model(x)
    assert out.shape == (4, N_ASSETS)


def test_fit_lstm_policy():
    X_tr, y_tr, X_va, y_va = _toy_seq()
    result = fit_direct_weight_policy(
        LSTMPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss,
        seed=1, max_epochs=3, patience=3,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dim=8,
    )
    assert isinstance(result, DirectWeightFitResult)


def test_fit_crra_loss():
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    result = fit_direct_weight_policy(
        MLPPolicy, X_tr, y_tr, X_va, y_va,
        lambda w, r: crra_loss(w, r, gamma=5.0),
        seed=0, max_epochs=3, patience=3,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    assert result.summary["n_epochs_trained"] >= 1


# ── fit_direct_weight_seed_ensemble ──────────────────────────────────────────

def test_ensemble_trains_n_seeds():
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    seeds = (0, 1, 2)
    ens = fit_direct_weight_seed_ensemble(
        MLPPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss, seeds=seeds,
        max_epochs=2, patience=2,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    assert isinstance(ens, DirectWeightSeedEnsemble)
    assert len(ens.fits) == 3
    assert ens.seeds == (0, 1, 2)


def test_ensemble_predict_shape():
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    ens = fit_direct_weight_seed_ensemble(
        MLPPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss, seeds=(0, 1),
        max_epochs=2, patience=2,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    X_test = X_va[:4]
    preds = ens.predict_weights(X_test)
    assert preds.shape == (4, N_ASSETS)


def test_ensemble_seeds_produce_independent_models():
    """Different seeds should produce different model parameters (at least after init)."""
    X_tr, y_tr, X_va, y_va = _toy_tabular()
    ens = fit_direct_weight_seed_ensemble(
        MLPPolicy, X_tr, y_tr, X_va, y_va, sharpe_loss, seeds=(0, 1),
        max_epochs=1, patience=1,
        n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,),
    )
    import torch
    p0 = list(ens.fits[0].model.parameters())[0].detach()
    p1 = list(ens.fits[1].model.parameters())[0].detach()
    assert not torch.allclose(p0, p1), "Seeds 0 and 1 produced identical first-layer weights"


# ── build_policy_sequence_windows ────────────────────────────────────────────

def _make_panel_data(n_dates: int = 50, n_assets: int = N_ASSETS, seed: int = 0):
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, assets], names=["Date", "Asset"])
    feature_cols = [f"f{j}" for j in range(N_FEATURES)]
    fp = pd.DataFrame(rng.standard_normal((len(idx), N_FEATURES)), index=idx, columns=feature_cols)
    tp = pd.Series(rng.standard_normal(len(idx)) * 0.01, index=idx, name="target")
    return fp, tp, feature_cols, assets, dates


def test_build_windows_x_shape():
    fp, tp, feature_cols, assets, dates = _make_panel_data()
    X, y, meta = build_policy_sequence_windows(fp, tp, feature_cols, assets, lookback=LOOKBACK)
    assert X.ndim == 3 and X.shape[1] == LOOKBACK and X.shape[2] == N_FEATURES


def test_build_windows_y_shape():
    fp, tp, feature_cols, assets, dates = _make_panel_data()
    X, y, meta = build_policy_sequence_windows(fp, tp, feature_cols, assets, lookback=LOOKBACK)
    assert y.ndim == 2 and y.shape[1] == N_ASSETS
    assert X.shape[0] == y.shape[0]


def test_build_windows_meta_columns():
    fp, tp, feature_cols, assets, dates = _make_panel_data()
    _, _, meta = build_policy_sequence_windows(fp, tp, feature_cols, assets, lookback=LOOKBACK)
    for col in ("Date", "asset"):
        assert col in meta.columns, f"Missing column '{col}' in meta"


def test_build_windows_empty_on_impossible_lookback():
    """lookback > total dates should produce zero-length arrays."""
    fp, tp, feature_cols, assets, dates = _make_panel_data(n_dates=5)
    X, y, meta = build_policy_sequence_windows(fp, tp, feature_cols, assets, lookback=100)
    assert X.shape[0] == 0 and y.shape[0] == 0


def test_build_windows_allowed_dates_filter():
    """allowed_dates should restrict which terminal dates are included."""
    fp, tp, feature_cols, assets, dates = _make_panel_data(n_dates=40)
    all_X, all_y, _ = build_policy_sequence_windows(fp, tp, feature_cols, assets, lookback=LOOKBACK)
    subset_dates = set(dates[20:30])
    sub_X, sub_y, sub_meta = build_policy_sequence_windows(
        fp, tp, feature_cols, assets, lookback=LOOKBACK, allowed_dates=subset_dates
    )
    assert sub_X.shape[0] <= all_X.shape[0]
    assert all(pd.Timestamp(d) in subset_dates for d in sub_meta["Date"])
