"""Tests for src/aiam/dl/walkforward.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.dl.losses import sharpe_loss
from aiam.dl.policies import MLPPolicy
from aiam.dl.policy_workflow import (
    DirectWeightSeedEnsemble,
    fit_direct_weight_seed_ensemble,
)
from aiam.dl.walkforward import (
    WalkForwardEnsemble,
    _compute_train_window,
    fit_walkforward_direct_weight,
    generate_refit_dates,
)

ASSETS = ["A", "B"]
N_FEATURES = 3


def _empty_ensemble(seed_id: int) -> DirectWeightSeedEnsemble:
    return DirectWeightSeedEnsemble(fits=[], seeds=(seed_id,))


def _make_feature_panel(n_dates: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    feature_cols = [f"f{j}" for j in range(N_FEATURES)]
    idx = pd.MultiIndex.from_product([dates, ASSETS], names=["date", "asset"])
    fp = pd.DataFrame(rng.standard_normal((len(idx), N_FEATURES)), index=idx, columns=feature_cols)
    tp = pd.Series(rng.standard_normal(len(idx)) * 0.01, index=idx, name="target")
    return fp, tp, feature_cols, dates


# ── WalkForwardEnsemble.ensemble_for_date ─────────────────────────────────────

def _make_wfe(n_refits: int = 3) -> tuple[WalkForwardEnsemble, list]:
    base = pd.Timestamp("2024-07-01")
    dates = [base + pd.DateOffset(months=i) for i in range(n_refits)]
    ensembles = [_empty_ensemble(i) for i in range(n_refits)]
    return WalkForwardEnsemble(refit_dates=dates, ensembles=ensembles), ensembles


def test_ensemble_for_date_before_all_raises():
    wfe, _ = _make_wfe()
    with pytest.raises(ValueError):
        wfe.ensemble_for_date(pd.Timestamp("2024-06-01"))


def test_ensemble_for_date_at_first():
    wfe, ens = _make_wfe()
    assert wfe.ensemble_for_date(pd.Timestamp("2024-07-01")) is ens[0]


def test_ensemble_for_date_between_first_and_second():
    wfe, ens = _make_wfe()
    assert wfe.ensemble_for_date(pd.Timestamp("2024-07-15")) is ens[0]


def test_ensemble_for_date_at_second():
    wfe, ens = _make_wfe()
    assert wfe.ensemble_for_date(pd.Timestamp("2024-08-01")) is ens[1]


def test_ensemble_for_date_after_all():
    wfe, ens = _make_wfe()
    assert wfe.ensemble_for_date(pd.Timestamp("2025-06-01")) is ens[-1]


# ── generate_refit_dates ──────────────────────────────────────────────────────

def test_monthly_22_dates():
    dates = generate_refit_dates(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2026-04-30"), "monthly"
    )
    assert len(dates) == 22
    assert dates[0] == pd.Timestamp("2024-07-01")
    assert dates[-1] == pd.Timestamp("2026-04-01")
    assert all(isinstance(d, pd.Timestamp) for d in dates)


def test_monthly_all_business_days():
    dates = generate_refit_dates(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2026-04-30"), "monthly"
    )
    for d in dates:
        assert d.dayofweek < 5, f"{d.date()} is not a business day"


def test_quarterly_count():
    dates = generate_refit_dates(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2026-04-30"), "quarterly"
    )
    assert len(dates) == 8


def test_generate_unknown_cadence_raises():
    with pytest.raises(ValueError, match="cadence"):
        generate_refit_dates(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), "biannual"
        )


def test_calendar_snapping():
    """Refit dates snap forward to nearest valid calendar day."""
    dates = generate_refit_dates(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30"), "monthly"
    )
    # Build a calendar that skips all Tuesdays
    all_bdays = pd.bdate_range("2024-06-01", "2024-10-31")
    calendar = all_bdays[all_bdays.dayofweek != 1]  # remove Tuesdays

    snapped = generate_refit_dates(
        pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30"), "monthly",
        calendar=calendar,
    )
    for d in snapped:
        assert d.dayofweek != 1, f"{d.date()} is a Tuesday (should have been snapped)"
        assert d in calendar


# ── _compute_train_window ─────────────────────────────────────────────────────

def test_compute_train_window():
    refit = pd.Timestamp("2024-07-01")
    train_start, train_end = _compute_train_window(refit, 24, 8)
    expected_end = refit - pd.DateOffset(days=8)
    expected_start = expected_end - pd.DateOffset(months=24)
    assert train_end == expected_end
    assert train_start == expected_start


def test_train_window_matches_spec():
    """train_start = refit - 24 months - 8 days; train_end = refit - 8 days."""
    refit = pd.Timestamp("2025-06-01")
    train_start, train_end = _compute_train_window(refit, 24, 8)
    assert train_end == pd.Timestamp("2025-05-24")
    assert train_start == pd.Timestamp("2023-05-24")


# ── fit_walkforward_direct_weight ─────────────────────────────────────────────

def test_fit_walkforward_returns_wfe():
    fp, tp, feature_cols, dates = _make_feature_panel(200)
    refit_1 = pd.Timestamp("2022-07-01")
    refit_2 = pd.Timestamp("2022-09-01")

    result = fit_walkforward_direct_weight(
        fp, tp, feature_cols, ASSETS,
        [refit_1, refit_2], MLPPolicy, "sharpe", seeds=[0],
        training_window_months=2, quarantine_days=5,
        n_features=N_FEATURES, n_assets=len(ASSETS),
        hidden_dims=(8,), max_epochs=5, patience=5,
    )
    assert isinstance(result, WalkForwardEnsemble)
    assert len(result.ensembles) == 2
    assert len(result.refit_dates) == 2


def test_fit_walkforward_each_has_one_seed():
    fp, tp, feature_cols, _ = _make_feature_panel(200)
    result = fit_walkforward_direct_weight(
        fp, tp, feature_cols, ASSETS,
        [pd.Timestamp("2022-07-01"), pd.Timestamp("2022-09-01")],
        MLPPolicy, "sharpe", seeds=[0],
        training_window_months=2, quarantine_days=5,
        n_features=N_FEATURES, n_assets=len(ASSETS),
        hidden_dims=(8,), max_epochs=5, patience=5,
    )
    assert len(result.ensembles[0].fits) == 1
    assert len(result.ensembles[1].fits) == 1


def test_fit_walkforward_refits_are_independent():
    """Two seeds → two independent fits per refit; different params expected."""
    fp, tp, feature_cols, _ = _make_feature_panel(200)
    result = fit_walkforward_direct_weight(
        fp, tp, feature_cols, ASSETS,
        [pd.Timestamp("2022-07-01"), pd.Timestamp("2022-09-01")],
        MLPPolicy, "sharpe", seeds=[0, 1],
        training_window_months=2, quarantine_days=5,
        n_features=N_FEATURES, n_assets=len(ASSETS),
        hidden_dims=(8,), max_epochs=3, patience=3,
    )
    import torch
    p0 = list(result.ensembles[0].fits[0].model.parameters())[0].detach()
    p1 = list(result.ensembles[0].fits[1].model.parameters())[0].detach()
    assert not torch.allclose(p0, p1), "Seeds 0 and 1 produced identical parameters"


# ── predict_weights routing ───────────────────────────────────────────────────

def test_predict_weights_routing():
    """predict_weights_for_date routes to the ensemble from the correct refit."""
    fp, tp, feature_cols, _ = _make_feature_panel(200)
    refit_1 = pd.Timestamp("2022-07-01")
    refit_2 = pd.Timestamp("2022-09-01")
    result = fit_walkforward_direct_weight(
        fp, tp, feature_cols, ASSETS,
        [refit_1, refit_2], MLPPolicy, "sharpe", seeds=[0],
        training_window_months=2, quarantine_days=5,
        n_features=N_FEATURES, n_assets=len(ASSETS),
        hidden_dims=(8,), max_epochs=3, patience=3,
    )

    X_dummy = np.random.randn(len(ASSETS), N_FEATURES).astype("float32")

    # Date between refit_1 and refit_2 → uses refit_1 ensemble
    mid_date = pd.Timestamp("2022-08-01")
    w_early = result.ensemble_for_date(mid_date).predict_weights(X_dummy)
    w_via_wfe = result.predict_weights_for_date(X_dummy, mid_date)
    np.testing.assert_array_equal(w_early, w_via_wfe)

    # Date on or after refit_2 → uses refit_2 ensemble
    late_date = pd.Timestamp("2022-09-15")
    w_late = result.ensemble_for_date(late_date).predict_weights(X_dummy)
    # Refit_1 and refit_2 ensembles were trained on different windows → differ
    assert not np.allclose(w_early, w_late)
