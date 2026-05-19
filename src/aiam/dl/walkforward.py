"""Walk-forward refit orchestration for direct-weight portfolio policies."""
from __future__ import annotations

import bisect
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from aiam.dl.policy_workflow import DirectWeightSeedEnsemble, fit_direct_weight_seed_ensemble


@dataclass
class WalkForwardEnsemble:
    """Collection of DirectWeightSeedEnsembles, one per refit date.

    Routes predictions to the ensemble fit at the most-recent refit date <= query date.
    """

    refit_dates: list[pd.Timestamp]   # sorted ascending
    ensembles: list[DirectWeightSeedEnsemble]

    def ensemble_for_date(self, date: pd.Timestamp) -> DirectWeightSeedEnsemble:
        """Return ensemble from most-recent refit_date <= date (O(log n))."""
        idx = bisect.bisect_right(self.refit_dates, date) - 1
        if idx < 0:
            raise ValueError(
                f"date {date.date()} precedes earliest refit {self.refit_dates[0].date()}"
            )
        return self.ensembles[idx]

    def predict_weights_for_date(
        self, X_for_date: np.ndarray, date: pd.Timestamp
    ) -> np.ndarray:
        """Predict weights using the appropriate refit's ensemble."""
        return self.ensemble_for_date(date).predict_weights(X_for_date)


def generate_refit_dates(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    cadence: str = "monthly",
    calendar: Optional[pd.DatetimeIndex] = None,
) -> list[pd.Timestamp]:
    """Generate refit dates spanning the test period.

    Cadences: 'monthly' (first business day of each month), 'quarterly'
    (first business day of each quarter: Jan/Apr/Jul/Oct), 'weekly' (each Monday).

    First refit_date is at or before test_start; last is at or before test_end.
    If calendar is provided, each candidate is snapped forward to the nearest valid
    trading day.
    """
    _FREQ = {"monthly": "BMS", "quarterly": "BQS", "weekly": "W-MON"}
    if cadence not in _FREQ:
        raise ValueError(f"Unknown cadence {cadence!r}. Use 'monthly', 'quarterly', or 'weekly'.")

    gen_start = test_start - pd.DateOffset(months=2)
    all_cands = pd.date_range(start=gen_start, end=test_end, freq=_FREQ[cadence])

    before = all_cands[all_cands <= test_start]
    after = all_cands[(all_cands > test_start) & (all_cands <= test_end)]
    refit_dates = ([before[-1]] if len(before) else []) + list(after)

    if calendar is not None:
        snapped = []
        for rd in refit_dates:
            future = calendar[calendar >= rd]
            if len(future):
                snapped.append(future[0])
        refit_dates = snapped

    return [pd.Timestamp(d) for d in refit_dates]


def _compute_train_window(
    refit_date: pd.Timestamp,
    training_window_months: int,
    quarantine_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (train_start, train_end) for a single refit date."""
    train_end = refit_date - pd.DateOffset(days=quarantine_days)
    train_start = train_end - pd.DateOffset(months=training_window_months)
    return train_start, train_end


def _build_tabular_Xy(
    feature_panel: pd.DataFrame,
    target_panel: pd.Series,
    feature_cols: list[str],
    assets: list[str],
    dates: set,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for tabular training: one row per (date, asset), y = cross-asset returns."""
    fp_dates = feature_panel.index.get_level_values(0)
    tp_dates = target_panel.index.get_level_values(0)

    tgt_sub = target_panel.loc[tp_dates.isin(dates)]
    wide_y = tgt_sub.unstack(level=1).reindex(columns=assets).dropna()
    valid_dates = set(wide_y.index)

    fp_sub = feature_panel.loc[fp_dates.isin(valid_dates), feature_cols].reset_index()
    date_col, asset_col = fp_sub.columns[0], fp_sub.columns[1]
    fp_sub = fp_sub.sort_values([date_col, asset_col]).reset_index(drop=True)

    row_dates = pd.to_datetime(fp_sub[date_col])
    X = fp_sub[feature_cols].to_numpy(dtype="float32")
    y = wide_y.loc[row_dates.values].to_numpy(dtype="float32")

    valid = ~np.isnan(y).any(axis=1)
    return X[valid], y[valid]


def _make_loss_fn(loss_kind: str, benchmark_w: Optional[np.ndarray]):
    from aiam.dl.losses import crra_loss, crra_shrinkage_loss, sharpe_loss

    if loss_kind == "sharpe":
        return sharpe_loss
    if loss_kind == "crra":
        return lambda w, r: crra_loss(w, r, gamma=5.0)
    if loss_kind == "crra_shrinkage":
        if benchmark_w is None:
            raise ValueError("benchmark_w is required for loss_kind='crra_shrinkage'")
        import torch
        bw = torch.tensor(np.asarray(benchmark_w, dtype="float32"))
        return lambda w, r: crra_shrinkage_loss(w, r, bw, gamma=5.0)
    raise ValueError(f"Unknown loss_kind {loss_kind!r}. Use 'sharpe', 'crra', or 'crra_shrinkage'.")


def fit_walkforward_direct_weight(
    feature_panel: pd.DataFrame,
    target_panel: pd.Series,
    feature_cols: list[str],
    assets: list[str],
    refit_dates: list[pd.Timestamp],
    policy_class: type,
    loss_kind: str,
    seeds: Sequence[int],
    training_window_months: int = 24,
    validation_share: float = 0.15,
    quarantine_days: int = 8,
    benchmark_w: Optional[np.ndarray] = None,
    device: str = "cpu",
    verbose: bool = False,
    **train_kwargs,
) -> WalkForwardEnsemble:
    """Fit one DirectWeightSeedEnsemble per refit date; return as WalkForwardEnsemble.

    For each refit date D:
    - Training window: [D - training_window_months - quarantine_days, D - quarantine_days]
    - Last validation_share of window dates used for early stopping
    - Each refit is completely independent (no weight sharing across refits)

    train_kwargs are forwarded verbatim to fit_direct_weight_seed_ensemble
    (and on to both fit_direct_weight_policy and policy_class.__init__).
    """
    loss_fn = _make_loss_fn(loss_kind, benchmark_w)
    fp_dates = feature_panel.index.get_level_values(0)

    ensembles: list[DirectWeightSeedEnsemble] = []
    for i, refit_date in enumerate(refit_dates):
        train_start, train_end = _compute_train_window(
            refit_date, training_window_months, quarantine_days
        )

        window_dates = sorted(set(
            feature_panel.index[
                (fp_dates >= train_start) & (fp_dates <= train_end)
            ].get_level_values(0)
        ))

        n_val = max(1, int(len(window_dates) * validation_share))
        n_tr = len(window_dates) - n_val
        train_dates = set(window_dates[:n_tr])
        val_dates = set(window_dates[n_tr:])

        X_tr, y_tr = _build_tabular_Xy(feature_panel, target_panel, feature_cols, assets, train_dates)
        X_va, y_va = _build_tabular_Xy(feature_panel, target_panel, feature_cols, assets, val_dates)

        t0 = time.time()
        ens = fit_direct_weight_seed_ensemble(
            policy_class, X_tr, y_tr, X_va, y_va, loss_fn,
            seeds=seeds, device=device, **train_kwargs,
        )
        ensembles.append(ens)

        if verbose:
            val_losses = [fr.summary["best_val_loss"] for fr in ens.fits]
            print(
                f"  Refit {i + 1}/{len(refit_dates)}: "
                f"{train_start.date()} → {train_end.date()} "
                f"({n_tr} train, {len(val_dates)} val dates) | "
                f"val_loss={np.mean(val_losses):.4f} | {time.time() - t0:.1f}s"
            )

    return WalkForwardEnsemble(refit_dates=list(refit_dates), ensembles=ensembles)
