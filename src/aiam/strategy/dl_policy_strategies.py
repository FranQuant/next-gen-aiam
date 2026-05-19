"""Direct-weight DL policy strategies: MLP, LSTM, Transformer as PointInTimeStrategy.

Each strategy trains a seed ensemble on init and returns normalized weight vectors.
Weights are renormalized to sum to 1.0 and clipped at zero at inference time.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch

from aiam.data.panel import Panel
from aiam.data.split import TRAIN_END
from aiam.dl.losses import crra_loss, crra_shrinkage_loss, sharpe_loss
from aiam.dl.policies import LSTMPolicy, MLPPolicy, TransformerPolicy
from aiam.dl.policy_workflow import (
    DirectWeightSeedEnsemble,
    build_policy_sequence_windows,
    fit_direct_weight_seed_ensemble,
)
from aiam.ml.workflow import apply_standardizer, chronological_splits, fit_standardizer
from aiam.strategy.base import PointInTimeStrategy

LossKind = Literal["sharpe", "crra", "crra_shrinkage"]

_DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _normalize_weights(raw: np.ndarray, assets: list[str]) -> pd.Series:
    """Clip to non-negative, normalize to sum=1. Falls back to EW on degenerate output."""
    w = np.clip(raw, 0.0, None)
    total = w.sum()
    if total < 1e-12:
        w = np.ones(len(assets)) / len(assets)
    else:
        w = w / total
    return pd.Series(w, index=assets)


def _make_loss_fn(
    loss_kind: LossKind,
    gamma: float,
    benchmark_w: np.ndarray | None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_kind == "sharpe":
        return sharpe_loss
    if loss_kind == "crra":
        return lambda w, r: crra_loss(w, r, gamma=gamma)
    if loss_kind == "crra_shrinkage":
        if benchmark_w is None:
            raise ValueError("benchmark_w required for crra_shrinkage loss")
        bw = torch.tensor(benchmark_w, dtype=torch.float32)
        return lambda w, r: crra_shrinkage_loss(w, r, bw, gamma=gamma)
    raise ValueError(f"Unknown loss_kind '{loss_kind}'")


class _DLPolicyBase(PointInTimeStrategy):
    """Base class for direct-weight DL policy strategies.

    Fits a seed ensemble on construction; predict_weights returns the ensemble-
    averaged weights normalized to sum to 1 and clipped at zero.
    Subclasses set _lookback (None=tabular, int=sequence) and implement _build_ensemble().
    """

    _lookback: int | None = None

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        assets: list[str],
        train_end: str = TRAIN_END,
        validation_share: float = 0.15,
        seeds: Sequence[int] = _DEFAULT_SEEDS,
        loss_kind: LossKind = "sharpe",
        gamma: float = 5.0,
        benchmark_w: np.ndarray | None = None,
    ) -> None:
        dates = feature_panel.index.get_level_values(0).unique().sort_values()
        train_end_ts = pd.Timestamp(train_end)
        after = dates[dates > train_end_ts]
        test_start = str(after[0].date()) if len(after) > 0 else str(train_end_ts.date())
        train_dates, val_dates, test_dates = chronological_splits(
            dates, train_end=train_end, test_start=test_start, validation_share=validation_share
        )

        d_idx = feature_panel.index.get_level_values(0)
        X_tr = feature_panel.loc[d_idx.isin(train_dates), feature_cols]
        X_va = feature_panel.loc[d_idx.isin(val_dates), feature_cols]

        center, scale = fit_standardizer(X_tr, feature_cols)
        self._center = center
        self._scale = scale
        self._feature_cols = feature_cols
        self._feature_panel = feature_panel
        self._target_panel = target_panel
        self._assets = list(assets)
        self._test_dates = test_dates
        self._train_dates = train_dates
        self._val_dates = val_dates
        self.seeds = tuple(seeds)

        bw = benchmark_w if benchmark_w is not None else np.ones(len(assets)) / len(assets)
        self._loss_fn = _make_loss_fn(loss_kind, gamma, bw)
        # crra_shrinkage uses sigmoid output (multiplier in [0,1]); others use relu
        self._activation = "sigmoid" if loss_kind == "crra_shrinkage" else "relu"

        if self._lookback is not None:
            self._std_panel = apply_standardizer(feature_panel, center, scale, feature_cols)
            self._X_train, self._y_train, _ = self._build_windows(train_dates)
            self._X_val, self._y_val, _ = self._build_windows(val_dates)
        else:
            self._std_panel = apply_standardizer(feature_panel, center, scale, feature_cols)
            self._X_train, self._y_train = self._tabular_Xy(train_dates)
            self._X_val, self._y_val = self._tabular_Xy(val_dates)

        self._seed_ensemble: DirectWeightSeedEnsemble = self._build_ensemble()
        self._weight_cache: dict[pd.Timestamp, pd.Series] = {}
        self._cache_test_weights()

    def _tabular_Xy(self, dates: pd.Index) -> tuple[np.ndarray, np.ndarray]:
        """Build aligned (X, y) for tabular policies.

        Each row is one (date, asset) observation. y is the full cross-section of
        all assets' realized returns on that date, repeated for each asset-row.
        Only dates with complete cross-sections (no NaN for any asset) are included.
        """
        index_names = list(self._feature_panel.index.names)
        date_name = index_names[0]
        asset_name = index_names[1] if len(index_names) > 1 else "asset"

        d_idx = self._target_panel.index.get_level_values(0)
        tp = self._target_panel.loc[d_idx.isin(dates)]
        wide = tp.unstack(level=1).reindex(columns=self._assets).dropna()

        valid_dates = set(wide.index)
        fp_d_idx = self._std_panel.index.get_level_values(0)
        fp_sub = self._std_panel.loc[fp_d_idx.isin(valid_dates), self._feature_cols]
        fp_reset = fp_sub.reset_index().rename(columns={date_name: "Date", asset_name: "asset"})
        fp_reset = fp_reset.sort_values(["Date", "asset"]).reset_index(drop=True)

        X_rows, y_rows = [], []
        for _, row in fp_reset.iterrows():
            date_val = pd.Timestamp(row["Date"])
            if date_val not in wide.index:
                continue
            X_rows.append(row[self._feature_cols].to_numpy(dtype="float32"))
            y_rows.append(wide.loc[date_val].to_numpy(dtype="float32"))

        if not X_rows:
            return (
                np.empty((0, len(self._feature_cols)), dtype="float32"),
                np.empty((0, len(self._assets)), dtype="float32"),
            )
        return np.stack(X_rows), np.stack(y_rows)

    def _build_windows(
        self, dates: pd.Index
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        return build_policy_sequence_windows(
            self._std_panel,
            self._target_panel,
            self._feature_cols,
            self._assets,
            self._lookback,
            allowed_dates=set(dates),
        )

    def _build_ensemble(self) -> DirectWeightSeedEnsemble:
        raise NotImplementedError

    def _cache_test_weights(self) -> None:
        dates_with_data = set(
            self._feature_panel.index.get_level_values(0)
        ) & set(self._test_dates)
        for date in sorted(dates_with_data):
            try:
                w = self._infer_weights(date)
                self._weight_cache[date] = w
            except Exception:
                continue

    def _infer_weights(self, date: pd.Timestamp) -> pd.Series:
        if self._lookback is not None:
            return self._infer_seq(date)
        return self._infer_tabular(date)

    def _infer_tabular(self, date: pd.Timestamp) -> pd.Series:
        try:
            X_date = self._feature_panel.xs(date, level=0)[self._feature_cols]
        except KeyError:
            return pd.Series(1.0 / len(self._assets), index=self._assets)
        X_std = apply_standardizer(X_date, self._center, self._scale, self._feature_cols)
        raw = self._seed_ensemble.predict_weights(X_std.values.astype("float32"))
        mean_raw = raw.mean(axis=0)
        return _normalize_weights(mean_raw, self._assets)

    def _infer_seq(self, date: pd.Timestamp) -> pd.Series:
        windows, valid_assets = [], []
        for asset in self._assets:
            try:
                asset_slice = self._std_panel.xs(asset, level=1).sort_index()
            except KeyError:
                continue
            pos = asset_slice.index.searchsorted(date)
            if pos >= len(asset_slice) or asset_slice.index[pos] != date:
                continue
            start = pos - self._lookback + 1
            if start < 0:
                continue
            window = asset_slice.iloc[start : pos + 1][self._feature_cols].values.astype("float32")
            if window.shape[0] != self._lookback:
                continue
            windows.append(window)
            valid_assets.append(asset)

        if not windows:
            return pd.Series(1.0 / len(self._assets), index=self._assets)

        X_batch = np.stack(windows, axis=0)
        raw = self._seed_ensemble.predict_weights(X_batch)
        mean_raw = raw.mean(axis=0)
        return _normalize_weights(mean_raw, self._assets)

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        if asof in self._weight_cache:
            return self._weight_cache[asof]
        return self._infer_weights(asof)


class DirectWeightMLPStrategy(_DLPolicyBase):
    """MLP direct-weight policy. Tabular input (batch, n_features) → (batch, n_assets)."""

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        assets: list[str],
        *,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 12,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self._hp = dict(
            hidden_dims=hidden_dims, dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, device=device,
        )
        self._n_assets = len(assets)
        super().__init__(feature_panel, target_panel, feature_cols, assets, **kwargs)

    def _build_ensemble(self) -> DirectWeightSeedEnsemble:
        return fit_direct_weight_seed_ensemble(
            MLPPolicy,
            self._X_train, self._y_train,
            self._X_val, self._y_val,
            self._loss_fn,
            seeds=self.seeds,
            n_features=len(self._feature_cols),
            n_assets=self._n_assets,
            activation=self._activation,
            **self._hp,
        )


class DirectWeightLSTMStrategy(_DLPolicyBase):
    """LSTM direct-weight policy. Sequence input (batch, lookback, n_features) → (batch, n_assets)."""

    _lookback = 63

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        assets: list[str],
        *,
        hidden_dim: int = 24,
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 12,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self._hp = dict(
            hidden_dim=hidden_dim, dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, device=device,
        )
        self._n_assets = len(assets)
        super().__init__(feature_panel, target_panel, feature_cols, assets, **kwargs)

    def _build_ensemble(self) -> DirectWeightSeedEnsemble:
        n_features = len(self._feature_cols)
        return fit_direct_weight_seed_ensemble(
            LSTMPolicy,
            self._X_train, self._y_train,
            self._X_val, self._y_val,
            self._loss_fn,
            seeds=self.seeds,
            n_features=n_features,
            n_assets=self._n_assets,
            activation=self._activation,
            **self._hp,
        )


class DirectWeightTransformerStrategy(_DLPolicyBase):
    """Transformer direct-weight policy. Sequence input → (batch, n_assets)."""

    _lookback = 63

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        assets: list[str],
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 12,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self._hp = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, device=device,
        )
        self._n_assets = len(assets)
        super().__init__(feature_panel, target_panel, feature_cols, assets, **kwargs)

    def _build_ensemble(self) -> DirectWeightSeedEnsemble:
        n_features = len(self._feature_cols)
        return fit_direct_weight_seed_ensemble(
            TransformerPolicy,
            self._X_train, self._y_train,
            self._X_val, self._y_val,
            self._loss_fn,
            seeds=self.seeds,
            n_features=n_features,
            n_assets=self._n_assets,
            activation=self._activation,
            **self._hp,
        )


class DirectWeightShrinkageStrategy(_DLPolicyBase):
    """LSTM direct-weight policy with CRRA shrinkage loss.

    Network output acts as a per-asset multiplier on benchmark_w (Sigmoid activation).
    Effective weights = sigmoid_output * benchmark_w, normalized at inference.
    """

    _lookback = 63

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        assets: list[str],
        *,
        benchmark_w: np.ndarray | None = None,
        hidden_dim: int = 24,
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 12,
        gamma: float = 5.0,
        device: str = "cpu",
        seeds: Sequence[int] = _DEFAULT_SEEDS,
        **kwargs,
    ) -> None:
        n = len(assets)
        bw = benchmark_w if benchmark_w is not None else np.ones(n) / n
        self._hp = dict(
            hidden_dim=hidden_dim, dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, device=device,
        )
        self._n_assets = n
        self._benchmark_w_arr = bw
        super().__init__(
            feature_panel, target_panel, feature_cols, assets,
            seeds=seeds, loss_kind="crra_shrinkage", gamma=gamma,
            benchmark_w=bw, **kwargs,
        )

    def _build_ensemble(self) -> DirectWeightSeedEnsemble:
        n_features = len(self._feature_cols)
        return fit_direct_weight_seed_ensemble(
            LSTMPolicy,
            self._X_train, self._y_train,
            self._X_val, self._y_val,
            self._loss_fn,
            seeds=self.seeds,
            n_features=n_features,
            n_assets=self._n_assets,
            activation="sigmoid",
            **self._hp,
        )

    def _infer_weights(self, date: pd.Timestamp) -> pd.Series:
        raw_w = super()._infer_weights(date)
        effective = raw_w.values * self._benchmark_w_arr
        return _normalize_weights(effective, self._assets)
