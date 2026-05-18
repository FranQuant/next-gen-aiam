"""DL signal strategies: MLP, LSTM, Transformer wrapped as PointInTimeStrategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.dl.workflow import (
    SeedEnsembleResult,
    build_sequence_windows,
    fit_lstm_regressor,
    fit_mlp_regressor,
    fit_transformer_regressor,
    fit_with_seed_ensemble,
)
from aiam.ml.workflow import (
    apply_standardizer,
    chronological_splits,
    cross_sectional_score,
    fit_standardizer,
)
from aiam.strategy.base import PointInTimeStrategy


def _signal_tilt_weights(
    predictions: pd.Series,
    panel: Panel,
    asof: pd.Timestamp,
    tilt_strength: float,
) -> pd.Series:
    assets = panel.universe_at(asof)
    n = len(assets)
    base_w = pd.Series(1.0 / n, index=assets)
    score = cross_sectional_score(predictions, asof)
    if score.empty:
        return base_w.rename(asof)
    score = score.reindex(assets).fillna(0.0)
    std = score.std()
    zs = (score - score.mean()) / std if std > 1e-12 else pd.Series(0.0, index=assets)
    w = (base_w + tilt_strength * zs).clip(lower=0.0)
    total = w.sum()
    return (w / total if total > 1e-12 else base_w).rename(asof)


def _build_std_long_frame(
    feature_panel: pd.DataFrame,
    target_panel: pd.Series,
    feature_cols: list[str],
    center: pd.Series,
    scale: pd.Series,
    train_dates: pd.Index,
    val_dates: pd.Index,
    test_dates: pd.Index,
) -> pd.DataFrame:
    std_fp = apply_standardizer(feature_panel, center, scale, feature_cols)
    fp_reset = std_fp.reset_index()
    tp_reset = target_panel.reset_index()
    tp_reset = tp_reset.rename(columns={tp_reset.columns[-1]: "target"})
    level_names = list(feature_panel.index.names)
    merged = fp_reset.merge(tp_reset, on=level_names)
    merged = merged.rename(columns={level_names[0]: "Date", level_names[1]: "asset"})
    train_set, val_set, test_set = set(train_dates), set(val_dates), set(test_dates)
    merged["split"] = merged["Date"].map(
        lambda d: "train" if d in train_set else ("validation" if d in val_set else ("test" if d in test_set else "other"))
    )
    return merged


class _DLSignalBase(PointInTimeStrategy):
    """Shared scaffolding for DL signal strategies. Multi-seed ensemble; single-fit on train window.

    Subclasses set `_lookback` (None = tabular, int = sequence) and implement `_build_ensemble()`.
    """

    _lookback: int | None = None

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        train_end: str = "2022-12-31",
        validation_share: float = 0.15,
        tilt_strength: float = 0.5,
        seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
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
        y_d_idx = target_panel.index.get_level_values(0)
        y_tr = target_panel.loc[y_d_idx.isin(train_dates)]
        y_va = target_panel.loc[y_d_idx.isin(val_dates)]

        center, scale = fit_standardizer(X_tr, feature_cols)
        self._center = center
        self._scale = scale
        self._feature_cols = feature_cols
        self._feature_panel = feature_panel
        self._test_dates = test_dates
        self._train_dates = train_dates
        self._val_dates = val_dates
        self._n_pre_test_obs = (len(train_dates) + len(val_dates)) * feature_panel.index.get_level_values(1).nunique()
        self.tilt_strength = tilt_strength
        self.seeds = seeds

        if self._lookback is not None:
            std_frame = _build_std_long_frame(
                feature_panel, target_panel, feature_cols,
                center, scale, train_dates, val_dates, test_dates,
            )
            X_tr_seq, y_tr_seq, _ = build_sequence_windows(
                std_frame, feature_cols, "target", self._lookback, allowed_splits=("train",)
            )
            X_va_seq, y_va_seq, _ = build_sequence_windows(
                std_frame, feature_cols, "target", self._lookback, allowed_splits=("validation",)
            )
            self._X_train: np.ndarray = X_tr_seq
            self._y_train: np.ndarray = y_tr_seq
            self._X_val: np.ndarray = X_va_seq
            self._y_val: np.ndarray = y_va_seq
            self._std_panel = apply_standardizer(feature_panel, center, scale, feature_cols)
        else:
            self._X_train = apply_standardizer(X_tr, center, scale, feature_cols).values.astype("float32")
            self._y_train = y_tr.values.astype("float32")
            self._X_val = apply_standardizer(X_va, center, scale, feature_cols).values.astype("float32")
            self._y_val = y_va.values.astype("float32")

        self._seed_ensemble: SeedEnsembleResult = self._build_ensemble()
        self._cache_test_predictions()

    def _build_ensemble(self) -> SeedEnsembleResult:
        raise NotImplementedError

    def _cache_test_predictions(self) -> None:
        records: dict[pd.Timestamp, pd.Series] = {}
        for date in self._test_dates:
            try:
                X_date = self._feature_panel.xs(date, level=0)[self._feature_cols]
            except KeyError:
                continue
            assets = list(X_date.index)

            if self._lookback is None:
                X_std = apply_standardizer(X_date, self._center, self._scale, self._feature_cols)
                preds = self._seed_ensemble.predict_mean(X_std.values.astype("float32"))
                records[date] = pd.Series(preds, index=assets, name="pred")
            else:
                windows, valid_assets = self._build_sequence_batch(assets, date)
                if not windows:
                    continue
                X_batch = np.stack(windows, axis=0)
                preds = self._seed_ensemble.predict_mean(X_batch)
                records[date] = pd.Series(preds, index=valid_assets, name="pred")

        if records:
            self.predictions = pd.concat(records)
            self.predictions.index.names = ["Date", "Asset"]
        else:
            self.predictions = pd.Series(dtype=float, name="pred")

    def _build_sequence_batch(
        self, assets: list[str], date: pd.Timestamp
    ) -> tuple[list[np.ndarray], list[str]]:
        windows, valid_assets = [], []
        for asset in assets:
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
        return windows, valid_assets

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        return _signal_tilt_weights(self.predictions, panel, asof, self.tilt_strength)


class MLPSignalStrategy(_DLSignalBase):
    """MLP on 17-feature cross-section. Input shape (batch, n_features); multi-seed ensemble."""

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        *,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 120,
        patience: int = 15,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self._hp = dict(
            hidden_dims=hidden_dims, dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, device=device,
        )
        super().__init__(feature_panel, target_panel, feature_cols, **kwargs)

    def _build_ensemble(self) -> SeedEnsembleResult:
        return fit_with_seed_ensemble(
            fit_mlp_regressor,
            dict(X_train=self._X_train, y_train=self._y_train, X_val=self._X_val, y_val=self._y_val, **self._hp),
            seeds=self.seeds,
        )


class LSTMSignalStrategy(_DLSignalBase):
    """Per-asset LSTM. Input shape (batch, lookback=63, n_features); multi-seed ensemble."""

    _lookback = 63

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
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
        super().__init__(feature_panel, target_panel, feature_cols, **kwargs)

    def _build_ensemble(self) -> SeedEnsembleResult:
        return fit_with_seed_ensemble(
            fit_lstm_regressor,
            dict(X_train_seq=self._X_train, y_train=self._y_train, X_val_seq=self._X_val, y_val=self._y_val, **self._hp),
            seeds=self.seeds,
        )


class TransformerSignalStrategy(_DLSignalBase):
    """Per-asset Transformer encoder. Input shape (batch, lookback=63, n_features); multi-seed ensemble."""

    _lookback = 63

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
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
        super().__init__(feature_panel, target_panel, feature_cols, **kwargs)

    def _build_ensemble(self) -> SeedEnsembleResult:
        return fit_with_seed_ensemble(
            fit_transformer_regressor,
            dict(X_train_seq=self._X_train, y_train=self._y_train, X_val_seq=self._X_val, y_val=self._y_val, **self._hp),
            seeds=self.seeds,
        )


class EnsembleDLSignalStrategy(PointInTimeStrategy):
    """Equal-weighted average of multiple DL signal strategies fed into a signal-tilt wrapper.

    Decision 6 (option a): library class rather than notebook-level construction.
    Session 3c recommendation: consolidate with ML ensemble into a shared aiam.strategy.ensemble module.
    """

    def __init__(
        self,
        strategies: list[_DLSignalBase],
        tilt_strength: float = 0.5,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must be non-empty")
        self.strategies = strategies
        self.tilt_strength = tilt_strength

        all_preds = [s.predictions for s in strategies]
        stacked = pd.concat(all_preds, axis=1, join="inner")
        self.predictions = stacked.mean(axis=1).rename("pred")
        self.predictions.index.names = ["Date", "Asset"]

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        return _signal_tilt_weights(self.predictions, panel, asof, self.tilt_strength)
