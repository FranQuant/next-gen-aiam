"""ML signal strategies: Lasso, RandomForest, XGBoost wrapped as PointInTimeStrategy."""
from __future__ import annotations

import pandas as pd

from aiam.data.panel import Panel
from aiam.ml.workflow import (
    apply_standardizer,
    chronological_splits,
    cross_sectional_score,
    fit_standardizer,
)
from aiam.strategy.base import PointInTimeStrategy


class _MLSignalBase(PointInTimeStrategy):
    """Shared scaffolding for ML signal strategies.

    The model is fit ONCE on the full training window at construction. Predictions at each
    `asof` date are cached for all test dates; predict_weights looks up the cache and applies
    an EW-base SignalTilt with tilt_strength=0.5.
    """

    def __init__(
        self,
        feature_panel: pd.DataFrame,
        target_panel: pd.Series,
        feature_cols: list[str],
        train_end: str = "2022-12-31",
        validation_share: float = 0.15,
    ) -> None:
        dates = feature_panel.index.get_level_values(0).unique().sort_values()
        train_end_ts = pd.Timestamp(train_end)
        # derive test_start as first panel date after train_end
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
        self._X_train = apply_standardizer(X_tr, center, scale, feature_cols).values
        self._y_train = y_tr.values
        self._X_val = apply_standardizer(X_va, center, scale, feature_cols).values
        self._y_val = y_va.values
        self._center = center
        self._scale = scale
        self._feature_cols = feature_cols
        self._feature_panel = feature_panel
        self._test_dates = test_dates

        self.fit()
        self._cache_test_predictions()

    def fit(self) -> None:
        """Subclass must set self.model and call self.model.fit(self._X_train, self._y_train)."""
        raise NotImplementedError

    def _cache_test_predictions(self) -> None:
        records: dict[pd.Timestamp, pd.Series] = {}
        for date in self._test_dates:
            try:
                X_date = self._feature_panel.xs(date, level=0)[self._feature_cols]
            except KeyError:
                continue
            X_std = apply_standardizer(X_date, self._center, self._scale, self._feature_cols)
            preds = pd.Series(self.model.predict(X_std.values), index=X_date.index, name="pred")
            records[date] = preds
        if records:
            self.predictions = pd.concat(records)
            self.predictions.index.names = ["Date", "Asset"]
        else:
            self.predictions = pd.Series(dtype=float, name="pred")

    def _predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series:
        assets = panel.universe_at(asof)
        n = len(assets)
        base_w = pd.Series(1.0 / n, index=assets)
        score = cross_sectional_score(self.predictions, asof)
        if score.empty:
            return base_w.rename(asof)
        score = score.reindex(assets).fillna(0.0)
        std = score.std()
        zs = (score - score.mean()) / std if std > 1e-12 else pd.Series(0.0, index=assets)
        w = (base_w + 0.5 * zs).clip(lower=0.0)
        total = w.sum()
        return (w / total if total > 1e-12 else base_w).rename(asof)


class LassoSignalStrategy(_MLSignalBase):
    """Lasso regression on engineered features + asset-class one-hot.

    Predicts 21-day forward returns; cross-sectional z-score fed into EW-base SignalTilt.
    """

    def __init__(self, *args, alpha: float = 1e-4, **kwargs) -> None:
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def fit(self) -> None:
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.alpha, random_state=42, max_iter=10_000)
        self.model.fit(self._X_train, self._y_train)


class RFSignalStrategy(_MLSignalBase):
    """Random Forest for cross-sectional return prediction (Hilpisch §14 + JPM ML Quant).

    Defaults are conservative for noisy financial data: 100 trees, max_depth=8, min_samples_leaf=50.
    """

    def __init__(
        self,
        *args,
        n_estimators: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 50,
        **kwargs,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        super().__init__(*args, **kwargs)

    def fit(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(self._X_train, self._y_train)

    def permutation_importance(self, n_repeats: int = 5) -> pd.Series:
        """Sklearn permutation importance on the validation set.

        Returns Series indexed by feature name; higher = more important.
        """
        from sklearn.inspection import permutation_importance as _pi

        result = _pi(self.model, self._X_val, self._y_val, n_repeats=n_repeats, random_state=42)
        return pd.Series(result.importances_mean, index=self._feature_cols)


class XGBSignalStrategy(_MLSignalBase):
    """XGBoost gradient-boosted trees. 300 rounds, eta=0.05, max_depth=6, early stopping on val."""

    def __init__(
        self,
        *args,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        **kwargs,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        super().__init__(*args, **kwargs)

    def fit(self) -> None:
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        eval_set = [(self._X_val, self._y_val)] if len(self._X_val) > 0 else None
        fit_kwargs = {"eval_set": eval_set, "verbose": False} if eval_set else {}
        self.model.fit(self._X_train, self._y_train, **fit_kwargs)
