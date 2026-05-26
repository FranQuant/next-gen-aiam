from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aiam.strategy.ml_ensemble_msr import (
    FEATURE_COLUMNS,
    MLEnsembleMSRConfig,
    backtest_lagged_weights,
    build_ensemble_predictions,
    build_ml_feature_panel_from_ohlcv,
    build_target_21d,
    compute_msr_weights_from_mu,
    compute_performance_metrics,
)


ASSETS = ["AAPL.US", "MSFT.US", "SPY.US"]


def _returns(n: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    values = np.full((n, len(ASSETS)), 0.001)
    values[:, 1] = 0.0005
    values[:, 2] = -0.0002
    return pd.DataFrame(values, index=dates, columns=ASSETS)


def _ohlcv_dict(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
    open_ = close.shift(1).fillna(close.iloc[0]) * 1.0001
    high = pd.concat([open_, close]).groupby(level=0).max() * 1.002
    low = pd.concat([open_, close]).groupby(level=0).min() * 0.998
    volume = pd.DataFrame(1_000_000.0, index=returns.index, columns=returns.columns)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _prediction_series(values: list[list[float]], dates: pd.DatetimeIndex) -> pd.Series:
    idx = pd.MultiIndex.from_product([dates, ASSETS[: len(values[0])]], names=["Date", "Asset"])
    return pd.Series(np.array(values).ravel(), index=idx)


def test_build_target_21d_aligns_future_returns():
    dates = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame({"AAPL.US": [1, 2, 3, 4, 5, 6]}, index=dates, dtype=float)

    target = build_target_21d(returns, horizon=2)

    assert target.loc[(dates[1], "AAPL.US")] == 7.0
    assert target.loc[(dates[3], "AAPL.US")] == 11.0


def test_feature_panel_has_expected_feature_columns():
    returns = _returns()
    features = build_ml_feature_panel_from_ohlcv(returns, _ohlcv_dict(returns))

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.names == ["Date", "Asset"]
    assert set(features.index.get_level_values("Asset")) == set(ASSETS)


def test_ensemble_prediction_average_is_exact():
    dates = pd.bdate_range("2024-01-01", periods=2)
    lasso = _prediction_series([[1.0, 2.0], [3.0, 4.0]], dates)
    rf = _prediction_series([[2.0, 3.0], [4.0, 5.0]], dates)
    xgb = _prediction_series([[3.0, 4.0], [5.0, 6.0]], dates)

    ensemble = build_ensemble_predictions({"lasso": lasso, "rf": rf, "xgb": xgb})

    expected = _prediction_series([[2.0, 3.0], [4.0, 5.0]], dates)
    pd.testing.assert_series_equal(ensemble, expected.rename("ensemble_pred"))


def test_msr_weights_are_long_only_and_normalized():
    returns = _returns(n=30).iloc[:, :2]
    dates = returns.index[-2:]
    preds = _prediction_series([[0.02, 0.01], [0.03, 0.01]], dates)
    config = MLEnsembleMSRConfig(cov_lookback=20, min_assets=2)

    weights = compute_msr_weights_from_mu(preds, returns, config)

    assert not weights.empty
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights >= 0.0).all().all()
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0)
    assert list(weights.columns) == list(returns.columns)


def test_msr_fallback_produces_equal_weights_when_clipped_to_zero():
    returns = _returns(n=30).iloc[:, :2]
    dates = returns.index[-1:]
    preds = _prediction_series([[0.0, 0.0]], dates)
    config = MLEnsembleMSRConfig(cov_lookback=20, min_assets=2)

    weights = compute_msr_weights_from_mu(preds, returns, config)

    np.testing.assert_allclose(weights.loc[dates[0], returns.columns].to_numpy(), [0.5, 0.5])


def test_lagged_backtest_uses_next_return_date():
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.DataFrame({"AAPL.US": [0.0, 0.10, 0.30], "MSFT.US": [0.0, 0.20, 0.40]}, index=dates)
    weights = pd.DataFrame({"AAPL.US": [1.0], "MSFT.US": [0.0]}, index=[dates[0]])

    realized = backtest_lagged_weights(weights, returns)

    assert list(realized.index) == [dates[1]]
    assert realized.iloc[0] == 0.10


def test_metrics_contain_required_keys_and_finite_values():
    returns = pd.Series([0.01, -0.005, 0.002], index=pd.bdate_range("2024-01-01", periods=3))

    metrics = compute_performance_metrics(returns)

    assert set(metrics) == {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "total_return",
        "observations",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_cli_print_summary_json(monkeypatch, capsys):
    import aiam.strategy.ml_ensemble_msr as module

    def fake_run_ml_ensemble_msr_research(**kwargs):
        return {
            "universe_size": 3,
            "date_range": ("2024-01-01", "2024-01-03"),
            "config": kwargs["config"],
            "metrics": {
                "annual_return": 0.1,
                "annual_volatility": 0.2,
                "sharpe": 0.5,
                "max_drawdown": -0.01,
                "total_return": 0.02,
                "observations": 3.0,
            },
        }

    monkeypatch.setattr(module, "run_ml_ensemble_msr_research", fake_run_ml_ensemble_msr_research)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ml_ensemble_msr_research.py", "--print-summary-json", "--cov-lookback", "10"],
    )

    runpy.run_path("scripts/run_ml_ensemble_msr_research.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "MSR(Ensemble_mu_hat)"
    assert payload["cov_lookback"] == 10


def test_security_source_check_new_files():
    terms = [
        "dot" + "env",
        "os.get" + "env",
        "os.en" + "viron",
        "api" + "_" + "key",
        "sec" + "ret",
        "pass" + "word",
        "cred" + "ential",
        "re" + "quests",
        "htt" + "px",
        "ur" + "llib",
        "sock" + "et",
        "bro" + "ker",
        "ord" + "er",
        "tra" + "de",
        ".e" + "nv",
    ]
    paths = [
        Path("src/aiam/strategy/ml_ensemble_msr.py"),
        Path("scripts/run_ml_ensemble_msr_research.py"),
    ]
    for path in paths:
        text = path.read_text().lower()
        assert not [term for term in terms if term in text]
