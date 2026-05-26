from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aiam.strategy.ml_ensemble_msr import (
    ARTIFACT_FILES,
    FEATURE_COLUMNS,
    MLEnsembleMSRConfig,
    backtest_lagged_weights,
    build_report,
    build_run_manifest,
    build_ensemble_predictions,
    build_ml_feature_panel_from_ohlcv,
    build_target_21d,
    compute_concentration_diagnostics,
    compute_msr_weights_from_mu,
    compute_performance_metrics,
    compute_turnover_diagnostics,
    summary_payload,
    write_research_artifacts,
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
        "annual_return_arithmetic",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "total_return",
        "observations",
    }
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["annual_return"] == metrics["annual_return_arithmetic"]


def test_metrics_sharpe_uses_arithmetic_annualized_return_not_cagr():
    returns = pd.Series([0.10, -0.05, 0.02, 0.01], index=pd.bdate_range("2024-01-01", periods=4))

    metrics = compute_performance_metrics(returns)

    expected_arithmetic = float(returns.mean() * 252.0)
    expected_volatility = float(returns.std() * np.sqrt(252.0))
    np.testing.assert_allclose(metrics["annual_return_arithmetic"], expected_arithmetic)
    np.testing.assert_allclose(metrics["sharpe"], expected_arithmetic / expected_volatility)
    assert not np.isclose(metrics["sharpe"], metrics["cagr"] / expected_volatility)


def test_metrics_cagr_matches_known_compounded_return():
    returns = pd.Series([0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=2))

    metrics = compute_performance_metrics(returns)

    total_return = (1.01 * 1.02) - 1.0
    expected_cagr = (1.0 + total_return) ** (252.0 / 2.0) - 1.0
    np.testing.assert_allclose(metrics["total_return"], total_return)
    np.testing.assert_allclose(metrics["cagr"], expected_cagr)


def test_turnover_diagnostics_from_simple_weights():
    dates = pd.bdate_range("2024-01-01", periods=3)
    weights = pd.DataFrame(
        {
            "AAPL.US": [0.5, 0.25, 1.0],
            "MSFT.US": [0.5, 0.75, 0.0],
        },
        index=[dates[1], dates[2], dates[0]],
    )

    diagnostics = compute_turnover_diagnostics(weights)

    np.testing.assert_allclose(diagnostics["average_turnover"], 0.375)
    np.testing.assert_allclose(diagnostics["median_turnover"], 0.375)
    np.testing.assert_allclose(diagnostics["max_turnover"], 0.5)
    assert diagnostics["observations"] == 2.0


def test_turnover_diagnostics_require_two_rows():
    weights = pd.DataFrame({"AAPL.US": [1.0]}, index=[pd.Timestamp("2024-01-01")])

    diagnostics = compute_turnover_diagnostics(weights)

    assert np.isnan(diagnostics["average_turnover"])
    assert diagnostics["observations"] == 0.0


def test_concentration_diagnostics_equal_weight_and_concentrated_weights():
    weights = pd.DataFrame(
        {
            "AAPL.US": [0.5, 1.0],
            "MSFT.US": [0.5, 0.0],
            "SPY.US": [0.0, 0.0],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    diagnostics = compute_concentration_diagnostics(weights)

    np.testing.assert_allclose(diagnostics["average_herfindahl"], 0.75)
    np.testing.assert_allclose(diagnostics["average_effective_positions"], 1.5)
    np.testing.assert_allclose(diagnostics["average_max_weight"], 0.75)
    np.testing.assert_allclose(diagnostics["max_single_asset_weight"], 1.0)
    np.testing.assert_allclose(diagnostics["average_top_5_weight_share"], 1.0)
    np.testing.assert_allclose(diagnostics["max_top_5_weight_share"], 1.0)
    assert diagnostics["observations"] == 2.0


def test_concentration_diagnostics_empty_weights():
    diagnostics = compute_concentration_diagnostics(pd.DataFrame())

    assert np.isnan(diagnostics["average_herfindahl"])
    assert diagnostics["observations"] == 0.0


def _minimal_result() -> dict[str, object]:
    dates = pd.bdate_range("2024-01-01", periods=3)
    predictions = _prediction_series([[0.01, 0.02, 0.03]], dates[:1]).rename("ensemble_pred")
    weights = pd.DataFrame(
        {"AAPL.US": [0.5], "MSFT.US": [0.3], "SPY.US": [0.2]},
        index=dates[:1],
    )
    strategy_returns = pd.Series([0.01, -0.002], index=dates[1:], name="MSR(Ensemble_mu_hat)")
    return {
        "universe_size": 3,
        "date_range": ("2024-01-01", "2024-01-03"),
        "config": MLEnsembleMSRConfig(cov_lookback=10),
        "metrics": {
            "annual_return": 0.1,
            "annual_return_arithmetic": 0.1,
            "cagr": 0.08,
            "annual_volatility": 0.2,
            "sharpe": 0.5,
            "max_drawdown": -0.01,
            "total_return": 0.02,
            "observations": 3.0,
        },
        "turnover_diagnostics": {
            "average_turnover": 0.1,
            "median_turnover": 0.08,
            "max_turnover": 0.2,
            "observations": 2.0,
        },
        "concentration_diagnostics": {
            "average_herfindahl": 0.4,
            "average_effective_positions": 2.5,
            "average_max_weight": 0.6,
            "max_single_asset_weight": 0.9,
            "average_top_5_weight_share": 1.0,
            "max_top_5_weight_share": 1.0,
            "observations": 3.0,
        },
        "caveats": [
            "CAGR and arithmetic annualized return are both reported",
            "Sharpe uses arithmetic annualized return over annualized volatility",
            "concentration diagnostics are descriptive; no constraint is imposed in this baseline",
        ],
        "predictions": predictions,
        "weights": weights,
        "strategy_returns": strategy_returns,
    }


def test_summary_payload_includes_turnover_and_concentration_sections():
    payload = summary_payload(_minimal_result())

    assert "turnover_diagnostics" in payload
    assert "concentration_diagnostics" in payload
    assert payload["turnover_diagnostics"]["average_turnover"] == 0.1
    assert payload["concentration_diagnostics"]["average_herfindahl"] == 0.4


def test_build_report_includes_institutional_metric_sections():
    report = build_report(_minimal_result())

    assert "## Performance Metrics" in report
    assert "## Turnover Diagnostics" in report
    assert "## Concentration Diagnostics" in report
    assert "CAGR" in report
    assert "arithmetic annualized return" in report


def test_build_run_manifest_contains_required_top_level_keys():
    manifest = build_run_manifest(_minimal_result())

    assert set(manifest) == {
        "strategy",
        "created_at_utc",
        "universe_size",
        "date_range",
        "train_end",
        "test_start",
        "feature_count",
        "feature_columns",
        "model_components",
        "cov_lookback",
        "horizon",
        "validation_share",
        "artifact_files",
        "metrics",
        "turnover_diagnostics",
        "concentration_diagnostics",
        "caveats",
        "reproducibility_notes",
    }


def test_build_run_manifest_contains_strategy_and_artifact_files():
    manifest = build_run_manifest(_minimal_result())

    assert manifest["strategy"] == "MSR(Ensemble_mu_hat)"
    assert manifest["artifact_files"] == ARTIFACT_FILES
    assert set(manifest["artifact_files"]) == {
        "predictions.parquet",
        "weights.parquet",
        "strategy_returns.parquet",
        "metrics.json",
        "report.md",
        "run_manifest.json",
    }


def test_build_run_manifest_contains_metrics_and_diagnostics():
    result = _minimal_result()

    manifest = build_run_manifest(result)

    assert manifest["metrics"] == result["metrics"]
    assert manifest["turnover_diagnostics"] == result["turnover_diagnostics"]
    assert manifest["concentration_diagnostics"] == result["concentration_diagnostics"]


def test_write_research_artifacts_writes_expected_files(tmp_path):
    write_research_artifacts(_minimal_result(), tmp_path)

    written = {path.name for path in tmp_path.iterdir()}
    assert written == set(ARTIFACT_FILES)


def test_written_manifest_and_metrics_json_are_structured(tmp_path):
    write_research_artifacts(_minimal_result(), tmp_path)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert manifest["strategy"] == "MSR(Ensemble_mu_hat)"
    assert manifest["artifact_files"] == ARTIFACT_FILES
    assert manifest["reproducibility_notes"] == [
        "local cache only",
        "no live EODHD call",
        "no notebook execution",
        "single-fit ML setup",
        "no transaction costs in baseline",
        "weights lagged by one trading day",
    ]
    assert set(metrics) == {
        "metrics",
        "turnover_diagnostics",
        "concentration_diagnostics",
    }
    assert metrics["metrics"]["sharpe"] == 0.5
    assert metrics["turnover_diagnostics"]["average_turnover"] == 0.1
    assert metrics["concentration_diagnostics"]["average_herfindahl"] == 0.4


def test_cli_print_summary_json(monkeypatch, capsys):
    import aiam.strategy.ml_ensemble_msr as module

    def fake_run_ml_ensemble_msr_research(**kwargs):
        result = _minimal_result()
        result.update(
            {
                "universe_size": 3,
                "date_range": ("2024-01-01", "2024-01-03"),
                "config": kwargs["config"],
            }
        )
        return result

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
        Path("tests/test_ml_ensemble_msr.py"),
    ]
    for path in paths:
        text = path.read_text().lower()
        assert not [term for term in terms if term in text]
