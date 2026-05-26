from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aiam.research_agents.ml_ensemble_msr_artifacts import (
    EXPECTED_ARTIFACT_FILES,
    EXPECTED_HANDOFF_FIGURES,
    generate_handoff_figures,
    load_metrics,
    load_report_markdown,
    load_run_manifest,
    render_research_handoff,
    summarize_artifact_inventory,
    summarize_predictions_artifact,
    summarize_strategy_returns_artifact,
    summarize_weights_artifact,
    validate_ml_ensemble_msr_artifact_contract,
    validate_research_handoff,
)


def _write_synthetic_artifacts(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2024-01-01", periods=3)
    assets = ["AAPL.US", "MSFT.US"]
    index = pd.MultiIndex.from_product([dates[:2], assets], names=["Date", "Asset"])
    predictions = pd.DataFrame({"ensemble_pred": [0.01, 0.02, 0.03, np.nan]}, index=index)
    weights = pd.DataFrame(
        {"AAPL.US": [0.6, 0.4, 0.5], "MSFT.US": [0.4, 0.6, 0.5]},
        index=dates,
    )
    weights.index.name = "Date"
    returns = pd.DataFrame({"return": [0.01, -0.002, 0.004]}, index=dates)
    returns.index.name = "Date"
    metrics = {
        "metrics": {
            "annual_return": 0.12,
            "annual_return_arithmetic": 0.12,
            "cagr": 0.1,
            "annual_volatility": 0.2,
            "sharpe": 0.6,
            "max_drawdown": -0.03,
            "total_return": 0.04,
            "observations": 3.0,
        },
        "turnover_diagnostics": {
            "average_turnover": 0.1,
            "median_turnover": 0.1,
            "max_turnover": 0.2,
            "observations": 2.0,
        },
        "concentration_diagnostics": {
            "average_herfindahl": 0.52,
            "average_effective_positions": 1.93,
            "average_max_weight": 0.57,
            "max_single_asset_weight": 0.6,
            "average_top_5_weight_share": 1.0,
            "max_top_5_weight_share": 1.0,
            "observations": 3.0,
        },
    }
    manifest = {
        "strategy": "MSR(Ensemble_mu_hat)",
        "created_at_utc": "1970-01-01T00:00:00Z",
        "universe_size": 2,
        "date_range": ["2024-01-01", "2024-01-03"],
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "feature_count": 17,
        "feature_columns": ["mom_21", "vol_60"],
        "model_components": ["Lasso", "Random Forest", "XGBoost"],
        "cov_lookback": 20,
        "horizon": 21,
        "validation_share": 0.15,
        "artifact_files": EXPECTED_ARTIFACT_FILES,
        "metrics": metrics["metrics"],
        "turnover_diagnostics": metrics["turnover_diagnostics"],
        "concentration_diagnostics": metrics["concentration_diagnostics"],
        "caveats": [
            "historical backtest only",
            "no investment advice",
            "human review required",
        ],
        "reproducibility_notes": ["local cache only", "no notebook execution"],
    }

    predictions.to_parquet(tmp_path / "predictions.parquet")
    weights.to_parquet(tmp_path / "weights.parquet")
    returns.to_parquet(tmp_path / "strategy_returns.parquet")
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "report.md").write_text("# Synthetic Report\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_contract_validation_passes_when_all_files_exist(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)

    result = validate_ml_ensemble_msr_artifact_contract(artifact_dir)

    assert result["ok"] is True
    assert result["missing_files"] == []
    assert result["present_files"] == EXPECTED_ARTIFACT_FILES


def test_contract_validation_fails_with_missing_files(tmp_path):
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

    result = validate_ml_ensemble_msr_artifact_contract(tmp_path)

    assert result["ok"] is False
    assert "predictions.parquet" in result["missing_files"]
    assert result["errors"]


def test_manifest_and_metrics_loaders_read_expected_json(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)

    manifest = load_run_manifest(artifact_dir)
    metrics = load_metrics(artifact_dir)
    report = load_report_markdown(artifact_dir)

    assert manifest["strategy"] == "MSR(Ensemble_mu_hat)"
    assert metrics["metrics"]["sharpe"] == 0.6
    assert "Synthetic Report" in report


def test_artifact_inventory_reports_sizes_and_missing_files(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    (artifact_dir / "report.md").unlink()

    inventory = summarize_artifact_inventory(artifact_dir)

    assert inventory["ok"] is False
    assert inventory["artifact_count"] == 5
    assert "report.md" in inventory["missing_files"]
    assert any(item["size_bytes"] for item in inventory["artifacts"] if item["exists"])


def test_parquet_summaries_are_bounded_and_json_serializable(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)

    predictions = summarize_predictions_artifact(artifact_dir)
    weights = summarize_weights_artifact(artifact_dir)
    returns = summarize_strategy_returns_artifact(artifact_dir)

    json.dumps({"predictions": predictions, "weights": weights, "returns": returns})
    assert predictions["shape"] == [4, 1]
    assert predictions["asset_count"] == 2
    assert weights["asset_count"] == 2
    assert weights["row_sum_min"] == 1.0
    assert returns["observations"] == 3
    assert "return" not in returns


def test_generate_handoff_figures_creates_expected_pngs(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)

    result = generate_handoff_figures(artifact_dir)

    assert result["ok"] is True
    assert Path(result["figure_dir"]).is_dir()
    assert [item["file"] for item in result["figures"]] == [
        f"figures/{name}" for name in EXPECTED_HANDOFF_FIGURES
    ]
    for name in EXPECTED_HANDOFF_FIGURES:
        path = artifact_dir / "figures" / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generate_handoff_figures_metadata_is_json_serializable(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)

    result = generate_handoff_figures(artifact_dir)

    json.dumps(result)
    assert all(item["exists"] for item in result["figures"])


def test_handoff_markdown_contains_required_sections(tmp_path):
    markdown = render_research_handoff(_write_synthetic_artifacts(tmp_path))

    for section in [
        "# ML Ensemble MSR Research Handoff",
        "## Executive Summary",
        "## Deterministic Run Inventory",
        "## Strategy Mechanics",
        "## Performance Metrics",
        "## Turnover and Concentration",
        "## Artifact Inventory",
        "## Methodology Caveats",
        "## Risk Review",
        "## Open Questions",
        "## Human Review Checklist",
    ]:
        assert section in markdown
    assert "research-only" in markdown
    assert "historical weights are not target allocations" in markdown


def test_handoff_markdown_contains_figure_links_when_figures_exist(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    generate_handoff_figures(artifact_dir)

    markdown = render_research_handoff(artifact_dir)

    assert "## Figures" in markdown
    assert "![Cumulative Returns](figures/cumulative_returns.png)" in markdown
    assert "![Drawdown](figures/drawdown.png)" in markdown
    assert "![Turnover](figures/turnover.png)" in markdown
    assert "![Concentration](figures/concentration.png)" in markdown
    assert "![Top Weights](figures/top_weights.png)" in markdown


def test_handoff_markdown_contains_expected_tables(tmp_path):
    markdown = render_research_handoff(_write_synthetic_artifacts(tmp_path))

    assert "| Metric | Value |" in markdown
    assert "Turnover diagnostics:" in markdown
    assert "Concentration diagnostics:" in markdown
    assert "| File | Exists | Size Bytes |" in markdown
    assert markdown.count("| --- | ---: |") >= 3


def test_handoff_validator_rejects_missing_required_sections():
    result = validate_research_handoff("# ML Ensemble MSR Research Handoff\n")

    assert result["ok"] is False
    assert "## Executive Summary" in result["missing_sections"]
    assert result["errors"]


def test_cli_default_mode_works_on_synthetic_artifact_directory(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_ml_ensemble_msr_handoff.py", "--output-dir", str(artifact_dir)],
    )

    runpy.run_path("scripts/build_ml_ensemble_msr_handoff.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["contract_validation"]["ok"] is True
    assert payload["handoff_validation"]["ok"] is True
    assert payload["artifact_count"] == 6


def test_cli_write_figures_print_summary_json_reports_metadata(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ml_ensemble_msr_handoff.py",
            "--output-dir",
            str(artifact_dir),
            "--write-figures",
            "--print-summary-json",
        ],
    )

    runpy.run_path("scripts/build_ml_ensemble_msr_handoff.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["figure_generation"]["ok"] is True
    assert len(payload["figure_generation"]["figures"]) == 5
    assert all(item["exists"] for item in payload["figure_generation"]["figures"])


def test_cli_default_mode_reports_not_ok_for_missing_artifact_directory(tmp_path, monkeypatch, capsys):
    missing_dir = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_ml_ensemble_msr_handoff.py", "--output-dir", str(missing_dir)],
    )

    runpy.run_path("scripts/build_ml_ensemble_msr_handoff.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["contract_validation"]["ok"] is False
    assert payload["handoff_validation"]["ok"] is False
    assert payload["handoff_validation"]["warnings"] == [
        "not evaluated because artifact contract failed"
    ]
    assert payload["artifact_count"] == 0


def test_cli_print_markdown_works_on_synthetic_artifact_directory(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ml_ensemble_msr_handoff.py",
            "--output-dir",
            str(artifact_dir),
            "--print-markdown",
        ],
    )

    runpy.run_path("scripts/build_ml_ensemble_msr_handoff.py", run_name="__main__")

    output = capsys.readouterr().out
    assert output.startswith("# ML Ensemble MSR Research Handoff")
    assert "## Artifact Inventory" in output


def test_cli_write_figures_print_markdown_includes_figure_links(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ml_ensemble_msr_handoff.py",
            "--output-dir",
            str(artifact_dir),
            "--write-figures",
            "--print-markdown",
        ],
    )

    runpy.run_path("scripts/build_ml_ensemble_msr_handoff.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "## Figures" in output
    assert "![Cumulative Returns](figures/cumulative_returns.png)" in output
    assert "![Top Weights](figures/top_weights.png)" in output


def test_source_check_for_disallowed_terms():
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
        ".e" + "nv",
    ]
    paths = [
        Path("src/aiam/research_agents/ml_ensemble_msr_artifacts.py"),
        Path("scripts/build_ml_ensemble_msr_handoff.py"),
        Path("tests/research_agents/test_ml_ensemble_msr_artifacts.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not [term for term in terms if term in text]
