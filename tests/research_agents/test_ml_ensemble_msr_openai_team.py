from __future__ import annotations

import importlib
import inspect
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiam.research_agents.ml_ensemble_msr_artifacts import EXPECTED_ARTIFACT_FILES


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


def test_module_imports_without_sdk_installed(monkeypatch):
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "agents":
            raise ImportError("missing optional sdk")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    module = importlib.reload(
        importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")
    )

    assert module.OPTIONAL_SDK_ERROR.startswith("Optional OpenAI Agents SDK")


def test_deterministic_wrappers_return_valid_json_and_markdown(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    module = importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")

    contract = json.loads(module.validate_artifact_contract_json(artifact_dir))
    summary = json.loads(module.build_handoff_summary_json(artifact_dir))
    markdown = module.build_handoff_markdown(artifact_dir)

    assert contract["ok"] is True
    assert summary["ok"] is True
    assert summary["bounded_artifact_summaries"]["weights"]["asset_count"] == 2
    assert markdown.startswith("# ML Ensemble MSR Research Handoff")


def test_cli_default_mode_works_without_sdk_and_prints_ok(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ml_ensemble_msr_openai_team.py", "--output-dir", str(artifact_dir)],
    )

    runpy.run_path("scripts/run_ml_ensemble_msr_openai_team.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_print_markdown_works_without_sdk(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ml_ensemble_msr_openai_team.py",
            "--output-dir",
            str(artifact_dir),
            "--print-markdown",
        ],
    )

    runpy.run_path("scripts/run_ml_ensemble_msr_openai_team.py", run_name="__main__")

    output = capsys.readouterr().out
    assert output.startswith("# ML Ensemble MSR Research Handoff")
    assert "## Human Review Checklist" in output


def test_cli_write_handoff_without_run_team_writes_deterministic_handoff(
    tmp_path,
    monkeypatch,
    capsys,
):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ml_ensemble_msr_openai_team.py",
            "--output-dir",
            str(artifact_dir),
            "--write-handoff",
        ],
    )

    runpy.run_path("scripts/run_ml_ensemble_msr_openai_team.py", run_name="__main__")

    payload = json.loads(capsys.readouterr().out)
    handoff_path = artifact_dir / "research_team_handoff.md"
    assert payload["ok"] is True
    assert handoff_path.is_file()
    assert handoff_path.read_text(encoding="utf-8").startswith(
        "# ML Ensemble MSR Research Handoff"
    )


def test_live_functions_raise_clear_error_if_sdk_missing(tmp_path, monkeypatch):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    module = importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")
    monkeypatch.setattr(
        module,
        "_load_agents_sdk",
        lambda: (_ for _ in ()).throw(RuntimeError(module.OPTIONAL_SDK_ERROR)),
    )

    with pytest.raises(RuntimeError, match="Optional OpenAI Agents SDK"):
        module.build_ml_ensemble_msr_openai_team("gpt-test")
    with pytest.raises(RuntimeError, match="Optional OpenAI Agents SDK"):
        module.run_ml_ensemble_msr_openai_team(artifact_dir, "gpt-test")


def test_model_id_is_required_for_live_run(tmp_path):
    artifact_dir = _write_synthetic_artifacts(tmp_path)
    module = importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")

    with pytest.raises(ValueError, match="model_id is required"):
        module.build_ml_ensemble_msr_openai_team("")
    with pytest.raises(ValueError, match="model_id is required"):
        module.run_ml_ensemble_msr_openai_team(artifact_dir, "")


def test_agent_instruction_strings_contain_required_constraints():
    module = importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")
    instruction_names = [
        "RESEARCH_MANAGER_INSTRUCTIONS",
        "DATA_QA_INSTRUCTIONS",
        "QUANT_STRATEGY_INSTRUCTIONS",
        "PORTFOLIO_RISK_INSTRUCTIONS",
        "RESEARCH_HANDOFF_INSTRUCTIONS",
    ]

    for name in instruction_names:
        text = getattr(module, name).lower()
        assert "research-only" in text
        assert "no investment advice" in text
        assert "no target allocations" in text
        assert "no trading recommendations" in text
        assert "no live api calls" in text
        assert "no arbitrary file reads" in text
        assert "use only deterministic tool outputs" in text
        assert "historical weights are not target allocations" in text
        assert "human review required" in text


def test_tool_functions_expose_no_arbitrary_file_reader_interface():
    module = importlib.import_module("aiam.research_agents.ml_ensemble_msr_openai_team")
    tool_names = [
        "validate_artifact_contract",
        "load_handoff_summary",
        "render_handoff_markdown",
        "generate_figures",
    ]

    for name in tool_names:
        signature = inspect.signature(getattr(module, name))
        assert "output_dir" in signature.parameters
        assert "file_path" not in signature.parameters
        assert "path" not in signature.parameters
        assert "glob" not in signature.parameters


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
        Path("src/aiam/research_agents/ml_ensemble_msr_openai_team.py"),
        Path("scripts/run_ml_ensemble_msr_openai_team.py"),
        Path("tests/research_agents/test_ml_ensemble_msr_openai_team.py"),
        Path("pyproject.toml"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not [term for term in terms if term in text]

