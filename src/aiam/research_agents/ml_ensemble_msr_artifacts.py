"""Deterministic inspection helpers for ML Ensemble MSR research artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ARTIFACT_FILES = [
    "predictions.parquet",
    "weights.parquet",
    "strategy_returns.parquet",
    "metrics.json",
    "report.md",
    "run_manifest.json",
]

REQUIRED_HANDOFF_SECTIONS = [
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
]

REQUIRED_HANDOFF_PHRASES = [
    "research-only",
    "no investment advice",
    "human review required",
    "historical backtest only",
    "historical weights are not target allocations",
]


def validate_ml_ensemble_msr_artifact_contract(output_dir: str | Path) -> dict[str, Any]:
    """Validate the expected artifact file set without reading artifact content."""
    artifact_dir = Path(output_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if artifact_dir.exists() and not artifact_dir.is_dir():
        errors.append("output_dir must be an artifact directory, not an individual file")

    if not artifact_dir.exists():
        errors.append("output_dir does not exist")

    present_files = [
        name for name in EXPECTED_ARTIFACT_FILES if (artifact_dir / name).is_file()
    ]
    missing_files = [name for name in EXPECTED_ARTIFACT_FILES if name not in present_files]

    if missing_files:
        errors.append("missing expected artifact files")

    extra_files: list[str] = []
    if artifact_dir.is_dir():
        extra_files = sorted(
            path.name for path in artifact_dir.iterdir() if path.is_file() and path.name not in EXPECTED_ARTIFACT_FILES
        )
    if extra_files:
        warnings.append(f"unexpected files present: {', '.join(extra_files)}")

    return {
        "ok": not errors,
        "output_dir": str(artifact_dir),
        "missing_files": missing_files,
        "present_files": present_files,
        "errors": errors,
        "warnings": warnings,
    }


def load_run_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Load the deterministic run manifest from the expected artifact name."""
    return _load_json_expected(output_dir, "run_manifest.json")


def load_metrics(output_dir: str | Path) -> dict[str, Any]:
    """Load metrics from the expected artifact name."""
    return _load_json_expected(output_dir, "metrics.json")


def load_report_markdown(output_dir: str | Path) -> str:
    """Load the markdown report from the expected artifact name."""
    return _expected_path(output_dir, "report.md").read_text(encoding="utf-8")


def summarize_artifact_inventory(output_dir: str | Path) -> dict[str, Any]:
    """Return a bounded inventory of expected artifacts and file sizes."""
    artifact_dir = Path(output_dir)
    artifacts = []
    missing_files = []
    for name in EXPECTED_ARTIFACT_FILES:
        path = artifact_dir / name
        exists = path.is_file()
        if not exists:
            missing_files.append(name)
        artifacts.append(
            {
                "file": name,
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists else None,
            }
        )
    return {
        "ok": not missing_files and artifact_dir.is_dir(),
        "artifact_count": sum(1 for artifact in artifacts if artifact["exists"]),
        "artifacts": artifacts,
        "missing_files": missing_files,
    }


def summarize_predictions_artifact(output_dir: str | Path) -> dict[str, Any]:
    """Summarize the predictions parquet without returning raw predictions."""
    frame = pd.read_parquet(_expected_path(output_dir, "predictions.parquet"))
    numeric = frame.select_dtypes(include=[np.number])
    index = frame.index
    dates = _index_dates(index)
    assets = _index_assets(index)
    values = numeric.to_numpy(dtype=float) if not numeric.empty else np.array([], dtype=float)
    return _json_ready(
        {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "index_names": [None if name is None else str(name) for name in index.names],
            "column_names": [str(column) for column in frame.columns],
            "date_min": _date_min(dates),
            "date_max": _date_max(dates),
            "asset_count": int(len(pd.Index(assets).unique())) if assets is not None else None,
            "null_count": int(frame.isna().sum().sum()),
            "finite_ratio": _finite_ratio(values),
        }
    )


def summarize_weights_artifact(output_dir: str | Path) -> dict[str, Any]:
    """Summarize the weights parquet without returning raw weights."""
    frame = pd.read_parquet(_expected_path(output_dir, "weights.parquet"))
    numeric = frame.select_dtypes(include=[np.number])
    dates = _index_dates(frame.index)
    row_sums = numeric.sum(axis=1) if not numeric.empty else pd.Series(dtype=float)
    nonzero_assets = (numeric.abs() > 0.0).sum(axis=1) if not numeric.empty else pd.Series(dtype=float)
    values = numeric.to_numpy(dtype=float) if not numeric.empty else np.array([], dtype=float)
    return _json_ready(
        {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "date_min": _date_min(dates),
            "date_max": _date_max(dates),
            "asset_count": int(frame.shape[1]),
            "row_sum_min": _safe_float(row_sums.min()) if not row_sums.empty else None,
            "row_sum_max": _safe_float(row_sums.max()) if not row_sums.empty else None,
            "min_weight": _safe_float(numeric.min().min()) if not numeric.empty else None,
            "max_weight": _safe_float(numeric.max().max()) if not numeric.empty else None,
            "average_nonzero_assets": _safe_float(nonzero_assets.mean()) if not nonzero_assets.empty else None,
            "finite_ratio": _finite_ratio(values),
        }
    )


def summarize_strategy_returns_artifact(output_dir: str | Path) -> dict[str, Any]:
    """Summarize the strategy returns parquet without returning raw returns."""
    frame = pd.read_parquet(_expected_path(output_dir, "strategy_returns.parquet"))
    if isinstance(frame, pd.Series):
        series = frame
    elif "return" in frame.columns:
        series = frame["return"]
    else:
        series = frame.select_dtypes(include=[np.number]).iloc[:, 0]

    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    dates = _index_dates(series.index)
    values = series.to_numpy(dtype=float) if len(series) else np.array([], dtype=float)
    return _json_ready(
        {
            "observations": int(len(series)),
            "date_min": _date_min(dates),
            "date_max": _date_max(dates),
            "min_return": _safe_float(clean.min()) if not clean.empty else None,
            "max_return": _safe_float(clean.max()) if not clean.empty else None,
            "mean_return": _safe_float(clean.mean()) if not clean.empty else None,
            "std_return": _safe_float(clean.std()) if len(clean) > 1 else None,
            "total_return": _safe_float((1.0 + clean).prod() - 1.0) if not clean.empty else None,
            "finite_ratio": _finite_ratio(values),
        }
    )


def render_research_handoff(output_dir: str | Path) -> str:
    """Render a compact GitHub-compatible markdown handoff memo."""
    contract = validate_ml_ensemble_msr_artifact_contract(output_dir)
    inventory = summarize_artifact_inventory(output_dir)
    manifest = load_run_manifest(output_dir) if (Path(output_dir) / "run_manifest.json").is_file() else {}
    metrics_payload = load_metrics(output_dir) if (Path(output_dir) / "metrics.json").is_file() else {}
    metrics = _section_dict(metrics_payload, "metrics")
    turnover = _section_dict(metrics_payload, "turnover_diagnostics")
    concentration = _section_dict(metrics_payload, "concentration_diagnostics")
    prediction_summary = _try_summary(summarize_predictions_artifact, output_dir)
    weights_summary = _try_summary(summarize_weights_artifact, output_dir)
    returns_summary = _try_summary(summarize_strategy_returns_artifact, output_dir)

    caveats = _as_list(manifest.get("caveats")) or [
        "historical backtest only",
        "no investment advice",
        "human review required",
    ]
    notes = _as_list(manifest.get("reproducibility_notes"))

    return (
        "# ML Ensemble MSR Research Handoff\n\n"
        "## Executive Summary\n\n"
        "This memo is research-only, no investment advice, and human review required. "
        "It summarizes a deterministic local artifact set for MSR(Ensemble_mu_hat); "
        "results are historical backtest only and historical weights are not target allocations.\n\n"
        "## Deterministic Run Inventory\n\n"
        f"- Artifact directory: `{contract['output_dir']}`\n"
        f"- Contract valid: `{contract['ok']}`\n"
        f"- Strategy: {_display(manifest.get('strategy', 'MSR(Ensemble_mu_hat)'))}\n"
        f"- Universe size: {_display(manifest.get('universe_size'))}\n"
        f"- Date range: {_display_range(manifest.get('date_range'))}\n"
        f"- Train end: {_display(manifest.get('train_end'))}\n"
        f"- Test start: {_display(manifest.get('test_start'))}\n"
        f"- Feature count: {_display(manifest.get('feature_count'))}\n"
        f"- Model components: {_display_join(manifest.get('model_components'))}\n\n"
        "## Strategy Mechanics\n\n"
        "- Equal-weighted ensemble of Lasso, Random Forest, and XGBoost expected-return forecasts.\n"
        "- Long-only maximum Sharpe ratio approximation from local cached inputs.\n"
        "- Portfolio weights are lagged by one trading day before return realization.\n"
        "- Baseline diagnostics do not include transaction costs or allocation constraints.\n\n"
        "## Performance Metrics\n\n"
        f"{_metrics_table(metrics)}\n\n"
        "## Turnover and Concentration\n\n"
        "Turnover diagnostics:\n\n"
        f"{_metrics_table(turnover)}\n\n"
        "Concentration diagnostics:\n\n"
        f"{_metrics_table(concentration)}\n\n"
        "Weights artifact summary:\n\n"
        f"{_compact_table(['Metric', 'Value'], _summary_rows(weights_summary))}\n\n"
        "## Artifact Inventory\n\n"
        f"{_artifact_table(inventory['artifacts'])}\n\n"
        "Additional bounded artifact summaries:\n\n"
        f"{_compact_table(['Artifact', 'Key', 'Value'], _combined_summary_rows(prediction_summary, returns_summary))}\n\n"
        "## Methodology Caveats\n\n"
        f"{_bullet_list(caveats)}\n\n"
        "## Risk Review\n\n"
        "- Historical backtest only; future performance can differ materially.\n"
        "- Historical weights are not target allocations and require human review before any use.\n"
        "- Optimizer concentration, turnover, transaction costs, and cache lineage should be reviewed.\n\n"
        "## Open Questions\n\n"
        "- Does the result reconcile with the Notebook 03 benchmark and published MSR diagnostics?\n"
        "- What transaction-cost and concentration limits should be applied before further research?\n"
        "- Are cache vintage, universe membership, and benchmark assumptions approved for review?\n\n"
        "## Human Review Checklist\n\n"
        "- Confirm artifact contract validity and source cache lineage.\n"
        "- Review performance, turnover, concentration, and methodology caveats.\n"
        "- Confirm no investment advice is inferred from this research-only handoff.\n"
        f"{_optional_notes(notes)}"
    )


def validate_research_handoff(markdown: str) -> dict[str, Any]:
    """Validate required handoff sections and governance phrases."""
    missing_sections = [section for section in REQUIRED_HANDOFF_SECTIONS if section not in markdown]
    lower_text = markdown.lower()
    missing_phrases = [phrase for phrase in REQUIRED_HANDOFF_PHRASES if phrase not in lower_text]
    errors = []
    if missing_sections:
        errors.append("missing required sections")
    if missing_phrases:
        errors.append("missing required governance phrases")
    return {
        "ok": not errors,
        "missing_sections": missing_sections,
        "errors": errors,
        "warnings": [f"missing phrase: {phrase}" for phrase in missing_phrases],
    }


def headline_metrics(output_dir: str | Path) -> dict[str, Any]:
    """Return a compact metric subset for CLI summaries."""
    payload = load_metrics(output_dir)
    metrics = _section_dict(payload, "metrics")
    return {
        key: _json_ready(metrics.get(key))
        for key in ["annual_return_arithmetic", "cagr", "annual_volatility", "sharpe", "max_drawdown", "total_return"]
        if key in metrics
    }


def _expected_path(output_dir: str | Path, file_name: str) -> Path:
    if file_name not in EXPECTED_ARTIFACT_FILES:
        raise ValueError(f"unexpected artifact name: {file_name}")
    artifact_dir = Path(output_dir)
    if not artifact_dir.is_dir():
        raise ValueError("output_dir must be an artifact directory")
    return artifact_dir / file_name


def _load_json_expected(output_dir: str | Path, file_name: str) -> dict[str, Any]:
    with _expected_path(output_dir, file_name).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{file_name} must contain a JSON object")
    return payload


def _index_dates(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        level_name = "Date" if "Date" in index.names else index.names[0]
        return pd.to_datetime(index.get_level_values(level_name), errors="coerce")
    return pd.to_datetime(index, errors="coerce")


def _index_assets(index: pd.Index) -> pd.Index | None:
    if not isinstance(index, pd.MultiIndex):
        return None
    level_name = "Asset" if "Asset" in index.names else index.names[-1]
    return index.get_level_values(level_name)


def _date_min(dates: pd.Index) -> str | None:
    clean = pd.Series(dates).dropna()
    return None if clean.empty else str(clean.min().date())


def _date_max(dates: pd.Index) -> str | None:
    clean = pd.Series(dates).dropna()
    return None if clean.empty else str(clean.max().date())


def _finite_ratio(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return _safe_float(np.isfinite(values).sum() / values.size)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _section_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _try_summary(function: Any, output_dir: str | Path) -> dict[str, Any]:
    try:
        result = function(output_dir)
    except (FileNotFoundError, ValueError, OSError, KeyError, IndexError) as exc:
        return {"error": str(exc)}
    return result


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _display(value: Any) -> str:
    if value is None:
        return "not available"
    return str(value)


def _display_range(value: Any) -> str:
    if isinstance(value, list | tuple) and len(value) == 2:
        return f"{value[0]} to {value[1]}"
    return _display(value)


def _display_join(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return _display(value)


def _format_value(value: Any) -> str:
    value = _json_ready(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _metrics_table(values: dict[str, Any]) -> str:
    rows = [(str(key), _format_value(value)) for key, value in values.items()]
    return _compact_table(["Metric", "Value"], rows)


def _artifact_table(artifacts: list[dict[str, Any]]) -> str:
    rows = [
        (artifact["file"], str(artifact["exists"]), _format_value(artifact["size_bytes"]))
        for artifact in artifacts
    ]
    return _compact_table(["File", "Exists", "Size Bytes"], rows)


def _compact_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    if rows:
        lines.extend("| " + " | ".join(_format_value(cell) for cell in row) + " |" for row in rows)
    else:
        lines.append("| n/a | " + " | ".join("n/a" for _ in headers[1:]) + " |")
    return "\n".join(lines)


def _summary_rows(summary: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(key, value) for key, value in summary.items()]


def _combined_summary_rows(
    prediction_summary: dict[str, Any],
    returns_summary: dict[str, Any],
) -> list[tuple[Any, ...]]:
    rows = []
    for artifact, summary in [
        ("predictions.parquet", prediction_summary),
        ("strategy_returns.parquet", returns_summary),
    ]:
        for key, value in summary.items():
            rows.append((artifact, key, value))
    return rows


def _bullet_list(items: list[Any]) -> str:
    if not items:
        return "- n/a"
    return "\n".join(f"- {item}" for item in items)


def _optional_notes(notes: list[Any]) -> str:
    if not notes:
        return ""
    return "\n\nReproducibility notes:\n\n" + _bullet_list(notes) + "\n"
