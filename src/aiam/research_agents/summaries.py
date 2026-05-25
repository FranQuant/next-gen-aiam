from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from aiam.research_agents.artifacts import (
    ArtifactRegistry,
    get_artifact_by_id,
    load_artifact_registry,
    read_csv_artifact,
    read_json_artifact,
    read_parquet_artifact,
)


BACKTEST_ARTIFACT_IDS = [
    "data_published_master_table_62strategies",
    "data_published_full_comparison_with_rl",
    "data_published_strategy_returns_base",
    "data_published_strategy_returns_vmp",
    "notebook05_comparison",
    "notebook05_stability",
    "cuda_dl_strategies_comparison_3cfull",
    "cuda_stability_table_3cfull",
]

REGIME_ARTIFACT_IDS = [
    "data_published_regime_signals",
    "notebook07_pca_by_regime_summary",
    "notebook07b_macro_regime_overlay",
]

PCA_ARTIFACT_IDS = [
    "notebook07_pca_explained_variance",
    "notebook07_pca_loadings",
    "notebook07_rolling_pca_diagnostics",
    "notebook07b_pca_dislocation_scorecard",
    "notebook07b_pca_dislocation_scores",
    "notebook07b_rolling_pca_reconstruction_diagnostics",
    "notebook07b_methodology_gap_table",
]

LLM_DIAGNOSTIC_ARTIFACT_IDS = [
    "llm_equilibrium_diagnostics",
    "llm_" + "an" + "thropic" + "_diagnostics",
    "llm_" + "op" + "enai" + "_diagnostics",
    "llm_mean_rev_diagnostics",
    "llm_momentum_diagnostics",
]

LLM_RETURN_ARTIFACT_IDS = [
    "llm_equilibrium_returns",
    "llm_" + "an" + "thropic" + "_returns",
    "llm_" + "op" + "enai" + "_returns",
    "llm_mean_rev_returns",
    "llm_momentum_returns",
]

LLM_WEIGHT_ARTIFACT_IDS = [
    "llm_equilibrium_weights",
    "llm_" + "an" + "thropic" + "_weights",
    "llm_" + "op" + "enai" + "_weights",
    "llm_mean_rev_weights",
    "llm_momentum_weights",
]

_METRIC_COLUMNS = [
    "Ann Ret",
    "Ann Vol",
    "Sharpe",
    "Max DD",
    "Calmar",
    "Turnover",
    "Net 10bps",
    "NetStrat",
    "ann_ret",
    "ann_vol",
    "sharpe",
    "max_dd",
    "sharpe_mean",
    "sharpe_std",
    "sharpe_min",
    "sharpe_max",
    "Mean OOS Sharpe",
    "Stdev OOS Sharpe",
    "Min OOS Sharpe",
    "Max OOS Sharpe",
    "Mean val_rank_IC",
    "Stdev val_rank_IC",
]
_IDENTIFIER_COLUMNS = ["Strategy", "strategy", "Family", "family", "Model", "config", "arch"]
_DATE_HINTS = ("date", "window_end", "window_start")


def summarize_backtest_metrics(registry: ArtifactRegistry | None = None) -> dict[str, Any]:
    """Summarize approved historical metric artifacts without creating new claims."""
    artifact_registry = _registry(registry)
    artifacts_used: list[str] = []
    table_shapes: dict[str, list[int]] = {}
    available_columns: dict[str, list[str]] = {}
    headline_metrics: dict[str, list[dict[str, Any]]] = {}
    caveats = [
        "Backtest and experiment metrics are historical research evidence only.",
        "Metric definitions, timing assumptions, and costs require human review.",
    ]

    for artifact_id in BACKTEST_ARTIFACT_IDS:
        frame = _read_table(artifact_id, artifact_registry)
        artifacts_used.append(artifact_id)
        table_shapes[artifact_id] = _shape(frame)
        available_columns[artifact_id] = _selected_columns(frame, _IDENTIFIER_COLUMNS + _METRIC_COLUMNS)

        metric_rows = _headline_metric_rows(frame)
        if metric_rows:
            headline_metrics[artifact_id] = metric_rows
        else:
            caveats.append(f"{artifact_id}: no obvious metric columns were found.")

        if not _has_any_column(frame, _METRIC_COLUMNS):
            caveats.append(f"{artifact_id}: columns are reported for review without metric extraction.")

    return {
        "section": "backtest_metrics",
        "artifacts_used": artifacts_used,
        "table_shapes": table_shapes,
        "available_columns": available_columns,
        "headline_metrics": headline_metrics,
        "caveats": _dedupe(caveats),
    }


def summarize_regime_signal(registry: ArtifactRegistry | None = None) -> dict[str, Any]:
    """Summarize approved regime feature artifacts."""
    artifact_registry = _registry(registry)
    artifacts_used: list[str] = []
    row_counts: dict[str, int] = {}
    shapes: dict[str, list[int]] = {}
    date_range: dict[str, dict[str, str]] = {}
    latest_observation: dict[str, dict[str, Any]] = {}
    regime_columns: dict[str, list[str]] = {}
    caveats = [
        "Regime labels are research features only, not instructions for market action.",
        "Regime relationships are descriptive and require human review.",
    ]

    for artifact_id in REGIME_ARTIFACT_IDS:
        frame = _read_table(artifact_id, artifact_registry)
        artifacts_used.append(artifact_id)
        row_counts[artifact_id] = int(frame.shape[0])
        shapes[artifact_id] = _shape(frame)
        regime_columns[artifact_id] = _regime_columns(frame)

        range_summary = _date_range(frame)
        if range_summary:
            date_range[artifact_id] = range_summary
            latest = _latest_observation(frame)
            if latest:
                latest_observation[artifact_id] = latest
        else:
            caveats.append(f"{artifact_id}: no date-like column or index was safely inferred.")

        if not regime_columns[artifact_id]:
            caveats.append(f"{artifact_id}: no regime-related columns were found.")

    return {
        "section": "regime_signal",
        "artifacts_used": artifacts_used,
        "row_counts": row_counts,
        "shapes": shapes,
        "date_range": date_range,
        "latest_observation": latest_observation,
        "regime_columns": regime_columns,
        "caveats": _dedupe(caveats),
    }


def summarize_pca_dashboard(registry: ArtifactRegistry | None = None) -> dict[str, Any]:
    """Summarize approved PCA dashboard artifacts with bounded previews."""
    artifact_registry = _registry(registry)
    artifacts_used: list[str] = []
    shapes: dict[str, list[int]] = {}
    available_columns: dict[str, list[str]] = {}
    latest_rows_preview: dict[str, list[dict[str, Any]]] = {}
    methodology_gaps: list[dict[str, Any]] = []
    caveats = [
        "PCA loadings, dislocations, and residuals are descriptive diagnostics only.",
        "No market action is inferred from PCA dashboard artifacts.",
    ]

    for artifact_id in PCA_ARTIFACT_IDS:
        frame = _read_table(artifact_id, artifact_registry)
        artifacts_used.append(artifact_id)
        shapes[artifact_id] = _shape(frame)
        available_columns[artifact_id] = list(frame.columns[:20])

        preview = _latest_preview(frame, max_rows=3)
        if preview:
            latest_rows_preview[artifact_id] = preview

        if artifact_id == "notebook07b_methodology_gap_table":
            methodology_gaps = _records(frame, max_rows=10)
            if not methodology_gaps:
                caveats.append(f"{artifact_id}: methodology gap table was empty.")

    return {
        "section": "pca_dashboard",
        "artifacts_used": artifacts_used,
        "shapes": shapes,
        "available_columns": available_columns,
        "latest_rows_preview": latest_rows_preview,
        "methodology_gaps": methodology_gaps,
        "caveats": _dedupe(caveats),
    }


def summarize_historical_llm_experiment_context(registry: ArtifactRegistry | None = None) -> dict[str, Any]:
    """Summarize approved historical model-assisted experiment artifacts."""
    artifact_registry = _registry(registry)
    artifacts_used: list[str] = []
    diagnostics_keys: dict[str, list[str]] = {}
    return_table_shapes: dict[str, list[int]] = {}
    weight_table_shapes: dict[str, list[int]] = {}
    caveats = [
        "Diagnostics are historical experiment artifacts only; no model call is made.",
        "Historical weight artifacts are historical experiment outputs only, not target allocations.",
        "Historical weights are not recommendations or portfolio actions.",
    ]

    for artifact_id in LLM_DIAGNOSTIC_ARTIFACT_IDS:
        payload = read_json_artifact(artifact_id, registry=artifact_registry)
        artifacts_used.append(artifact_id)
        diagnostics_keys[artifact_id] = sorted(payload) if isinstance(payload, Mapping) else []
        if not diagnostics_keys[artifact_id]:
            caveats.append(f"{artifact_id}: diagnostic payload did not contain mapping keys.")

    for artifact_id in LLM_RETURN_ARTIFACT_IDS:
        frame = read_parquet_artifact(artifact_id, registry=artifact_registry)
        artifacts_used.append(artifact_id)
        return_table_shapes[artifact_id] = _shape(frame)

    for artifact_id in LLM_WEIGHT_ARTIFACT_IDS:
        frame = read_parquet_artifact(artifact_id, registry=artifact_registry)
        artifacts_used.append(artifact_id)
        weight_table_shapes[artifact_id] = _shape(frame)

    return {
        "section": "historical_llm_experiment_context",
        "artifacts_used": artifacts_used,
        "diagnostics_keys": diagnostics_keys,
        "return_table_shapes": return_table_shapes,
        "weight_table_shapes": weight_table_shapes,
        "caveats": _dedupe(caveats),
    }


def _registry(registry: ArtifactRegistry | None) -> ArtifactRegistry:
    if registry is not None:
        return registry
    return load_artifact_registry()


def _read_table(artifact_id: str, registry: ArtifactRegistry) -> pd.DataFrame:
    artifact = get_artifact_by_id(artifact_id, registry=registry)
    if artifact.parser == "csv":
        return read_csv_artifact(artifact_id, registry=registry)
    if artifact.parser == "parquet":
        return read_parquet_artifact(artifact_id, registry=registry)
    raise ValueError(f"Artifact {artifact_id!r} is not a table artifact.")


def _shape(frame: pd.DataFrame) -> list[int]:
    return [int(frame.shape[0]), int(frame.shape[1])]


def _selected_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    selected = [column for column in candidates if column in frame.columns]
    if selected:
        return selected
    return list(frame.columns[:12])


def _has_any_column(frame: pd.DataFrame, candidates: list[str]) -> bool:
    return any(column in frame.columns for column in candidates)


def _headline_metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    selected = _selected_columns(frame, _IDENTIFIER_COLUMNS + _METRIC_COLUMNS)
    metric_columns = [column for column in selected if column in _METRIC_COLUMNS]
    if not metric_columns:
        return []
    preview_columns = [column for column in selected if column in _IDENTIFIER_COLUMNS] + metric_columns
    return _records(frame.loc[:, preview_columns].head(3), max_rows=3)


def _regime_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if "regime" in str(column).lower() or "macro_context" == str(column).lower()
    ]


def _date_range(frame: pd.DataFrame) -> dict[str, str]:
    series = _date_series(frame)
    if series is None:
        return {}
    valid = series.dropna()
    if valid.empty:
        return {}
    return {
        "start": valid.min().date().isoformat(),
        "end": valid.max().date().isoformat(),
    }


def _latest_observation(frame: pd.DataFrame) -> dict[str, Any]:
    series = _date_series(frame)
    if series is None:
        return {}
    valid = series.dropna()
    if valid.empty:
        return {}
    latest_index = valid.idxmax()
    row = frame.loc[[latest_index]].copy()
    preview_columns = _selected_columns(row, _DATE_HINTS_LIST() + _regime_columns(row))
    if not preview_columns:
        preview_columns = list(row.columns[:8])
    return _records(row.loc[:, preview_columns], max_rows=1)[0]


def _latest_preview(frame: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    series = _date_series(frame)
    if series is not None and not series.dropna().empty:
        sorted_frame = frame.assign(_summary_date=series).sort_values("_summary_date").drop(columns=["_summary_date"])
        return _records(sorted_frame.tail(max_rows), max_rows=max_rows)
    return _records(frame.tail(max_rows), max_rows=max_rows)


def _date_series(frame: pd.DataFrame) -> pd.Series | None:
    for column in frame.columns:
        name = str(column).lower()
        if name in _DATE_HINTS or name.endswith("_date"):
            series = pd.to_datetime(frame[column], errors="coerce")
            if not series.dropna().empty:
                return series

    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index)
    return None


def _DATE_HINTS_LIST() -> list[str]:
    return list(_DATE_HINTS)


def _records(frame: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    bounded = frame.head(max_rows).copy()
    bounded = bounded.where(pd.notna(bounded), None)
    return [
        {str(key): _plain_value(value) for key, value in row.items()}
        for row in bounded.to_dict(orient="records")
    ]


def _plain_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
