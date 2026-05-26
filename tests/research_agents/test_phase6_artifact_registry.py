from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from aiam.research_agents.artifacts import (
    ArtifactRecord,
    get_artifact_by_id,
    list_approved_artifacts,
    load_artifact_registry,
    read_csv_artifact,
    read_image_metadata_only,
    read_json_artifact,
    read_markdown_artifact,
    read_parquet_artifact,
    resolve_artifact_path,
)
from aiam.research_agents.manifest import DEFAULT_MANIFEST_PATH, load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    manifest_path = tmp_path / "phase6_artifact_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _registry():
    return load_artifact_registry(repo_root=REPO_ROOT)


def test_registry_loads_successfully_from_committed_manifest():
    registry = _registry()

    assert isinstance(registry, dict)
    assert all(isinstance(record, ArtifactRecord) for record in registry.values())
    assert "data_published_full_comparison_with_rl" in registry


def test_registry_contains_75_approved_artifacts():
    registry = _registry()

    assert len(registry) == 75
    assert len(list_approved_artifacts(registry=registry)) == 75


def test_known_artifact_id_can_be_retrieved():
    artifact = get_artifact_by_id("data_published_readme", registry=_registry())

    assert artifact.artifact_id == "data_published_readme"
    assert artifact.path == Path("data/published/README.md")
    assert artifact.parser == "markdown"
    assert artifact.research_only is True
    assert artifact.human_review_required is True


def test_unknown_artifact_id_fails():
    with pytest.raises(KeyError, match="Unknown Phase 6 artifact_id"):
        get_artifact_by_id("not_approved", registry=_registry())


def test_path_resolution_works_for_known_artifact():
    resolved_path = resolve_artifact_path("data_published_readme", registry=_registry(), repo_root=REPO_ROOT)

    assert resolved_path == (REPO_ROOT / "data/published/README.md").resolve()
    assert resolved_path.exists()


def test_arbitrary_path_reading_is_not_exposed():
    read_functions = [
        read_csv_artifact,
        read_parquet_artifact,
        read_json_artifact,
        read_markdown_artifact,
        read_image_metadata_only,
    ]

    for read_function in read_functions:
        signature = inspect.signature(read_function)
        assert list(signature.parameters)[0] == "artifact_id"
        assert "path" not in signature.parameters

    with pytest.raises(KeyError):
        read_markdown_artifact(".env", registry=_registry(), repo_root=REPO_ROOT)


def test_parser_mismatch_fails_before_reading():
    with pytest.raises(ValueError, match="expected 'csv'"):
        read_csv_artifact("data_published_readme", registry=_registry(), repo_root=REPO_ROOT)


def test_csv_read_works_for_small_approved_csv_artifact():
    frame = read_csv_artifact(
        "data_published_full_comparison_with_rl",
        registry=_registry(),
        repo_root=REPO_ROOT,
        nrows=3,
    )

    assert frame.shape[0] == 3
    assert "Sharpe" in frame.columns


def test_parquet_read_works_for_small_approved_parquet_artifact():
    frame = read_parquet_artifact("data_published_regime_signals", registry=_registry(), repo_root=REPO_ROOT)

    assert frame.shape[0] > 0
    assert "regime_GDP" in frame.columns


def test_markdown_read_works_for_approved_markdown_artifact():
    text = read_markdown_artifact("data_published_readme", registry=_registry(), repo_root=REPO_ROOT)

    assert "published" in text.lower()


def test_json_read_works_for_approved_json_diagnostic_artifact():
    payload = read_json_artifact("llm_equilibrium_diagnostics", registry=_registry(), repo_root=REPO_ROOT)

    assert isinstance(payload, dict)
    assert payload


def test_image_metadata_only_returns_metadata_without_pixel_parsing():
    metadata = read_image_metadata_only(
        "notebook07_fig_cluster_dendrogram",
        registry=_registry(),
        repo_root=REPO_ROOT,
    )

    assert metadata["artifact_id"] == "notebook07_fig_cluster_dendrogram"
    assert metadata["parser"] == "image_metadata_only"
    assert metadata["suffix"] == ".png"
    assert metadata["file_size_bytes"] > 0
    assert metadata["resolved_path"].exists()
    assert "width" not in metadata
    assert "height" not in metadata
    assert "pixels" not in metadata


def test_registry_construction_fails_if_manifest_validation_fails(tmp_path):
    manifest = deepcopy(load_manifest(REPO_ROOT / DEFAULT_MANIFEST_PATH))
    manifest["approved_artifacts"][0]["path"] = "data/published/does_not_exist.csv"
    manifest_path = _write_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="Phase 6 artifact manifest validation failed") as exc_info:
        load_artifact_registry(manifest_path=manifest_path, repo_root=REPO_ROOT)

    assert "path does not exist" in str(exc_info.value)
