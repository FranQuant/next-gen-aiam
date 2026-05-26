from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from aiam.research_agents.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from aiam.research_agents.validators import validate_phase6_artifact_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    manifest_path = tmp_path / "phase6_artifact_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _first_artifact(manifest: dict) -> dict:
    return manifest["approved_artifacts"][0]


def _mutated_manifest(tmp_path: Path, mutator) -> Path:
    manifest = deepcopy(load_manifest(REPO_ROOT / DEFAULT_MANIFEST_PATH))
    mutator(manifest)
    return _write_manifest(tmp_path, manifest)


def _validate_mutation(tmp_path: Path, mutator):
    manifest_path = _mutated_manifest(tmp_path, mutator)
    return validate_phase6_artifact_manifest(manifest_path=manifest_path, repo_root=REPO_ROOT)


def test_current_committed_manifest_validates_successfully():
    result = validate_phase6_artifact_manifest(repo_root=REPO_ROOT)

    assert result.ok is True
    assert result.errors == []
    assert result.approved_artifact_count > 0
    assert result.approved_artifact_ids


def test_duplicate_artifact_ids_fail(tmp_path):
    def mutate(manifest: dict) -> None:
        manifest["approved_artifacts"].append(deepcopy(manifest["approved_artifacts"][0]))

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("Duplicate artifact_id" in error for error in result.errors)


def test_forbidden_default_false_fails(tmp_path):
    result = _validate_mutation(tmp_path, lambda manifest: manifest.update(forbidden_default=False))

    assert result.ok is False
    assert any("forbidden_default must be exactly true" in error for error in result.errors)


def test_non_existing_approved_path_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["path"] = "data/published/does_not_exist.csv"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("path does not exist" in error for error in result.errors)


def test_absolute_path_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["path"] = "/tmp/phase6_artifact.csv"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("path must be repository-relative" in error for error in result.errors)


def test_path_with_parent_reference_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["path"] = "data/../published/README.md"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("must not contain '..'" in error for error in result.errors)


def test_excluded_raw_llm_cache_path_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        artifact = _first_artifact(manifest)
        artifact["path"] = "data/cache/llm/example.json"
        artifact["parser"] = "json"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("excluded surface" in error or "raw LLM cache" in error for error in result.errors)


def test_model_weight_extension_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["path"] = "results/rl/n29/agent_seed_1.pt"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("model/checkpoint extension is forbidden" in error for error in result.errors)


def test_invalid_parser_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["parser"] = "python"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("parser 'python' is not allowed" in error for error in result.errors)


def test_invalid_allowed_use_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["allowed_use"] = "portfolio_recommendation"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("allowed_use 'portfolio_recommendation' is not allowed" in error for error in result.errors)


def test_invalid_canonicality_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        _first_artifact(manifest)["canonicality"] = "actionable"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("canonicality 'actionable' is not allowed" in error for error in result.errors)


@pytest.mark.parametrize("field_name", ["research_only", "human_review_required"])
def test_required_boolean_flags_false_fail(tmp_path, field_name):
    def mutate(manifest: dict) -> None:
        manifest[field_name] = False
        _first_artifact(manifest)[field_name] = False

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any(f"{field_name} must be exactly true" in error for error in result.errors)


def test_parser_extension_mismatch_fails(tmp_path):
    def mutate(manifest: dict) -> None:
        artifact = _first_artifact(manifest)
        artifact["path"] = "data/published/README.md"
        artifact["parser"] = "csv"

    result = _validate_mutation(tmp_path, mutate)

    assert result.ok is False
    assert any("parser 'csv' is inconsistent with path extension" in error for error in result.errors)
