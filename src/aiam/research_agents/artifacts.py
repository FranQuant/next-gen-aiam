from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aiam.research_agents.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from aiam.research_agents.validators import validate_phase6_artifact_manifest


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    path: Path
    artifact_type: str
    parser: str
    role: str
    allowed_use: str
    canonicality: str
    research_only: bool
    human_review_required: bool
    caveats: str


ArtifactRegistry = dict[str, ArtifactRecord]


def load_artifact_registry(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> ArtifactRegistry:
    """Load a validated Phase 6 artifact registry without reading artifact content."""
    repo_root_path = Path(repo_root)
    validation = validate_phase6_artifact_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root_path,
    )
    if not validation.ok:
        formatted_errors = "\n".join(f"- {error}" for error in validation.errors)
        raise ValueError(f"Phase 6 artifact manifest validation failed:\n{formatted_errors}")

    manifest_file = _resolve_manifest_path(manifest_path=manifest_path, repo_root=repo_root_path)
    manifest = load_manifest(manifest_file)

    registry: ArtifactRegistry = {}
    for artifact in manifest["approved_artifacts"]:
        record = ArtifactRecord(
            artifact_id=artifact["artifact_id"],
            path=Path(artifact["path"]),
            artifact_type=artifact["artifact_type"],
            parser=artifact["parser"],
            role=artifact["role"],
            allowed_use=artifact["allowed_use"],
            canonicality=artifact["canonicality"],
            research_only=artifact["research_only"],
            human_review_required=artifact["human_review_required"],
            caveats=artifact["caveats"],
        )
        registry[record.artifact_id] = record

    return registry


def list_approved_artifacts(
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> list[ArtifactRecord]:
    """List approved artifacts from a validated registry."""
    artifact_registry = (
        registry
        if registry is not None
        else load_artifact_registry(
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
    )
    return list(artifact_registry.values())


def get_artifact_by_id(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> ArtifactRecord:
    """Return a single approved artifact record, failing closed on unknown IDs."""
    artifact_registry = (
        registry
        if registry is not None
        else load_artifact_registry(
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
    )
    try:
        return artifact_registry[artifact_id]
    except KeyError as exc:
        raise KeyError(f"Unknown Phase 6 artifact_id: {artifact_id}") from exc


def resolve_artifact_path(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> Path:
    """Resolve only paths that are present in the validated approved manifest."""
    repo_root_path = Path(repo_root).resolve()
    artifact = get_artifact_by_id(
        artifact_id=artifact_id,
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root_path,
    )
    resolved_path = (repo_root_path / artifact.path).resolve()
    try:
        resolved_path.relative_to(repo_root_path)
    except ValueError as exc:
        raise ValueError(f"Approved artifact path escapes repository root: {artifact_id}") from exc
    return resolved_path


def read_csv_artifact(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read an approved CSV artifact by artifact_id."""
    path = _checked_artifact_path(
        artifact_id=artifact_id,
        expected_parser="csv",
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    return pd.read_csv(path, **read_csv_kwargs)


def read_parquet_artifact(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    **read_parquet_kwargs: Any,
) -> pd.DataFrame:
    """Read an approved parquet artifact by artifact_id."""
    path = _checked_artifact_path(
        artifact_id=artifact_id,
        expected_parser="parquet",
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    return pd.read_parquet(path, **read_parquet_kwargs)


def read_json_artifact(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> Any:
    """Read an approved JSON artifact by artifact_id."""
    path = _checked_artifact_path(
        artifact_id=artifact_id,
        expected_parser="json",
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_markdown_artifact(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> str:
    """Read an approved markdown artifact by artifact_id."""
    path = _checked_artifact_path(
        artifact_id=artifact_id,
        expected_parser="markdown",
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    return path.read_text(encoding="utf-8")


def read_image_metadata_only(
    artifact_id: str,
    registry: ArtifactRegistry | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Return filesystem metadata for an approved image without parsing pixels."""
    path = _checked_artifact_path(
        artifact_id=artifact_id,
        expected_parser="image_metadata_only",
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    artifact = get_artifact_by_id(
        artifact_id=artifact_id,
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    stat_result = path.stat()
    return {
        "artifact_id": artifact.artifact_id,
        "path": artifact.path.as_posix(),
        "resolved_path": path,
        "suffix": path.suffix.lower(),
        "file_size_bytes": stat_result.st_size,
        "parser": artifact.parser,
    }


def _resolve_manifest_path(manifest_path: str | Path, repo_root: Path) -> Path:
    manifest_file = Path(manifest_path)
    if manifest_file.is_absolute():
        return manifest_file
    return repo_root / manifest_file


def _checked_artifact_path(
    artifact_id: str,
    expected_parser: str,
    registry: ArtifactRegistry | None,
    manifest_path: str | Path,
    repo_root: str | Path,
) -> Path:
    artifact = get_artifact_by_id(
        artifact_id=artifact_id,
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    if artifact.parser != expected_parser:
        raise ValueError(
            f"Artifact {artifact_id!r} uses parser {artifact.parser!r}; "
            f"expected {expected_parser!r}"
        )
    return resolve_artifact_path(
        artifact_id=artifact_id,
        registry=registry,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
