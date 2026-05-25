from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from aiam.research_agents.manifest import DEFAULT_MANIFEST_PATH, load_manifest


TOP_LEVEL_REQUIRED_KEYS = (
    "manifest_version",
    "phase",
    "scope",
    "generated_for",
    "research_only",
    "human_review_required",
    "forbidden_default",
    "policy",
    "excluded_surfaces",
    "approved_artifacts",
)

ARTIFACT_REQUIRED_FIELDS = (
    "artifact_id",
    "path",
    "artifact_type",
    "parser",
    "role",
    "allowed_use",
    "canonicality",
    "research_only",
    "human_review_required",
    "caveats",
)

PARSER_EXTENSIONS = {
    "csv": (".csv",),
    "parquet": (".parquet",),
    "markdown": (".md", ".markdown"),
    "pdf": (".pdf",),
    "json": (".json",),
    "image_metadata_only": (".png", ".jpg", ".jpeg", ".webp"),
}

FORBIDDEN_MODEL_EXTENSIONS = {".pt", ".pth", ".ckpt", ".pkl", ".joblib"}
FORBIDDEN_PRIVATE_SURFACES = (
    ".env",
    ".claude/",
    ".venv/",
    "notebooks/.ipynb_checkpoints/",
    "notebooks/_rl_dev/",
    "__pycache__",
    ".pytest_cache",
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approved_artifact_count: int = 0
    approved_artifact_ids: list[str] = field(default_factory=list)


def validate_phase6_artifact_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    repo_root: str | Path = ".",
) -> ValidationResult:
    """Validate the Phase 6 artifact manifest without reading artifact content."""
    repo_root_path = Path(repo_root)
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = repo_root_path / manifest_file

    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_file.exists():
        return ValidationResult(
            ok=False,
            errors=[f"Manifest file does not exist: {manifest_file}"],
            warnings=warnings,
        )

    try:
        manifest = load_manifest(manifest_file)
    except Exception as exc:  # pragma: no cover - defensive load failure reporting.
        return ValidationResult(
            ok=False,
            errors=[f"Failed to load manifest {manifest_file}: {exc}"],
            warnings=warnings,
        )

    _validate_top_level(manifest, errors)

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        policy = {}

    allowed_parsers = _policy_list(policy, "allowed_parsers", errors)
    allowed_uses = _policy_list(policy, "allowed_uses", errors)
    canonicality_values = _policy_list(policy, "canonicality_values", errors)

    excluded_surfaces = manifest.get("excluded_surfaces")
    excluded_patterns = _excluded_patterns(excluded_surfaces, errors)

    approved_artifacts = manifest.get("approved_artifacts")
    if not isinstance(approved_artifacts, list):
        approved_artifacts = []

    approved_artifact_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, artifact in enumerate(approved_artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"approved_artifacts[{index}] must be a mapping")
            continue
        artifact_id = artifact.get("artifact_id")
        if isinstance(artifact_id, str):
            approved_artifact_ids.append(artifact_id)
            if artifact_id in seen_ids:
                errors.append(f"Duplicate artifact_id: {artifact_id}")
            seen_ids.add(artifact_id)
        else:
            artifact_id = f"approved_artifacts[{index}]"
        _validate_artifact(
            artifact=artifact,
            artifact_label=str(artifact_id),
            repo_root=repo_root_path,
            allowed_parsers=allowed_parsers,
            allowed_uses=allowed_uses,
            canonicality_values=canonicality_values,
            excluded_patterns=excluded_patterns,
            errors=errors,
        )

    return ValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        approved_artifact_count=len(approved_artifacts),
        approved_artifact_ids=approved_artifact_ids,
    )


def _validate_top_level(manifest: dict[str, Any], errors: list[str]) -> None:
    for key in TOP_LEVEL_REQUIRED_KEYS:
        if key not in manifest:
            errors.append(f"Missing top-level key: {key}")

    for key in ("forbidden_default", "research_only", "human_review_required"):
        if manifest.get(key) is not True:
            errors.append(f"{key} must be exactly true")

    for key in ("approved_artifacts", "excluded_surfaces"):
        value = manifest.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{key} must be a non-empty list")


def _policy_list(policy: dict[str, Any], key: str, errors: list[str]) -> set[str]:
    value = policy.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"policy.{key} must be a non-empty list")
        return set()
    invalid_values = [item for item in value if not isinstance(item, str)]
    if invalid_values:
        errors.append(f"policy.{key} must contain only strings")
    return {item for item in value if isinstance(item, str)}


def _excluded_patterns(excluded_surfaces: Any, errors: list[str]) -> list[str]:
    if not isinstance(excluded_surfaces, list):
        return []

    patterns: list[str] = []
    for index, surface in enumerate(excluded_surfaces):
        if not isinstance(surface, dict):
            errors.append(f"excluded_surfaces[{index}] must be a mapping")
            continue
        path = surface.get("path")
        if isinstance(path, str) and path:
            patterns.append(path)
        else:
            errors.append(f"excluded_surfaces[{index}].path must be a non-empty string")
    return patterns


def _validate_artifact(
    artifact: dict[str, Any],
    artifact_label: str,
    repo_root: Path,
    allowed_parsers: set[str],
    allowed_uses: set[str],
    canonicality_values: set[str],
    excluded_patterns: list[str],
    errors: list[str],
) -> None:
    for field_name in ARTIFACT_REQUIRED_FIELDS:
        if field_name not in artifact:
            errors.append(f"{artifact_label}: missing required field {field_name}")

    artifact_path = artifact.get("path")
    parser = artifact.get("parser")
    allowed_use = artifact.get("allowed_use")
    canonicality = artifact.get("canonicality")

    if parser not in allowed_parsers:
        errors.append(f"{artifact_label}: parser {parser!r} is not allowed")
    if allowed_use not in allowed_uses:
        errors.append(f"{artifact_label}: allowed_use {allowed_use!r} is not allowed")
    if canonicality not in canonicality_values:
        errors.append(f"{artifact_label}: canonicality {canonicality!r} is not allowed")
    if artifact.get("research_only") is not True:
        errors.append(f"{artifact_label}: research_only must be exactly true")
    if artifact.get("human_review_required") is not True:
        errors.append(f"{artifact_label}: human_review_required must be exactly true")

    if not isinstance(artifact_path, str) or not artifact_path:
        errors.append(f"{artifact_label}: path must be a non-empty string")
        return

    _validate_artifact_path(
        artifact_label=artifact_label,
        artifact_path=artifact_path,
        parser=parser,
        repo_root=repo_root,
        excluded_patterns=excluded_patterns,
        errors=errors,
    )


def _validate_artifact_path(
    artifact_label: str,
    artifact_path: str,
    parser: Any,
    repo_root: Path,
    excluded_patterns: list[str],
    errors: list[str],
) -> None:
    pure_path = PurePosixPath(artifact_path)
    if pure_path.is_absolute() or Path(artifact_path).is_absolute():
        errors.append(f"{artifact_label}: path must be repository-relative: {artifact_path}")
        return

    if ".." in pure_path.parts:
        errors.append(f"{artifact_label}: path must not contain '..': {artifact_path}")
        return

    if _matches_excluded_surface(artifact_path, excluded_patterns):
        errors.append(f"{artifact_label}: path matches excluded surface: {artifact_path}")

    if _is_raw_llm_cache_path(pure_path):
        errors.append(f"{artifact_label}: raw LLM cache JSON files are forbidden: {artifact_path}")

    if _matches_private_surface(artifact_path):
        errors.append(f"{artifact_label}: credentials or private development surface is forbidden: {artifact_path}")

    suffix = pure_path.suffix.lower()
    if suffix in FORBIDDEN_MODEL_EXTENSIONS:
        errors.append(f"{artifact_label}: model/checkpoint extension is forbidden: {artifact_path}")

    if isinstance(parser, str) and parser in PARSER_EXTENSIONS:
        expected_suffixes = PARSER_EXTENSIONS[parser]
        if suffix not in expected_suffixes:
            errors.append(
                f"{artifact_label}: parser {parser!r} is inconsistent with path extension "
                f"{suffix!r}; expected one of {expected_suffixes}"
            )

    resolved_path = repo_root / artifact_path
    if not resolved_path.exists():
        errors.append(f"{artifact_label}: path does not exist: {artifact_path}")
    elif not resolved_path.is_file():
        errors.append(f"{artifact_label}: path must point to a file: {artifact_path}")


def _matches_excluded_surface(path: str, patterns: list[str]) -> bool:
    normalized = path.strip("/")
    for pattern in patterns:
        if " " in pattern:
            continue
        normalized_pattern = pattern.strip("/")
        if not normalized_pattern:
            continue
        if pattern.endswith("/") and (
            normalized == normalized_pattern or normalized.startswith(f"{normalized_pattern}/")
        ):
            return True
        if fnmatch(normalized, normalized_pattern):
            return True
    return False


def _matches_private_surface(path: str) -> bool:
    normalized = path.strip("/")
    for surface in FORBIDDEN_PRIVATE_SURFACES:
        normalized_surface = surface.strip("/")
        if surface.endswith("/") and (
            normalized == normalized_surface or normalized.startswith(f"{normalized_surface}/")
        ):
            return True
        parts = PurePosixPath(normalized).parts
        if normalized == normalized_surface or normalized_surface in parts:
            return True
    return False


def _is_raw_llm_cache_path(path: PurePosixPath) -> bool:
    return (
        len(path.parts) == 4
        and path.parts[0] == "data"
        and path.parts[1] == "cache"
        and path.parts[2] == "llm"
        and path.suffix.lower() == ".json"
    )
