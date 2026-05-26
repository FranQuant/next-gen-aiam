from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_MANIFEST_PATH = Path("configs/phase6_artifact_manifest.yaml")


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load the Phase 6 artifact manifest as a plain dictionary."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest is None:
        return {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must load to a mapping: {manifest_path}")
    return manifest
