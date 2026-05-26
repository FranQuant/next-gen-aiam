from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiam.research_agents.validators import validate_phase6_artifact_manifest


def main() -> int:
    result = validate_phase6_artifact_manifest()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
