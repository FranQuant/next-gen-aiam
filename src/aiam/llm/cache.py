from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("data/cache/llm")


class PromptCache:
    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        disabled: bool = False,
    ) -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_DIR
        self.disabled = disabled
        if not self.disabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, system: str | None, prompt: str) -> str:
        raw = json.dumps({"model": model, "system": system, "prompt": prompt}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, system: str | None, prompt: str) -> str | None:
        if self.disabled:
            return None
        path = self._dir / f"{self._key(model, system, prompt)}.json"
        if path.exists():
            data = json.loads(path.read_text())
            logger.debug("cache hit: %s", path.name[:16])
            return data["response"]
        return None

    def set(self, model: str, system: str | None, prompt: str, response: str) -> None:
        if self.disabled:
            return
        path = self._dir / f"{self._key(model, system, prompt)}.json"
        path.write_text(
            json.dumps({"model": model, "system": system, "prompt": prompt, "response": response})
        )
        logger.debug("cache set: %s", path.name[:16])
