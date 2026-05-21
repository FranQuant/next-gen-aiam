from __future__ import annotations

import json
import re

from pydantic import BaseModel, field_validator, model_validator


class AssetView(BaseModel):
    asset: str
    expected_excess_return: float
    confidence: float

    @field_validator("asset")
    @classmethod
    def asset_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("asset must be non-empty")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    @field_validator("expected_excess_return")
    @classmethod
    def return_magnitude(cls, v: float) -> float:
        if abs(v) > 1.0:
            raise ValueError(f"|expected_excess_return| must be ≤ 1.0, got {v}")
        return v


class ViewSet(BaseModel):
    views: list[AssetView]
    rationale: str | None = None

    @model_validator(mode="after")
    def no_duplicate_assets(self) -> "ViewSet":
        assets = [v.asset for v in self.views]
        if len(assets) != len(set(assets)):
            seen, dups = set(), []
            for a in assets:
                if a in seen:
                    dups.append(a)
                seen.add(a)
            raise ValueError(f"Duplicate assets in views: {dups}")
        return self


class ParseError(ValueError):
    pass


def parse_viewset(raw: str) -> ViewSet:
    """Strip markdown fences and validate JSON into ViewSet; raises ParseError on failure."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON decode failed: {exc}\nRaw: {raw!r}") from exc
    try:
        return ViewSet.model_validate(data)
    except Exception as exc:
        raise ParseError(f"ViewSet validation failed: {exc}") from exc
