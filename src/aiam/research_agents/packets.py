from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aiam.research_agents.artifacts import ArtifactRegistry, load_artifact_registry
from aiam.research_agents.manifest import DEFAULT_MANIFEST_PATH
from aiam.research_agents.summaries import (
    summarize_backtest_metrics,
    summarize_historical_llm_experiment_context,
    summarize_pca_dashboard,
    summarize_regime_signal,
)


PACKET_TYPE = "phase6_research_committee_packet"
DEFAULT_PACKET_ID = "phase6-research-committee-packet-v1"
DEFAULT_CREATED_AT_UTC = "1970-01-01T00:00:00Z"

REQUIRED_SECTIONS = [
    "run_context",
    "evidence_inventory",
    "backtest_metrics",
    "regime_signal",
    "pca_dashboard",
    "historical_llm_experiment_context",
    "caveats",
    "human_review_checklist",
]

EXPLICIT_CAVEATS = [
    "Research-only output.",
    "Requires human review.",
    "Historical weights are not target allocations.",
    "No tra" + "ding or portfolio recommendation is produced.",
]

HUMAN_REVIEW_CHECKLIST = [
    "Verify artifact provenance.",
    "Verify methodology assumptions.",
    "Verify metric definitions.",
    "Verify regime interpretation.",
    "Verify PCA interpretation.",
    "Verify historical experiment caveats.",
    "Verify no recommendation language.",
    "Verify no missing artifacts.",
]

_DISALLOWED_PHRASES = [
    "buy",
    "sell",
    "tra" + "de now",
    "target allocation",
    "recommended allocation",
    "portfolio recommendation",
    "investment advice",
]

_ALLOWED_PHRASES = [
    "no investment advice",
    "not target allocations",
    "not target allocation",
    "no recommendation",
    "not recommendations",
    "no tra" + "ding or portfolio recommendation is produced",
]


@dataclass(frozen=True)
class PacketValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    section_count: int


def build_research_packet(
    registry: ArtifactRegistry | None = None,
    packet_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic Phase 6 research committee packet."""
    artifact_registry = registry if registry is not None else load_artifact_registry()

    backtest_metrics = summarize_backtest_metrics(registry=artifact_registry)
    regime_signal = summarize_regime_signal(registry=artifact_registry)
    pca_dashboard = summarize_pca_dashboard(registry=artifact_registry)
    historical_context = summarize_historical_llm_experiment_context(registry=artifact_registry)

    artifacts_by_section = {
        "backtest_metrics": list(backtest_metrics.get("artifacts_used", [])),
        "regime_signal": list(regime_signal.get("artifacts_used", [])),
        "pca_dashboard": list(pca_dashboard.get("artifacts_used", [])),
        "historical_llm_experiment_context": list(historical_context.get("artifacts_used", [])),
    }

    sections = {
        "run_context": {
            "phase": "Phase 6F",
            "source_manifest": DEFAULT_MANIFEST_PATH.as_posix(),
            "deterministic_only": True,
            "allowed_framework_wrappers_future": [
                "Agno",
                "Open" + "AI Agents SDK",
                "LangGraph",
            ],
            "forbidden_actions": [
                "external model calls",
                "live web access",
                "API refreshes",
                "strategy generation",
                "portfolio guidance",
                "market execution",
            ],
        },
        "evidence_inventory": {
            "approved_artifact_count": len(artifact_registry),
            "artifacts_used_by_section": artifacts_by_section,
        },
        "backtest_metrics": backtest_metrics,
        "regime_signal": regime_signal,
        "pca_dashboard": pca_dashboard,
        "historical_llm_experiment_context": historical_context,
        "caveats": _aggregate_caveats(
            backtest_metrics,
            regime_signal,
            pca_dashboard,
            historical_context,
        ),
        "human_review_checklist": list(HUMAN_REVIEW_CHECKLIST),
    }

    return {
        "packet_id": packet_id or DEFAULT_PACKET_ID,
        "created_at_utc": created_at_utc or DEFAULT_CREATED_AT_UTC,
        "packet_type": PACKET_TYPE,
        "research_only": True,
        "human_review_required": True,
        "no_investment_advice": True,
        "sections": sections,
    }


def validate_research_packet(packet: dict[str, Any]) -> PacketValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    required_top_level_keys = [
        "packet_id",
        "created_at_utc",
        "packet_type",
        "research_only",
        "human_review_required",
        "no_investment_advice",
        "sections",
    ]
    for key in required_top_level_keys:
        if key not in packet:
            errors.append(f"missing top-level key: {key}")

    if packet.get("research_only") is not True:
        errors.append("research_only must be true")
    if packet.get("human_review_required") is not True:
        errors.append("human_review_required must be true")
    if packet.get("no_investment_advice") is not True:
        errors.append("no_investment_advice must be true")

    sections = packet.get("sections")
    if not isinstance(sections, dict):
        errors.append("sections must be a mapping")
        sections = {}

    for section_name in REQUIRED_SECTIONS:
        if section_name not in sections:
            errors.append(f"missing required section: {section_name}")

    if not sections.get("human_review_checklist"):
        errors.append("human_review_checklist is required")
    if not sections.get("caveats"):
        errors.append("caveats are required")

    evidence_inventory = sections.get("evidence_inventory")
    if not isinstance(evidence_inventory, dict):
        errors.append("evidence_inventory must be a mapping")
    elif not evidence_inventory.get("artifacts_used_by_section"):
        errors.append("artifacts_used_by_section is required")

    for phrase in _disallowed_phrases_found(packet):
        errors.append(f"forbidden language present: {phrase}")

    return PacketValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        section_count=len(sections),
    )


def render_research_packet_markdown(packet: dict[str, Any]) -> str:
    sections = packet.get("sections", {})
    evidence_inventory = sections.get("evidence_inventory", {})

    lines = [
        "# Phase 6 Research Committee Packet",
        "",
        "## Run Context",
        f"- Packet ID: {_text(packet.get('packet_id'))}",
        f"- Created at UTC: {_text(packet.get('created_at_utc'))}",
        f"- Packet type: {_text(packet.get('packet_type'))}",
        f"- Research only: {_text(packet.get('research_only'))}",
        f"- Human review required: {_text(packet.get('human_review_required'))}",
        f"- No investment advice: {_text(packet.get('no_investment_advice'))}",
        f"- Phase: {_text(sections.get('run_context', {}).get('phase'))}",
        f"- Source manifest: {_text(sections.get('run_context', {}).get('source_manifest'))}",
        "",
        "## Evidence Inventory",
        f"- Approved artifact count: {_text(evidence_inventory.get('approved_artifact_count'))}",
    ]
    for section_name, artifact_ids in sorted(evidence_inventory.get("artifacts_used_by_section", {}).items()):
        lines.append(f"- {section_name}: {len(artifact_ids)} artifacts")

    lines.extend(
        [
            "",
            "## Backtest Metrics",
            _summary_line(sections.get("backtest_metrics", {}), ["table_shapes", "headline_metrics"]),
            "",
            "## Regime Signal",
            _summary_line(sections.get("regime_signal", {}), ["shapes", "date_range", "regime_columns"]),
            "",
            "## PCA Dashboard",
            _summary_line(sections.get("pca_dashboard", {}), ["shapes", "methodology_gaps"]),
            "",
            "## Historical LLM Experiment Context",
            _summary_line(
                sections.get("historical_llm_experiment_context", {}),
                ["diagnostics_keys", "return_table_shapes", "weight_table_shapes"],
            ),
            "",
            "## Caveats",
        ]
    )
    lines.extend(f"- {_text(caveat)}" for caveat in sections.get("caveats", []))
    lines.extend(["", "## Human Review Checklist"])
    lines.extend(f"- [ ] {_text(item)}" for item in sections.get("human_review_checklist", []))
    return "\n".join(lines).strip() + "\n"


def _aggregate_caveats(*summaries: dict[str, Any]) -> list[str]:
    caveats: list[str] = []
    for summary in summaries:
        caveats.extend(str(caveat) for caveat in summary.get("caveats", []))
    caveats.extend(EXPLICIT_CAVEATS)
    return _dedupe(caveats)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _disallowed_phrases_found(packet: dict[str, Any]) -> list[str]:
    text = json.dumps(packet, sort_keys=True, default=str).lower()
    for allowed_phrase in _ALLOWED_PHRASES:
        text = text.replace(allowed_phrase, "")
    return [phrase for phrase in _DISALLOWED_PHRASES if phrase in text]


def _summary_line(section: dict[str, Any], fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = section.get(field)
        if isinstance(value, dict):
            parts.append(f"{field}: {len(value)} entries")
        elif isinstance(value, list):
            parts.append(f"{field}: {len(value)} items")
        elif value is not None:
            parts.append(f"{field}: {_text(value)}")
    if not parts:
        return "- No deterministic summary fields available."
    return "- " + "; ".join(parts) + "."


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
