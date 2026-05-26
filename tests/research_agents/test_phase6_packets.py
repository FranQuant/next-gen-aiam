from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

from aiam.research_agents.artifacts import load_artifact_registry
from aiam.research_agents.packets import (
    REQUIRED_SECTIONS,
    build_research_packet,
    render_research_packet_markdown,
    validate_research_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKETS_SOURCE = REPO_ROOT / "src/aiam/research_agents/packets.py"


def _registry():
    return load_artifact_registry(repo_root=REPO_ROOT)


def _packet():
    return build_research_packet(registry=_registry())


def test_packet_builds_successfully_with_required_top_level_keys():
    packet = _packet()

    assert packet["packet_id"]
    assert packet["created_at_utc"]
    assert packet["packet_type"] == "phase6_research_committee_packet"
    assert packet["research_only"] is True
    assert packet["human_review_required"] is True
    assert packet["no_investment_advice"] is True
    assert isinstance(packet["sections"], dict)


def test_packet_includes_all_required_sections():
    packet = _packet()

    assert set(REQUIRED_SECTIONS).issubset(packet["sections"])


def test_packet_validates_successfully():
    result = validate_research_packet(_packet())

    assert result.ok is True
    assert result.errors == []
    assert result.section_count == len(REQUIRED_SECTIONS)


def test_packet_includes_artifacts_used_by_section():
    packet = _packet()
    inventory = packet["sections"]["evidence_inventory"]

    assert inventory["approved_artifact_count"] == 75
    assert inventory["artifacts_used_by_section"]
    assert inventory["artifacts_used_by_section"]["backtest_metrics"]
    assert inventory["artifacts_used_by_section"]["regime_signal"]
    assert inventory["artifacts_used_by_section"]["pca_dashboard"]
    assert inventory["artifacts_used_by_section"]["historical_llm_experiment_context"]


def test_packet_includes_aggregated_caveats_and_human_review_checklist():
    packet = _packet()
    caveat_text = " ".join(packet["sections"]["caveats"]).lower()
    checklist_text = " ".join(packet["sections"]["human_review_checklist"]).lower()

    assert "backtest and experiment metrics" in caveat_text
    assert "regime labels are research features" in caveat_text
    assert "pca loadings" in caveat_text
    assert "historical weight artifacts" in caveat_text
    assert "research-only output" in caveat_text
    assert "requires human review" in caveat_text
    assert "not target allocations" in caveat_text
    assert "no trading or portfolio recommendation is produced" in caveat_text
    assert "verify artifact provenance" in checklist_text
    assert "verify no recommendation language" in checklist_text


def test_markdown_renderer_includes_required_headings():
    markdown = render_research_packet_markdown(_packet())

    for heading in [
        "# Phase 6 Research Committee Packet",
        "## Run Context",
        "## Evidence Inventory",
        "## Backtest Metrics",
        "## Regime Signal",
        "## PCA Dashboard",
        "## Historical LLM Experiment Context",
        "## Caveats",
        "## Human Review Checklist",
    ]:
        assert heading in markdown


def test_markdown_renderer_avoids_actionable_recommendation_language():
    markdown = render_research_packet_markdown(_packet()).lower()

    assert "buy" not in markdown
    assert "sell" not in markdown
    assert "trade now" not in markdown
    assert "recommended allocation" not in markdown


def test_validator_fails_if_required_section_is_missing():
    packet = _packet()
    del packet["sections"]["regime_signal"]

    result = validate_research_packet(packet)

    assert result.ok is False
    assert "missing required section: regime_signal" in result.errors


def test_validator_fails_if_research_only_is_false():
    packet = _packet()
    packet["research_only"] = False

    result = validate_research_packet(packet)

    assert result.ok is False
    assert "research_only must be true" in result.errors


def test_validator_fails_if_recommendation_language_is_inserted():
    packet = deepcopy(_packet())
    packet["sections"]["caveats"].append("This is a recommended allocation.")

    result = validate_research_packet(packet)

    assert result.ok is False
    assert "forbidden language present: recommended allocation" in result.errors


def test_public_functions_do_not_expose_arbitrary_path_parameters():
    for public_function in [
        build_research_packet,
        validate_research_packet,
        render_research_packet_markdown,
    ]:
        signature = inspect.signature(public_function)
        assert "path" not in signature.parameters
        assert "manifest_path" not in signature.parameters
        assert "repo_root" not in signature.parameters

    assert list(inspect.signature(build_research_packet).parameters) == [
        "registry",
        "packet_id",
        "created_at_utc",
    ]


def test_packets_module_security_forbidden_terms_absent():
    source = PACKETS_SOURCE.read_text(encoding="utf-8").lower()
    forbidden_terms = [
        "dotenv",
        "os.getenv",
        "os.environ",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "openai",
        "anthropic",
        "tavily",
        "broker",
        "order",
        "trade",
        "api_key",
        "secret",
        "token",
        "password",
        "credential",
    ]

    for term in forbidden_terms:
        assert term not in source
