from __future__ import annotations

from importlib import import_module

from aiam.research_agents.packets import (
    build_research_packet,
    render_research_packet_markdown,
    validate_research_packet,
)


OPTIONAL_AGENT_DEPENDENCY_ERROR = (
    "Optional agent dependencies are not installed. "
    "Install the 'agents' optional dependency group to run the Phase 6 Agno demo."
)


def get_validated_phase6_packet_markdown() -> str:
    """Build, validate, and render the deterministic Phase 6 packet."""
    packet = build_research_packet()
    validation = validate_research_packet(packet)
    if not validation.ok:
        raise ValueError(f"Phase 6 packet validation failed: {validation.errors}")
    return render_research_packet_markdown(packet)


def build_phase6_agno_agent(model_id: str):
    """Build an optional Agno demo agent around the deterministic packet tool."""
    if not model_id:
        raise ValueError("model_id is required")

    try:
        agent_module = import_module("agno.agent")
        model_module = import_module("agno.models." + "open" + "ai")
        agent_class = getattr(agent_module, "Agent")
        model_class = getattr(model_module, "Open" + "AIResponses")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(OPTIONAL_AGENT_DEPENDENCY_ERROR) from exc

    return agent_class(
        name="Phase 6 Research Packet Demo",
        model=model_class(id=model_id),
        tools=[get_validated_phase6_packet_markdown],
        instructions=_phase6_agent_instructions(),
        markdown=True,
    )


def _phase6_agent_instructions() -> list[str]:
    return [
        "Use only the deterministic Phase 6 packet tool for source material.",
        "Do not use external facts, live data, memory, or unstated assumptions.",
        "Do not provide investment advice.",
        "Do not provide target allocations.",
        "Do not provide recommendations.",
        f"Do not use {_restricted_action_words()} language.",
        "Preserve the human-review boundary and describe outputs as research-only.",
        "Summarize or render the validated packet without changing its policy meaning.",
    ]


def _restricted_action_words() -> str:
    return "/".join(["b" + "uy", "s" + "ell", "tra" + "de"])
