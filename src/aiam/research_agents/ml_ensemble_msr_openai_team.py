"""OpenAI Agents SDK wrapper for the ML Ensemble MSR research team.

This module stays importable without the optional SDK installed. Deterministic
artifact helpers remain the source of truth for every metric and table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiam.research_agents.ml_ensemble_msr_artifacts import (
    generate_handoff_figures,
    headline_metrics,
    render_research_handoff,
    summarize_artifact_inventory,
    summarize_predictions_artifact,
    summarize_strategy_returns_artifact,
    summarize_weights_artifact,
    validate_ml_ensemble_msr_artifact_contract,
    validate_research_handoff,
)

OPTIONAL_SDK_ERROR = (
    "Optional OpenAI Agents SDK dependencies are not installed. "
    "Install with: python -m pip install -e '.[agents]'"
)

COMMON_AGENT_CONSTRAINTS = """
Constraints:
- research-only.
- no investment advice.
- no target allocations.
- no trading recommendations.
- no live API calls.
- no environment-file access.
- no arbitrary file reads.
- use only deterministic tool outputs.
- do not change features, targets, model hyperparameters, optimizer rules, weights, or metrics.
- historical weights are not target allocations.
- human review required.
"""

RESEARCH_MANAGER_INSTRUCTIONS = (
    "You are the Research Manager Agent for the ML Ensemble MSR research team. "
    "Coordinate the workflow, validate the artifact contract, ask specialist "
    "agents for review, and ensure the final handoff is grounded in bounded "
    "deterministic artifacts.\n"
    + COMMON_AGENT_CONSTRAINTS
)

DATA_QA_INSTRUCTIONS = (
    "You are the Data QA Agent. Review run manifest evidence, date range, "
    "universe size, feature columns, local-cache assumptions, and artifact "
    "contract status. Flag data caveats for human review.\n"
    + COMMON_AGENT_CONSTRAINTS
)

QUANT_STRATEGY_INSTRUCTIONS = (
    "You are the Quant Strategy Agent. Review Lasso, Random Forest, and XGBoost "
    "ensemble mechanics, target horizon, feature set, Sharpe convention, "
    "single-fit setup, and Notebook 03 reproduction caveats. Critique only; "
    "do not alter deterministic choices.\n"
    + COMMON_AGENT_CONSTRAINTS
)

PORTFOLIO_RISK_INSTRUCTIONS = (
    "You are the Portfolio Risk Agent. Review turnover, concentration, drawdown, "
    "volatility, max single-asset weight, top-5 concentration, and the absence "
    "of transaction-cost modeling. Treat historical weights as evidence only.\n"
    + COMMON_AGENT_CONSTRAINTS
)

RESEARCH_HANDOFF_INSTRUCTIONS = (
    "You are the Research Handoff Agent. Produce a clean Markdown research memo "
    "with useful tables, required caveats, and a human-review checklist. Do not "
    "dump raw JSON and do not use recommendation language.\n"
    + COMMON_AGENT_CONSTRAINTS
)

DEFAULT_TEAM_PROMPT = """
Run the five-agent ML Ensemble MSR research handoff workflow.

Use the deterministic tools to validate artifacts, load bounded summaries, and
render the deterministic markdown handoff. Specialist agents should critique the
bounded evidence only. The final output must be clean Markdown, research-only,
no investment advice, no target allocations, no trading recommendations, and
human review required.
"""


def validate_artifact_contract_json(output_dir: str | Path) -> str:
    """Return JSON for the deterministic artifact contract validation."""
    result = validate_ml_ensemble_msr_artifact_contract(output_dir)
    return json.dumps(result, sort_keys=True)


def build_handoff_markdown(output_dir: str | Path, write_figures: bool = False) -> str:
    """Return deterministic handoff Markdown from local artifacts."""
    if write_figures:
        generate_handoff_figures(output_dir)
    markdown = render_research_handoff(output_dir)
    validation = validate_research_handoff(markdown)
    if not validation["ok"]:
        raise RuntimeError(json.dumps({"handoff_validation": validation}, sort_keys=True))
    return markdown


def build_handoff_summary_json(output_dir: str | Path, write_figures: bool = False) -> str:
    """Return bounded JSON summary suitable for agent tool output."""
    contract = validate_ml_ensemble_msr_artifact_contract(output_dir)
    inventory = summarize_artifact_inventory(output_dir)
    figure_generation = (
        generate_handoff_figures(output_dir)
        if write_figures
        else {
            "ok": None,
            "figure_dir": str(Path(output_dir) / "figures"),
            "figures": [],
            "errors": [],
            "warnings": ["not requested"],
        }
    )
    if contract["ok"]:
        markdown = render_research_handoff(output_dir)
        handoff = validate_research_handoff(markdown)
        metrics = headline_metrics(output_dir)
        predictions = summarize_predictions_artifact(output_dir)
        weights = summarize_weights_artifact(output_dir)
        strategy_returns = summarize_strategy_returns_artifact(output_dir)
    else:
        handoff = {
            "ok": False,
            "missing_sections": [],
            "errors": [],
            "warnings": ["not evaluated because artifact contract failed"],
        }
        metrics = {}
        predictions = {}
        weights = {}
        strategy_returns = {}

    payload = {
        "ok": bool(contract["ok"] and handoff["ok"]),
        "contract_validation": contract,
        "handoff_validation": handoff,
        "artifact_inventory": inventory,
        "headline_metrics": metrics,
        "bounded_artifact_summaries": {
            "predictions": predictions,
            "weights": weights,
            "strategy_returns": strategy_returns,
        },
        "figure_generation": figure_generation,
    }
    return json.dumps(payload, sort_keys=True)


def validate_artifact_contract(output_dir: str) -> str:
    """Safe tool: validate only the expected ML Ensemble MSR artifact contract."""
    return validate_artifact_contract_json(output_dir)


def load_handoff_summary(output_dir: str, write_figures: bool = False) -> str:
    """Safe tool: return bounded summaries without raw parquet rows."""
    return build_handoff_summary_json(output_dir, write_figures=write_figures)


def render_handoff_markdown(output_dir: str, write_figures: bool = False) -> str:
    """Safe tool: render deterministic Markdown from the expected artifacts."""
    return build_handoff_markdown(output_dir, write_figures=write_figures)


def generate_figures(output_dir: str) -> str:
    """Safe tool: generate deterministic figures from expected artifacts."""
    return json.dumps(generate_handoff_figures(output_dir), sort_keys=True)


def build_ml_ensemble_msr_openai_team(model_id: str) -> Any:
    """Build the five-agent SDK team, returning the manager as the entry agent."""
    if not model_id:
        raise ValueError("model_id is required for a live team run")

    sdk = _load_agents_sdk()
    Agent = sdk["Agent"]
    function_tool = sdk["function_tool"]

    tools = [
        function_tool(validate_artifact_contract),
        function_tool(load_handoff_summary),
        function_tool(render_handoff_markdown),
        function_tool(generate_figures),
    ]

    data_qa = _agent(
        Agent,
        name="Data QA Agent",
        model=model_id,
        instructions=DATA_QA_INSTRUCTIONS,
        tools=tools,
    )
    quant_strategy = _agent(
        Agent,
        name="Quant Strategy Agent",
        model=model_id,
        instructions=QUANT_STRATEGY_INSTRUCTIONS,
        tools=tools,
    )
    portfolio_risk = _agent(
        Agent,
        name="Portfolio Risk Agent",
        model=model_id,
        instructions=PORTFOLIO_RISK_INSTRUCTIONS,
        tools=tools,
    )
    research_handoff = _agent(
        Agent,
        name="Research Handoff Agent",
        model=model_id,
        instructions=RESEARCH_HANDOFF_INSTRUCTIONS,
        tools=tools,
    )
    return _agent(
        Agent,
        name="Research Manager Agent",
        model=model_id,
        instructions=RESEARCH_MANAGER_INSTRUCTIONS,
        tools=tools,
        handoffs=[data_qa, quant_strategy, portfolio_risk, research_handoff],
    )


def run_ml_ensemble_msr_openai_team(
    output_dir: str | Path,
    model_id: str,
    prompt: str | None = None,
    write_figures: bool = False,
) -> str:
    """Run the live SDK team and return its final text output."""
    if not model_id:
        raise ValueError("model_id is required for a live team run")

    contract = json.loads(validate_artifact_contract_json(output_dir))
    if not contract["ok"]:
        raise RuntimeError(json.dumps({"contract_validation": contract}, sort_keys=True))
    if write_figures:
        generate_handoff_figures(output_dir)

    sdk = _load_agents_sdk()
    manager = build_ml_ensemble_msr_openai_team(model_id)
    runner_prompt = _team_prompt(output_dir, prompt)
    Runner = sdk["Runner"]

    if hasattr(Runner, "run_sync"):
        result = Runner.run_sync(manager, runner_prompt)
    else:
        import asyncio

        result = asyncio.run(Runner.run(manager, runner_prompt))
    return _final_output_text(result)


def _load_agents_sdk() -> dict[str, Any]:
    try:
        from agents import Agent, Runner, function_tool
    except ImportError as exc:
        raise RuntimeError(OPTIONAL_SDK_ERROR) from exc
    return {"Agent": Agent, "Runner": Runner, "function_tool": function_tool}


def _agent(Agent: Any, **kwargs: Any) -> Any:
    try:
        return Agent(**kwargs)
    except TypeError:
        trimmed = dict(kwargs)
        trimmed.pop("handoffs", None)
        return Agent(**trimmed)


def _team_prompt(output_dir: str | Path, prompt: str | None) -> str:
    requested_prompt = prompt or DEFAULT_TEAM_PROMPT
    return (
        f"{requested_prompt}\n\n"
        f"Artifact directory: {Path(output_dir)}\n"
        "Use validate_artifact_contract first, then load_handoff_summary, then "
        "render_handoff_markdown for the final memo."
    )


def _final_output_text(result: Any) -> str:
    for attr in ("final_output", "output", "content"):
        value = getattr(result, attr, None)
        if value is not None:
            return str(value)
    return str(result)

