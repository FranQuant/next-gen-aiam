"""OpenAI Agents SDK wrapper for the ML Ensemble MSR research team.

This module stays importable without the optional SDK installed. Deterministic
artifact helpers remain the source of truth for every metric and table.
"""
from __future__ import annotations

import json
import re
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

RESEARCH_MANAGER_AGENT_NAME = "research_manager_agent"
DATA_QA_AGENT_NAME = "data_qa_agent"
QUANT_STRATEGY_AGENT_NAME = "quant_strategy_agent"
PORTFOLIO_RISK_AGENT_NAME = "portfolio_risk_agent"
RESEARCH_HANDOFF_AGENT_NAME = "research_handoff_agent"
SDK_SAFE_AGENT_NAMES = (
    RESEARCH_MANAGER_AGENT_NAME,
    DATA_QA_AGENT_NAME,
    QUANT_STRATEGY_AGENT_NAME,
    PORTFOLIO_RISK_AGENT_NAME,
    RESEARCH_HANDOFF_AGENT_NAME,
)
TEAM_HANDOFF_TITLE = "# ML Ensemble MSR Research Handoff"
DETERMINISTIC_HANDOFF_APPENDIX_TITLE = "# Deterministic Rendered Handoff Memo"
FIGURE_LINKS = (
    "figures/cumulative_returns.png",
    "figures/drawdown.png",
    "figures/turnover.png",
    "figures/concentration.png",
    "figures/top_weights.png",
)
HUMAN_REVIEW_PATTERNS = (
    r"\bhuman review required\b",
    r"\bhuman review is required\b",
    r"\brequires human review\b",
)
GOVERNANCE_FOOTER = (
    "---\n\n"
    "Human review required. This memo is research-only, provides no investment "
    "advice, no target allocations, and no trading recommendations. Historical "
    "weights are not target allocations."
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
    "as the canonical GitHub-compatible Markdown deliverable. Use the deterministic "
    "handoff as evidence, but do not paste the deterministic handoff verbatim. Do "
    "not include a '# Deterministic Rendered Handoff Memo' section. Produce one "
    "top-level '# ML Ensemble MSR Research Handoff' title only. Include a "
    "'## Figures' section with relative Markdown image links when figures are "
    "available. Reference deterministic artifacts instead of appending them, and "
    "include an '## Appendix / Source Artifacts' section listing run_manifest.json, "
    "metrics.json, report.md, predictions.parquet, weights.parquet, "
    "strategy_returns.parquet, and figures/*.png. Do not dump raw JSON and do not "
    "use recommendation language. Include the exact phrase 'Human review required.'\n"
    + COMMON_AGENT_CONSTRAINTS
)

DEFAULT_TEAM_PROMPT = """
Run the five-agent ML Ensemble MSR research handoff workflow.

Use the deterministic tools to validate artifacts, load bounded summaries, and
render the deterministic markdown handoff for evidence only. Specialist agents
should critique the bounded evidence only. The final output must be the
canonical GitHub-compatible Markdown deliverable, research-only, no investment
advice, no target allocations, no trading recommendations, and human review
required. Include the exact phrase 'Human review required.' The Research Handoff
Agent must not paste the deterministic handoff verbatim, must not include a
section titled '# Deterministic Rendered Handoff Memo', and must produce exactly
one top-level '# ML Ensemble MSR Research Handoff' title. Use this structure:

# ML Ensemble MSR Research Handoff

## Executive Summary
## Five-Agent Review Summary
## Performance Metrics
## Turnover and Concentration
## Figures
## Methodology Caveats
## Open Questions for Human Review
## Human Review Checklist
## Appendix / Source Artifacts

When figures are available, include this exact Figures section:

## Figures

![Cumulative Returns](figures/cumulative_returns.png)
![Drawdown](figures/drawdown.png)
![Turnover](figures/turnover.png)
![Concentration](figures/concentration.png)
![Top Weights](figures/top_weights.png)

In Appendix / Source Artifacts, state that the deterministic source artifacts are:
- run_manifest.json
- metrics.json
- report.md
- predictions.parquet
- weights.parquet
- strategy_returns.parquet
- figures/*.png
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
        name=DATA_QA_AGENT_NAME,
        model=model_id,
        instructions=DATA_QA_INSTRUCTIONS,
        tools=tools,
    )
    quant_strategy = _agent(
        Agent,
        name=QUANT_STRATEGY_AGENT_NAME,
        model=model_id,
        instructions=QUANT_STRATEGY_INSTRUCTIONS,
        tools=tools,
    )
    portfolio_risk = _agent(
        Agent,
        name=PORTFOLIO_RISK_AGENT_NAME,
        model=model_id,
        instructions=PORTFOLIO_RISK_INSTRUCTIONS,
        tools=tools,
    )
    research_handoff = _agent(
        Agent,
        name=RESEARCH_HANDOFF_AGENT_NAME,
        model=model_id,
        instructions=RESEARCH_HANDOFF_INSTRUCTIONS,
        tools=tools,
    )
    return _agent(
        Agent,
        name=RESEARCH_MANAGER_AGENT_NAME,
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
    final_output = finalize_team_handoff_output(_final_output_text(result))
    validation = validate_team_handoff_output(
        final_output,
        require_figures=write_figures or _figures_exist(output_dir),
    )
    if not validation["ok"]:
        raise RuntimeError(json.dumps({"team_handoff_validation": validation}, sort_keys=True))
    return final_output


def finalize_team_handoff_output(markdown: str) -> str:
    """Append a concise governance footer only when required language is missing."""
    lower_markdown = markdown.lower()
    required_present = (
        "research-only" in lower_markdown
        and "no investment advice" in lower_markdown
        and "no target allocations" in lower_markdown
        and "no trading recommendations" in lower_markdown
        and "historical weights are not target allocations" in lower_markdown
        and _contains_human_review_language(markdown)
    )
    if required_present:
        return markdown
    return f"{markdown.rstrip()}\n\n{GOVERNANCE_FOOTER}"


def validate_team_handoff_output(markdown: str, require_figures: bool = False) -> dict[str, Any]:
    """Validate the live team memo did not append the deterministic handoff."""
    errors = []
    if DETERMINISTIC_HANDOFF_APPENDIX_TITLE in markdown:
        errors.append("contains deterministic rendered handoff appendix title")
    title_count = sum(line.strip() == TEAM_HANDOFF_TITLE for line in markdown.splitlines())
    if title_count != 1:
        errors.append(f"expected exactly one top-level team handoff title, found {title_count}")
    if "## Appendix / Source Artifacts" not in markdown:
        errors.append("missing Appendix / Source Artifacts section")
    lower_markdown = markdown.lower()
    if "no investment advice" not in lower_markdown:
        errors.append("missing no investment advice language")
    if not _contains_human_review_language(markdown):
        errors.append("missing human review language")
    if require_figures:
        if "## Figures" not in markdown:
            errors.append("missing Figures section")
        missing_links = [link for link in FIGURE_LINKS if link not in markdown]
        if missing_links:
            errors.append(f"missing figure links: {missing_links}")
    return {"ok": not errors, "errors": errors}


def _contains_human_review_language(markdown: str) -> bool:
    return any(re.search(pattern, markdown, flags=re.IGNORECASE) for pattern in HUMAN_REVIEW_PATTERNS)


def _figures_exist(output_dir: str | Path) -> bool:
    figure_dir = Path(output_dir) / "figures"
    return all((figure_dir / Path(link).name).is_file() for link in FIGURE_LINKS)


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
