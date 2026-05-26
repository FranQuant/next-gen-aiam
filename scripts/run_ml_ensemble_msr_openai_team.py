#!/usr/bin/env python
"""CLI for the ML Ensemble MSR OpenAI Agents SDK research team."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aiam.research_agents.ml_ensemble_msr_openai_team import (  # noqa: E402
    build_handoff_markdown,
    build_handoff_summary_json,
    run_ml_ensemble_msr_openai_team,
    validate_artifact_contract_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML Ensemble MSR research team.")
    parser.add_argument(
        "--output-dir",
        default="results/ml_ensemble_msr",
        help="Local ML Ensemble MSR artifact directory.",
    )
    parser.add_argument("--print-summary-json", action="store_true")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--write-figures", action="store_true")
    parser.add_argument("--run-team", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--prompt")
    parser.add_argument("--write-handoff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.run_team:
        if not args.model_id:
            raise SystemExit("--model-id is required with --run-team")
        _print_team_progress()
        try:
            final_output = run_ml_ensemble_msr_openai_team(
                output_dir=output_dir,
                model_id=args.model_id,
                prompt=args.prompt,
                write_figures=args.write_figures,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.write_handoff:
            (output_dir / "research_team_handoff.md").write_text(
                final_output,
                encoding="utf-8",
            )
        print(final_output)
        return

    contract = json.loads(validate_artifact_contract_json(output_dir))
    if not contract["ok"]:
        print(build_handoff_summary_json(output_dir, write_figures=False))
        return

    if args.print_markdown:
        markdown = build_handoff_markdown(output_dir, write_figures=args.write_figures)
        if args.write_handoff:
            (output_dir / "research_team_handoff.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        return

    if args.write_handoff:
        markdown = build_handoff_markdown(output_dir, write_figures=args.write_figures)
        (output_dir / "research_team_handoff.md").write_text(markdown, encoding="utf-8")

    print(build_handoff_summary_json(output_dir, write_figures=args.write_figures))


def _print_team_progress() -> None:
    print("[Research Manager] Starting artifact validation...", file=sys.stderr)
    print("[Data QA Agent] Reviewing manifest...", file=sys.stderr)
    print("[Quant Strategy Agent] Reviewing strategy mechanics...", file=sys.stderr)
    print("[Portfolio Risk Agent] Reviewing risk diagnostics...", file=sys.stderr)
    print("[Research Handoff Agent] Producing final memo...", file=sys.stderr)


if __name__ == "__main__":
    main()

