#!/usr/bin/env python
"""Build a deterministic ML Ensemble MSR research handoff from local artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aiam.research_agents.ml_ensemble_msr_artifacts import (  # noqa: E402
    generate_handoff_figures,
    headline_metrics,
    render_research_handoff,
    summarize_artifact_inventory,
    validate_ml_ensemble_msr_artifact_contract,
    validate_research_handoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML Ensemble MSR research handoff.")
    parser.add_argument(
        "--output-dir",
        default="results/ml_ensemble_msr",
        help="Local ML Ensemble MSR artifact directory.",
    )
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--print-summary-json", action="store_true")
    parser.add_argument("--write-figures", action="store_true")
    return parser.parse_args()


def build_summary(output_dir: str | Path, *, write_figures: bool = False) -> dict:
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
    else:
        handoff = {
            "ok": False,
            "missing_sections": [],
            "errors": [],
            "warnings": ["not evaluated because artifact contract failed"],
        }
    metrics = headline_metrics(output_dir) if contract["ok"] else {}
    ok = bool(contract["ok"] and handoff["ok"])
    return {
        "ok": ok,
        "contract_validation": contract,
        "handoff_validation": handoff,
        "artifact_count": inventory["artifact_count"],
        "headline_metrics": metrics,
        "figure_generation": figure_generation,
    }


def main() -> None:
    args = parse_args()
    if args.print_markdown:
        if args.write_figures:
            generate_handoff_figures(args.output_dir)
        markdown = render_research_handoff(args.output_dir)
        validation = validate_research_handoff(markdown)
        if not validation["ok"]:
            raise SystemExit(json.dumps({"handoff_validation": validation}, sort_keys=True))
        print(markdown)
        return

    summary = build_summary(args.output_dir, write_figures=args.write_figures)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
