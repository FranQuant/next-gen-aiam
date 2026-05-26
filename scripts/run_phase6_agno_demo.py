from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiam.research_agents.agno_adapter import (  # noqa: E402
    build_phase6_agno_agent,
    get_validated_phase6_packet_markdown,
)
from aiam.research_agents.packets import (  # noqa: E402
    build_research_packet,
    validate_research_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the optional Phase 6 Agno packet demo.")
    parser.add_argument("--print-packet", action="store_true", help="Print the deterministic packet markdown.")
    parser.add_argument("--run-agent", action="store_true", help="Run the optional Agno wrapper.")
    parser.add_argument("--model-id", help="Explicit model id for --run-agent.")
    args = parser.parse_args()

    if args.print_packet:
        print(get_validated_phase6_packet_markdown(), end="")
        return 0

    if args.run_agent:
        if not args.model_id:
            parser.error("--run-agent requires --model-id")
        try:
            agent = build_phase6_agno_agent(model_id=args.model_id)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        prompt = (
            "Use the deterministic packet tool once. Return a concise research-only summary "
            "that preserves every human-review boundary."
        )
        response = agent.run(prompt)
        print(_response_text(response))
        return 0

    packet = build_research_packet()
    validation = validate_research_packet(packet)
    print(json.dumps(asdict(validation), sort_keys=True))
    return 0 if validation.ok else 1


def _response_text(response) -> str:
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)


if __name__ == "__main__":
    raise SystemExit(main())
