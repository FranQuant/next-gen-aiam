from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiam.research_agents.packets import (
    build_research_packet,
    render_research_packet_markdown,
    validate_research_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the deterministic Phase 6 packet.")
    parser.add_argument("--print-markdown", action="store_true", help="Print the rendered markdown packet.")
    parser.add_argument("--output-md", type=Path, help="Optional markdown output path.")
    args = parser.parse_args()

    packet = build_research_packet()
    validation = validate_research_packet(packet)
    print(json.dumps(asdict(validation), indent=2, sort_keys=True))

    if args.print_markdown or args.output_md:
        markdown = render_research_packet_markdown(packet)
        if args.print_markdown:
            print(markdown, end="")
        if args.output_md:
            args.output_md.write_text(markdown, encoding="utf-8")

    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
