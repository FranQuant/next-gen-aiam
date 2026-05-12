# CLAUDE.md

## Project
next-gen-aiam — comparative model harness for AI-driven asset management.
Common data panel (EODHD), uniform `Strategy` interface, side-by-side
comparison of classical methods, regime models, ML, DL, and RL on the
same data.

## Status
Scaffolding (May 2026). Environment, package skeleton, and git in place.
Architectural commitments locked in the handoff doc; v0 build begins next
session.

## Environment

    cd ~/Projects/next-gen-aiam
    source .venv/bin/activate

Reinstall if needed:

    pip install -r requirements.txt
    pip install -e ".[dev]"

Python: 3.14.0 on this machine.

## What exists today
- `src/aiam/` — empty package (just `__init__.py` with version)
- `pyproject.toml`, `requirements.txt` — installable, dev tooling
- `.gitignore`, `README.md`

## What's planned (v0, next session)
- `src/aiam/data/` — Panel, Universe, EODHD client + per-data-type modules
- `src/aiam/strategies/` — Strategy ABC + classical, regime, overlay subpackages
- `src/aiam/evaluation/` — performance_stats with corrected Sharpe
- `src/aiam/harness.py` — run_horse_race
- `tests/` — port from old repo
- `notebooks/01_data_and_universe.ipynb`, `notebooks/05_horse_race.ipynb`

## Non-negotiable rules
- Never commit secrets (EODHD API key reads from environment variable only)
- Never commit data files (`data/cache/` is gitignored)
- `nbstripout` to be configured before notebooks land in git

## Reference repos
- Old lab: `~/Projects/ai_asset_management_lab` — source of `paam_lab` code to port forward
- Hilpisch: github.com/yhilpisch/paamcode — book's official code

## Conventions
Full notebook/code conventions land next session with the Panel and Strategy
ABC. Until then: write defensively; ask before adding top-level files or new
top-level dependencies.
