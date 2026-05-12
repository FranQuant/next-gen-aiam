# next-gen-aiam

Comparative model harness for AI-driven asset management. One canonical data
panel on real market data (EODHD); a uniform `Strategy` interface so classical
methods, regime-conditional models, ML, and RL all plug into the same
comparison.

**Status:** scaffolding. Architecture locked; v0 build begins next session.

## Stack

- Python 3.12
- Installable package: `aiam` (`src/aiam/`)
- Data source: EODHD

## Planned layout

    src/aiam/         # library — strategies, data, evaluation, harness
    notebooks/        # narrative comparisons
    tests/            # pytest
    data/cache/       # local parquet cache (gitignored)
    docs/             # design notes
