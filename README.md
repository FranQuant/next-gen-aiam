# next-gen-aiam

Comparative model harness for AI-driven asset management. One canonical data
panel on real market data (EODHD); a uniform `Strategy` interface so classical
methods, regime-conditional models, ML, and RL all plug into the same
comparison.

**Status:** scaffolding. Architecture locked; v0 build begins next session.

## Reproducing the paper

Published datasets for reproducing the paper's results are in [`data/published/`](data/published/). Three reproduction levels are supported:

1. **Verify numbers directly** — `data/published/master_table_62strategies.csv`
2. **Reproduce metrics from returns** — `data/published/strategy_returns_*.parquet`
3. **Re-derive from prices** — `data/published/ohlcv_29assets_2003_2026.csv` + scripts

See [`data/published/README.md`](data/published/README.md) for full instructions.

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
