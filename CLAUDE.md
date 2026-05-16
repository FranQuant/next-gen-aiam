# CLAUDE.md

## Project
next-gen-aiam — a comparative model harness for AI-driven asset management. Common data panel (EODHD), uniform `Strategy` interface, side-by-side comparison of classical methods, regime models, ML, DL, and RL on the same 29-asset 2003-2026 universe.

## Status
Session 1 closed (May 2026). The static-baselines deliverable is complete:
- Paper (38 pages, 11 figures) at `docs/results.pdf` — 62 strategies on 29-asset 2003-2026 universe
- Reproducibility notebooks: `01_paper_reproduction.ipynb` + `02_practitioner_analytics.ipynb`
- Production pipeline in `scripts/` and `src/aiam/`
- Published datasets in `data/published/` (three-level reproduction: master table / strategy returns / OHLCV input)
- Test suite covering 11 components

Next: Session 1.5B (feature engineering for ML strategies).

## Environment

    cd ~/Projects/next-gen-aiam
    source .venv/bin/activate

Python: 3.12 (M4 Mac).

    pip install -r requirements.txt
    pip install -e ".[dev]"

## Architecture (locked decisions)

**Strategy ABC.** Two base classes in `src/aiam/strategy/` (singular package name):
- `PointInTimeStrategy.predict_weights(panel, asof) → weights` — memoryless; covers classical, ML, DL strategies that produce weights as a function of past data alone.
- `SequentialStrategy.step(observation, ...) → action` — stateful; covers RL and LLM agents.

**Panel** — `src/aiam/data/panel.py`. Thin wrapper around a dict of DataFrames (prices, returns, regimes, macro). Frozen via `MappingProxyType` after construction. Single access method: `panel.slice(asof, kind, lookback=None, freq=None, fill=None)` is the no-look-ahead gatekeeper.

**Estimators** injected as Callables (`sample_cov`, `ledoit_wolf_cov`, `oas_cov` from `src/aiam/estimators/`).

**Train/test split** owned by the harness, not the Panel. Constants in `src/aiam/data/split.py`:
- `TRAIN_END = "2022-12-31"`
- `TEST_START = "2023-01-01"`

**Regime engine** is a data product (precomputed parquet at `data/cache/regime_signals_2003_2026.parquet`, published at `data/published/regime_signals.parquet`). Monthly classifications (forward-filled to daily in `SwitchingStrategy`). 8 indicators (GDP, CPI, UNEM, YC_10Y, YC_2Y, YC_STEP, VIX, SPX), 8 regimes (0-7), feature pipeline: level + first-diff + second-diff per indicator. SWITCH(v2a) routing: R0→MSR(LW), R5→MSR(sample), default→MDP(LW). Macro data indexed by publication date (no look-ahead).

**VMP overlay defaults** (volatility-managed portfolio, Moreira & Muir 2017):
- Lookback: 21 trading days
- Lag: 1 day
- Clip: (0.25, 1.5)
- Target: each strategy's long-run realized volatility

## Repository structure

    ├── CLAUDE.md
    ├── README.md
    ├── pyproject.toml          # installable package
    ├── requirements.txt
    ├── docs/
    │   ├── architecture.md     # design rationale
    │   ├── results.md          # paper source (markdown)
    │   ├── results.pdf         # paper PDF (38 pages, 11 figures)
    │   ├── refs.bib            # bibliography
    │   └── figures/            # 11 PNGs + SVGs (regenerable from generate_figures.py)
    ├── src/aiam/
    │   ├── data/               # Panel, EODHD client, returns, regimes (incl. split.py)
    │   ├── estimators/         # sample_cov, ledoit_wolf_cov, oas_cov
    │   ├── evaluation/         # performance_stats, transaction_costs, switch_assembly
    │   ├── harness/            # run_horse_race (single-strategy, dynamic dispatch)
    │   └── strategy/           # Strategy ABC + 31 base strategy classes
    ├── scripts/                # canonical pipeline (11 build scripts + generate_figures.py)
    ├── notebooks/
    │   ├── _shared.py          # common setup, FAMILY_MAP, data loaders, regime helpers
    │   ├── 01_paper_reproduction.ipynb       # §3.1 master table, rankings, TC, §7.4 sub-period, reproducibility
    │   ├── 02_practitioner_analytics.ipynb   # §2.1-2.15 practitioner views + Part 3 OOS
    │   └── _archived_01_static_baselines_pre_split.ipynb  # pre-split monolith (preserved in git history)
    ├── data/
    │   ├── published/          # tracked: canonical reproduction artifacts (see data/published/README.md)
    │   ├── cache/              # gitignored: pipeline outputs, regenerable
    │   └── raw/                # gitignored: EODHD pulls, regenerable
    └── tests/                  # 11 test files (pytest)

## Strategy inventory (31 base + 31 VMP = 62 total)

| Family | Strategies | Count |
|---|---|---|
| Classical MV | EW, GMV(sample/LW/OAS), MSR(sample/LW) | 6 |
| Constrained MV | MSR_C, MVO_C × {sample, LW} | 4 |
| Diversification | MDP, RP, HRP × {sample, LW} | 6 |
| Regime Switch | SWITCH(sample), SWITCH(LW) | 2 |
| TS Momentum | TSMOM(12m), TSMOM(6m) | 2 |
| Black-Litterman | BL-Eq × {sample, LW}, BL-Mom(LW), BL-Rev(LW) | 4 |
| Factor | FF3-Mom, FF3-LowVol, FF3-Quality, FF3-Multi | 4 |
| Long-Short | TSMOM-LS(12m), BL-Mom-LS(LW), FF3-Mom-LS | 3 |

Each base strategy is paired with a VMP-overlay variant.

## Canonical commands

**Build the full pipeline from prices:**

    python scripts/build_ohlcv_29.py
    python scripts/build_returns_29.py
    python scripts/build_regime_signals_2003.py
    python scripts/build_all_strategies_29.py
    python scripts/build_weights_cache.py
    python scripts/build_switch_v2a_weights.py

**Regenerate paper figures:**

    python scripts/generate_figures.py    # produces 11 PNGs in docs/figures/

**Render paper PDF (canonical pandoc command):**

    pandoc docs/results.md -o docs/results.pdf \
      --pdf-engine=xelatex \
      --citeproc \
      --bibliography=docs/refs.bib \
      --metadata title="Comparative Asset Allocation Harness: A 62-Strategy Walk-Forward Study" \
      --metadata author="Francisco Salazar" \
      --metadata date="May 2026" \
      --variable geometry:margin=1in \
      --variable fontsize=11pt \
      --variable mainfont="Times New Roman" \
      --variable colorlinks=true \
      --variable linkcolor=NavyBlue \
      --toc --toc-depth=2 \
      --number-sections \
      --resource-path=docs \
      --include-in-header=<(echo '\usepackage{pdflscape}'; echo '\usepackage{longtable}'; echo '\usepackage{multirow}'; echo '\usepackage{booktabs}'; echo '\raggedbottom')

**Execute notebooks:**

    jupyter nbconvert --to notebook --execute --inplace notebooks/01_paper_reproduction.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_practitioner_analytics.ipynb

## Claude Code prompt conventions

When firing Claude Code prompts in this repo, use this preamble:

> Execution mode: ship without checking in. Do not ask clarifying questions. Do not request approval at intermediate checkpoints. If a step fails, debug and continue. Report only at the end. Single atomic commit.
>
> Concision discipline: code <40 lines per function, vectorized pandas, no defensive scaffolding. Prose: 2-4 tight sentences, no throat-clearing.

Every Claude Code prompt should include:
- A **self-verification block** before commit (concrete checks against rendered output, e.g., `pdftotext docs/results.pdf - | grep -c "X"`; verify the *output*, not just the source)
- A **single atomic commit** with a structured message (subject line + bullet body)
- A **final report** with actual values from the verification block

## Non-negotiable rules
- Never commit secrets (EODHD API key reads from `EODHD_API_KEY` environment variable only)
- Never commit `data/cache/` or `data/raw/` (gitignored). Distribute via `data/published/`.
- Strategy returns and figures are products of the pipeline — don't hand-edit them; rebuild from scripts.
- Paper claims must be reproducible from `data/published/`. If a paper revision changes a number, the published data + figures must update in the same commit.

## Reference repos
- `~/Projects/ai_asset_management_lab` (paam_lab) — local source repo for prior-iteration code. Most useful code has been ported forward.
- `github.com/yhilpisch/paamcode` — Hilpisch book's official code reference.
- Hilpisch books in project knowledge: `pyaiam.pdf` (AIAM, forthcoming a), `hilpisch_excerpts*.pdf` (Python for Finance 3rd ed, forthcoming b).

## Roadmap

**Session 1 (done) — static baselines paper.** 62 strategies on 29-asset 2003-2026 universe. Paper at commit `60b6830`. Notebooks split (commits `efda396` → `e9bc679` → `60b6830`). Repo cleanup pass A (`df0e490`) + pass B (`e4b07ba`). Dataset publication (`bdd8636`).

**Session 1.5B (next) — feature engineering.** `src/aiam/features/technical.py` mirroring Hilpisch §19.2 SignalEngine API: `momentum(returns, lookback)`, `volatility(returns, lookback)`, `forward_returns(returns, horizon)`, `zscore(series, window)`, `information_coefficient(signal, forward_ret)`. OHLCV-based features: RSI, ATR, Bollinger, gap, volume. Plus `src/aiam/evaluation/ic.py` (IC diagnostic) and `src/aiam/strategy/signal_tilt.py` (SignalTilt baseline). Forward-return target: `rets.shift(-h).rolling(h).sum()`. ETA: 3-4h agent work.

**Session 2 — ML strategies.** Lasso, Random Forest, XGBoost expected-return estimation. `TimeSeriesSplit` cross-validation, no-look-ahead. New notebook `notebooks/03_ml_strategies.ipynb`. ETA: 6-10h.

**Session 3 — DL strategies.** MLP, LSTM, Transformer sequence models. `notebooks/04_dl_strategies.ipynb`.

**Session 4 — RL strategies.** PPO, SAC agents via `SequentialStrategy.step()`. `notebooks/05_rl_strategies.ipynb`.

## Known issues (defer to Session 1.5B)

- `src/aiam/harness/horse_race.py` line 12: `_PRICES_CACHE` points to `prices_30.parquet` but the 29-asset pipeline uses `prices_29.parquet`. No correctness impact (cache regenerates if absent), but should be updated.
- `scripts/build_all_strategies_29.py` lines 54-67: SWITCH(sample)/SWITCH(LW) canonical routing uses R0→EW (v1-style baseline), distinct from SWITCH(v2a)'s R0→MSR(LW). Add an inline comment clarifying this distinction.
- `scripts/generate_figures.py` module-level docstring: verify it says "Generate 11 publication-quality figures" (was "Generate 4" originally, updated in pass-7 but worth re-confirming).
