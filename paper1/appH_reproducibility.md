# Appendix H — Reproducibility

This appendix documents the artifacts, software stack, and validation infrastructure required to
reproduce the results in this paper from publicly available data.

---

## H.1 Repository structure

The paper is produced from a single git repository (`next-gen-aiam`) with the following
top-level layout relevant to reproduction:

```
paper1/          — manuscript source (sec*.md, app*.md, glossary.md)
src/aiam/        — strategy implementations and evaluation harness
  data/          — Panel, EODHD client, returns, regimes (incl. split.py)
  estimators/    — sample_cov, ledoit_wolf_cov, oas_cov
  evaluation/    — performance_stats, transaction_costs, ic.py
  harness/       — run_horse_race (walk-forward dispatcher)
  strategy/      — Strategy ABC + 31 base strategy classes
scripts/         — canonical build pipeline (11 scripts + generate_figures.py)
data/published/  — tracked reproduction artifacts (see below)
docs/figures/    — 11 PNGs and SVGs regenerable from generate_figures.py
notebooks/
  99_reproducibility_guard.ipynb  — automated check suite
```

All strategy performance numbers in the paper are derived from the build pipeline in `scripts/`
applied to EODHD daily prices. Raw prices and intermediate cache files are not tracked; the
published data artifacts are the canonical source for reproduction without an EODHD API key.

---

## H.2 Published data artifacts

The `data/published/` directory contains five tracked artifacts:

| File | Description |
|---|---|
| `master_table_62strategies.csv` | Full-sample and OOS statistics for all 62 strategies |
| `strategy_returns_base.parquet` | Daily returns for 31 base strategies, 2003-01-03 to 2026-04-30 |
| `strategy_returns_vmp.parquet` | Daily returns for 31 VMP-overlay strategies, same period |
| `ohlcv_29assets_2003_2026.csv` | Daily OHLCV prices for 29 instruments (source data) |
| `regime_signals.parquet` | Monthly regime classifications (R0–R7) for 2003–2026 |

All paper figures and tables are reproducible from these five files. Sub-period Sharpe
calculations (Appendix G, §6.3) use `strategy_returns_base.parquet` and
`strategy_returns_vmp.parquet` directly; `master_table_62strategies.csv` is the canonical
reference for the full-sample statistics in Appendix C.

---

## H.3 The reproducibility guard — Notebook 99

`notebooks/99_reproducibility_guard.ipynb` is an automated check suite that loads
`master_table_62strategies.csv` and verifies 20 numerical claims from the paper:

- **Full-sample Sharpe spot checks** (8 checks): GMV(sample) degenerate Sharpe, EW, MDP(LW),
  VMP(MDP(LW)), MSR(LW), VMP(MSR(LW)), SWITCH(v2a), VMP(SWITCH(v2a)) — each checked to
  two decimal places against the published CSV.
- **OOS Sharpe spot checks** (3 checks): VMP(MDP(LW)) 2.422, VMP(MDP(sample)) 2.416,
  MDP(LW) 2.304 — test period 2023–2026.
- **VMP universal sign-test** (1 check): all 24 base–VMP pairs show positive $\Delta$Sharpe
  on both the full sample and the OOS test period; count must equal 24.
- **SWITCH(v2a) vs SWITCH(LW)** (2 checks): base Sharpe differential (+0.434) and OOS
  differential (+0.104).
- **Memmel test values** (2 checks): SWITCH(v2a) vs SWITCH(LW) $z=2.05$, $p=0.040$;
  MSR(LW) vs MSR(sample) $z=1.13$, $p=0.259$.
- **Transaction cost checks** (4 checks): VMP(MDP(LW)) net 10 bps Sharpe 1.336;
  FF3-Mom net 10 bps Sharpe 0.394; VMP(SWITCH(LW)) net 10 bps Sharpe 1.201;
  FF3-Mom gross-to-net degradation 0.291.

The notebook currently passes **20/20** checks. It is the first executable to run after any
change to the build pipeline or published data.

---

## H.4 The `research_agents` framework

`src/aiam/research_agents/` implements the agentic governance substrate described in Paper 3.
For this paper its role is methodological: it provides the deterministic validation layer that
ensures reproducibility under multi-agent orchestration.

Key modules:

| Module | Purpose |
|---|---|
| `artifacts.py` | Artifact registry — typed containers for strategy results, figures, and tables |
| `manifest.py` | Manifest schema — provenance tracking for all published outputs |
| `validators.py` | Claim-level validators — the same checks run in Notebook 99, callable from Python |
| `packets.py` | Agent communication schema — structured result packets for agent-to-agent handoffs |
| `summaries.py` | Summary generators — narrative synthesis from structured result dicts |

The framework enforces a governance-first, deterministic-core-then-orchestration pattern:
numerical results are computed deterministically from the canonical pipeline; the agentic layer
(summarisation, narrative generation, view synthesis in Paper 2) operates only on already-
committed outputs. This separation ensures that LLM non-determinism cannot enter the
quantitative claims.

---

## H.5 Software stack and random seeds

**Python version:** 3.12. **Platform:** Apple Silicon (M4 Mac, darwin); results are
expected to be portable across platforms for the classical strategies (no GPU dependency).
ML/DL strategies (Paper 2) require `libomp` on macOS for XGBoost and PyTorch for deep
learning architectures.

**Key package versions (as of submission):**

| Package | Version | Role |
|---|---|---|
| numpy | 2.4.4 | Numerical core |
| pandas | 3.0.3 | Data wrangling and panel management |
| scipy | 1.17.1 | Optimization (SLSQP for risk parity), statistical tests |
| scikit-learn | 1.8.0 | OAS covariance estimator, Ledoit-Wolf |
| cvxpy | 1.8.2 | Convex optimization (MSR, MDP, constrained MV) |
| xgboost | 3.2.0 | Gradient boosting (ML track, Paper 2) |
| torch | 2.12.0 | Deep learning (DL/RL tracks, Paper 2) |

The full dependency list is in `requirements.txt`; the installable package spec is in
`pyproject.toml`.

**Random seeds.** All strategies in the 62-strategy classical comparison are deterministic:
no random number generation is used. The VMP overlay, regime classifier, and all
covariance-based optimizers are fully deterministic given the input price series. Sub-period
Sharpe ratios and Memmel test statistics are analytically derived from the return series with
no sampling step.

**Test suite.** `tests/` contains 481 pytest tests covering strategy implementations,
estimators, harness logic, evaluation metrics, and the regime pipeline. All 481 tests pass
on the submission codebase. Run with `pytest tests/` from the repository root with the
virtual environment activated.

---

## H.6 Rebuilding from prices

To regenerate all results from the EODHD price data (requires `EODHD_API_KEY` environment
variable):

```bash
python scripts/build_ohlcv_29.py
python scripts/build_returns_29.py
python scripts/build_regime_signals_2003.py
python scripts/build_all_strategies_29.py
python scripts/build_weights_cache.py
python scripts/build_switch_v2a_weights.py
python scripts/generate_figures.py    # produces 11 PNGs in docs/figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/99_reproducibility_guard.ipynb
```

Running `99_reproducibility_guard.ipynb` after the build confirms all 20 numerical checks
pass before any paper revision is committed.
