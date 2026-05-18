# CLAUDE.md

## Project
next-gen-aiam — a comparative model harness for AI-driven asset management. Common data panel (EODHD), uniform `Strategy` interface, side-by-side comparison of classical methods, regime models, ML, DL, and RL on the same 29-asset 2003-2026 universe.

## Status
Sessions 1 + 1.5B + 2 closed (May 2026). Current state at commit `edf88c9`:
- Paper (38 pages, 11 figures) at `docs/results.pdf` — 62-strategy comparative harness on 29-asset 2003-2026 universe
- Notebooks: `01_paper_reproduction.ipynb` + `02_practitioner_analytics.ipynb` + `03_ml_strategies.ipynb`
- ML scaffolding: `src/aiam/ml/`, `src/aiam/features/asset_class.py`, `src/aiam/strategy/ml_strategies.py`
- Notebook 03 fully populated: 17 sections, 9+ figures, 28-strategy extended comparison (single-fit + VMP + ensemble + walk-forward)
- Published datasets in `data/published/` (5 artifacts + README)
- Test suite: 124 tests, all passing

Next: Session 3 — DL strategies (MLP, LSTM, Transformer). Sharpe bar to clear: 2.579 from MSR(Ensemble_μ̂).

## Environment

    cd ~/Projects/next-gen-aiam
    source .venv/bin/activate

Python: 3.12 (M4 Mac).

On Apple Silicon (M4 Mac), `brew install libomp` is required before `pip install xgboost` due to OpenMP linkage.

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

**Session 1.5B (done) — feature engineering.** `src/aiam/features/technical.py` (SignalEngine + 9 functions), `src/aiam/evaluation/ic.py` (IC + ic_summary), `src/aiam/strategy/signal_tilt.py` (SignalTilt + momentum_signal_fn). 44 tests, all passing. Validated on 29-asset live data (commit `803fc00`). See Validation Findings below.

**Session 2 (done) — ML strategies.** Lasso, RF, XGBoost expected-return estimation in single-fit and walk-forward regimes. Sub-passes: 2a (scaffolding: `src/aiam/ml/`, `src/aiam/strategy/ml_strategies.py`), 2b (notebook 03, Approaches A + B), 2c-A (VMP overlay + ensemble), 2c-B (walk-forward refit, default vs val-optimal HPs), 2c-C (HP sensitivity diagnostic), 2c-D (visualization polish). Final commit `edf88c9`. Headline: MSR(Ensemble_μ̂) Sharpe 2.579, beating all classical baselines OOS.

**Session 3 — DL strategies.** MLP, LSTM, Transformer sequence models. `notebooks/04_dl_strategies.ipynb`. Sharpe bar to beat: 2.579 (MSR(Ensemble_μ̂) from Session 2). Natural DL angle: capture temporal patterns and cross-asset attention that tree-based models miss.

Session 3a scaffolding complete (commit `4077968`): `src/aiam/dl/` (workflow.py, models.py), `src/aiam/strategy/dl_strategies.py`, 35 tests (161 total). Architecture: MLPRegressor 1,121 params; LSTMRegressor 4,153 params; TransformerRegressor 42,401 params (includes learnable pos_embed of shape (1, 512, d_model), std=0.02 init). Multi-seed ensemble (default 5 seeds) baked into all strategy classes. EnsembleDLSignalStrategy at library level. OMP_NUM_THREADS=1 in conftest.py required for PyTorch + XGBoost coexistence on Apple Silicon.

**Session 4 — RL strategies.** PPO, SAC agents via `SequentialStrategy.step()`. `notebooks/05_rl_strategies.ipynb`.

## Known issues (defer to Session 1.5B)

- `src/aiam/harness/horse_race.py` line 12: `_PRICES_CACHE` points to `prices_30.parquet` but the 29-asset pipeline uses `prices_29.parquet`. No correctness impact (cache regenerates if absent), but should be updated.
- `scripts/build_all_strategies_29.py` lines 54-67: SWITCH(sample)/SWITCH(LW) canonical routing uses R0→EW (v1-style baseline), distinct from SWITCH(v2a)'s R0→MSR(LW). Add an inline comment clarifying this distinction.
- `scripts/generate_figures.py` module-level docstring: verify it says "Generate 11 publication-quality figures" (was "Generate 4" originally, updated in pass-7 but worth re-confirming).

All three resolved in commit `803fc00` (Session 1.5B).

## Validation Findings

**Session 1.5B feature library validation (29-asset 2003-2026 universe):**

- **252-day momentum**: mean IC = +0.073 (t = 15.5), hit rate 60.4%. In expected range for a diversified cross-asset universe.
- **21-day momentum**: mean IC = +0.019 (t = 4.4), near zero. Expected — short-horizon momentum is weak / reversal-prone.
- **60-day volatility**: mean IC = **+0.125** (t = 25.5), hit rate 63.5%. **Positive**, not negative as the equity-only low-vol anomaly would suggest. On this cross-asset universe, the **cross-asset risk premium dominates** the intra-equity low-vol effect. Robust across sub-periods (2003-09: +0.145; 2016-22: +0.152; 2023-26: +0.168) and sub-universes (equities +0.096; bonds +0.078; alts +0.054). The low-vol anomaly is an intra-equity-class phenomenon; on a cross-asset universe, raw vol is a positive signal.
- **SignalTilt(momentum_252, tilt=0.5)**: ΔSharpe = +0.267 vs EW (Sharpe 1.156 vs 0.889), confirming the feature plumbing works end-to-end. Partly attributable to momentum + high-vol name overlap (tech equities dominate both).

**Implications for Session 2 ML:**
- Do not impose "low-vol = good" as a prior. Feed raw volatility as a feature alongside an asset-class indicator; let the model learn within-class vs cross-class structure.
- Momentum and volatility are structurally correlated on this universe and window. Treat as collinear features in Lasso; check feature-importance interpretation carefully.

Full validation report: [`docs/validation/session_1.5b_feature_library.md`](docs/validation/session_1.5b_feature_library.md).

---

**Session 2 ML strategies — final synthesis (29-asset 2003-2026 universe, test period 2023–2026):**

1. **MSR(ML μ̂) is the value extractor.** Approach A (MSR with ML-predicted μ̂) outperforms Approach B (SignalTilt wrapping) for Lasso and RF. Best single-fit results: MSR(RF_μ̂) 2.394, MSR(Lasso_μ̂) 2.272. Exception: XGB where SignalTilt(XGB) 2.304 edges MSR(XGB_μ̂) 2.180, explained by the optimizer amplifying XGB's noisy predictions. MSR wrapping works best when predictions are well-calibrated.

2. **Ensemble is the headline result.** MSR(Ensemble_μ̂) Sharpe **2.579** tops the 28-strategy extended comparison, beating the next-best classical strategy VMP(MDP(LW)) at 2.422 and all single-model ML entries. Equal-weighted average of Lasso + RF + XGB predictions fed into MSR; no retraining required. Ensemble reduces idiosyncratic model noise and stabilizes the optimizer inputs.

3. **VMP overlay does not help ML strategies.** VMP-wrapped ML variants consistently rank below their un-wrapped counterparts: VMP(SignalTilt(XGB)) 2.292 < SignalTilt(XGB) 2.304; VMP(MSR(RF_μ̂)) 2.177 < MSR(RF_μ̂) 2.394. VMP helps classical strategies (whose returns are noisier) but ML strategies already apply a form of signal normalization; the additional vol-scaling layer adds friction without signal.

4. **HP sensitivity: sharp val/OOS disagreement for XGB.** XGB default (depth=6) shows catastrophic validation overfitting — `best_iteration=0`, val_IC = 0.013 — yet ranks near the top OOS at Sharpe 2.304. The 2019–2022 validation window (COVID + rate-hike cycle) is poorly representative of 2023–2026 (post-shock environment). Generalizes across models: val_IC ranking (Lasso > RF > XGB) disagrees with OOS Sharpe ranking (XGB > RF > Lasso for SignalTilt). HP validation must be treated skeptically on this regime-shifted universe.

5. **Walk-forward refit underperforms single-fit.** All 9 walk-forward strategies (annual refit, trailing 10-year window, default and val-optimal HPs) rank below their single-fit counterparts. Best WF: SignalTilt(WF-lasso-default) 2.033 vs single-fit SignalTilt(Lasso) 2.140. Single-fit models trained through 2020 encode the COVID + early rate-shock regime in their feature weighting; the test period (2023–2026) remains a post-shock environment where that memory is an asset. Rolling refits that drop 2003–2010 lose regime diversity. Default HPs beat val-optimal for Lasso in WF (validation bias confirmed) but not for RF or XGB.

**Implications for Session 3 DL:**
- Use single-fit methodology for the first DL pass. Walk-forward underperformed here; don't repeat without specific motivation.
- Test both MSR(DL μ̂) and SignalTilt(DL) wrappers — the winner is model-dependent.
- Skip VMP-wrapping for DL unless a specific reason emerges.
- Plan to ensemble DL models (MLP + LSTM + Transformer) as the headline DL strategy.
- Sharpe bar: 2.579. DL angle: sequence dependencies and cross-asset attention that tree-based models miss.
