# Project State Review — 2026-05-20

## 1. Repo identity & branch state

Commands run: `git status`, `git log --oneline -20`, `git branch -a`, `git show -s --format='%H%n%s%n%cI' HEAD`, `git branch -vv`, `git rev-list --left-right --count main...rl/foundation`, `git rev-list --left-right --count origin/main...main`, `git rev-list --left-right --count origin/rl/foundation...rl/foundation`.

`git status` before writing this report: clean on `rl/foundation`. HEAD is `4fd493c07c30291300b40482e1dabc699cd9447e`, message `rl: N=29 walk-forward training, λ ablation, comparison vs ML/DL paradigms`, committed `2026-05-19T18:26:53-05:00`.

`git branch -a` shows local `main`, current `rl/foundation`, local `unsupervised/representation-learning`, and remotes `origin/main`, `origin/rl/foundation`. `git branch -vv` shows `main` at `d236ce9 [origin/main]`, `rl/foundation` at `4fd493c`, and `unsupervised/representation-learning` at `5cc1ee4`; local `main` and `rl/foundation` are each `0 0` against their upstreams.

`main...rl/foundation` is `2 4`, so RL work is still branch-local relative to `main`: `rl/foundation` has 4 commits not on local `main`, and local `main` has 2 commits not on `rl/foundation`. RL is not merged back to `main` yet.

Recent log:

```text
4fd493c (HEAD -> rl/foundation, origin/rl/foundation) rl: N=29 walk-forward training, λ ablation, comparison vs ML/DL paradigms
a8d3af6 rl: REINFORCE+baseline trainer, N=1/2/3 trained runs, multi-seed ensemble
108129d Notebook 05 closeout: findings doc + commit Colab results
e6e4498 rl: scaffold N-asset env, simplex policy, strategy wrapper, sanity notebook
1dd9fa2 notebook 05 §0: idempotent Colab setup (prevents directory nesting on rerun)
3a79865 notebook 05 fix: Colab badge + CSV fallback + gitignore cleanup
5db3996 DL session 2: rewrite Notebook 05 for JPM-spec walk-forward
7386d17 DL session 1: walk-forward infrastructure + raw history features
4336479 session C: notebook 05 — DL portfolio construction exploration on SPY+IEF
23d5ce4 session B: intraday data infrastructure + realized power feature
ab3544d cleanup: remove 04c (29-asset direct-weight) artifacts; preserve infrastructure modules
524c6fb 04b: restore Colab badge + sentinel fix + clean Colab metadata noise
e839131 04 polish: reorder §16/§17 + drop sanity-check row
d6a0312 session 3d-c closeout: direct-weight findings + figures + stability fix + framing cleanup
071e355 Created using Colab
83b2ccd Created using Colab
2c8914e session 3d-b: direct-weight DL experiment notebook
4177ada session 3d-a: direct-weight DL policy scaffolding
6ae8082 repo cleanup: move internal working artifacts to local-only
f830008 session 3 closeout: findings doc + verdict fix + Colab badge + results
```

## 2. Package & module inventory

Commands run: `find src/aiam -maxdepth 3 -type f | sort`, `find src/aiam -type f -name '*.py' -print0 | xargs -0 wc -l | sort -n`, plus read-only AST parsing for class inventory.

`src/aiam/` totals 5,109 Python lines. Two-level source inventory: `data/` has `intraday.py` 178, `panel.py` 44, `split.py` 15, `universe.py` 29, `regimes/regime_engine.py` 131; `strategy/` has 18 files led by `dl_policy_strategies.py` 427, `dl_strategies.py` 338, `ml_strategies.py` 190; `dl/` has 6 files led by `workflow.py` 314, `policy_workflow.py` 237, `walkforward.py` 199; `rl/` has 6 files led by `trainer.py` 204, `walkforward.py` 188, `env.py` 150.

Other packages: `features/technical.py` 129, `asset_class.py` 70, `realized_power.py` 68; `evaluation/transaction_costs.py` 76, `regime_conditional.py` 58, `ic.py` 52; `estimators/views.py` 53, `factor_signals.py` 49, `covariance.py` 21; `harness/horse_race.py` 86; `ml/workflow.py` 113.

Top files by subpackage and purpose:

| Subpackage | Top files | Purpose |
|---|---|---|
| `strategy/` | `dl_policy_strategies.py` 427; `dl_strategies.py` 338; `ml_strategies.py` 190 | direct-weight DL policies, DL signal strategies, ML signal-to-weight strategies |
| `dl/` | `workflow.py` 314; `policy_workflow.py` 237; `walkforward.py` 199 | supervised DL fitting, direct-weight policy fitting, walk-forward orchestration |
| `rl/` | `trainer.py` 204; `walkforward.py` 188; `env.py` 150 | REINFORCE trainer, walk-forward RL ensemble, portfolio environment |
| `data/` | `intraday.py` 178; `regimes/regime_engine.py` 131; `panel.py` 44 | intraday fetch/cache, macro regime pipeline, no-look-ahead data access |
| `features/` | `technical.py` 129; `asset_class.py` 70; `realized_power.py` 68 | technical signals, asset-class one-hot features, intraday realized power |
| `evaluation/` | `transaction_costs.py` 76; `regime_conditional.py` 58; `ic.py` 52 | net returns, regime Sharpe tables, information-coefficient diagnostics |

Potential orphan/duplication flags: `src/aiam/strategy/dl_policy_strategies.py` is preserved infrastructure from the reverted 29-asset direct-weight experiment but still used by Notebook 05; cache-era strategy files from 8/14/16/20/24/28 strategy builds still linger under `data/cache/portfolio_returns/`; `__pycache__/` files exist under source and tests but are ignored artifacts.

## 3. Strategy inventory

Commands run: `rg -n "^class [A-Za-z_][A-Za-z0-9_]*" src/aiam/strategy src/aiam/rl`, read-only AST parsing, and CSV read of `data/cache/appendix_a_canonical.csv`.

Concrete strategy classes in `src/aiam/strategy/`: `EqualWeight`, `GlobalMinVariance`, `MaximumSharpe`, `MaximumSharpeConstrained`, `MVOConstrained`, `MostDiversified`, `RiskParity`, `HierarchicalRiskParity`, `TSMOM`, `BlackLitterman`, `FactorPortfolio`, `FF3MomLongShort`, `MultiFactorPortfolio`, `SwitchingStrategy`, `SignalTilt`, `LassoSignalStrategy`, `RFSignalStrategy`, `XGBSignalStrategy`, `MLPSignalStrategy`, `LSTMSignalStrategy`, `TransformerSignalStrategy`, `EnsembleDLSignalStrategy`, `DirectWeightMLPStrategy`, `DirectWeightLSTMStrategy`, `DirectWeightTransformerStrategy`, `DirectWeightShrinkageStrategy`.

Concrete RL strategy wrapper is `src/aiam/rl/strategy.py:RLStrategy`. Abstract/base classes excluded from the count: `Strategy`, `PointInTimeStrategy`, `_MLSignalBase`, `_DLSignalBase`, `_DLPolicyBase`.

Family grouping: Mean-Variance = `EqualWeight`, `GlobalMinVariance`, `MaximumSharpe`, `MaximumSharpeConstrained`, `MVOConstrained`; Risk Parity / Diversification = `MostDiversified`, `RiskParity`, `HierarchicalRiskParity`; Factor = `FactorPortfolio`, `FF3MomLongShort`, `MultiFactorPortfolio`, `SignalTilt`; Regime = `SwitchingStrategy`; Bayesian / BL = `BlackLitterman`; ML = `LassoSignalStrategy`, `RFSignalStrategy`, `XGBSignalStrategy`; DL = signal and direct-weight DL classes; RL = `RLStrategy`; Other = `TSMOM`.

Appendix A canonical CSV is `data/cache/appendix_a_canonical.csv` with 62 rows: 31 base and 31 VMP. Family counts are Classical MV 12, Diversification 12, Black-Litterman 8, Factor 8, Constrained MV 8, Long-Short 6, Regime Switch 4, TS Momentum 4. Counts match the stated 31 base + 31 VMP paper harness; ML/DL/RL are extension tracks, not in Appendix A.

## 4. Notebook inventory

Commands run: `find notebooks -name '*.ipynb' -maxdepth 2 -print0 | xargs -0 ls -lT`, `find notebooks -name '*.ipynb' -maxdepth 2 -print0 | xargs -0 wc -l | sort -n`, plus read-only JSON parsing of first markdown cells.

| Notebook | Lines | Modified | Purpose |
|---|---:|---|---|
| `notebooks/01_paper_reproduction.ipynb` | 319 | May 16 12:33 | Paper reproduction comparative allocation harness |
| `notebooks/02_practitioner_analytics.ipynb` | 1,008 | May 16 12:33 | Practitioner analytics and OOS holdout |
| `notebooks/03_ml_strategies.ipynb` | 1,754 | May 17 06:49 | ML strategy setup and methodology |
| `notebooks/04_dl_strategies.ipynb` | 1,375 | May 19 07:45 | DL strategy setup and methodology |
| `notebooks/04b_dl_strategies_cuda.ipynb` | 1,091 | May 19 08:13 | CUDA DL run notebook |
| `notebooks/04c_rl_sanity.ipynb` | 244 | May 19 17:24 | RL sanity check, N=1 SPY |
| `notebooks/04d_rl_random_n2.ipynb` | 282 | May 19 17:34 | RL untrained N=2 sanity check |
| `notebooks/04e_rl_training.ipynb` | 919 | May 19 17:52 | RL training, N=1/2/3 |
| `notebooks/04f_rl_n29_training.ipynb` | 1,312 | May 19 18:25 | RL N=29 training |
| `notebooks/05_dl_portfolio_construction_exploration.ipynb` | 1,093 | May 19 17:24 | Notebook 05 DL portfolio construction |
| `notebooks/_archived_01_static_baselines_pre_split.ipynb` | 2,171 | May 16 12:33 | Archived pre-split static baselines |

Checkpoint notebooks mirror the first five notebooks and are present under `notebooks/.ipynb_checkpoints/`. No notebook has an mtime older than 5 days as of 2026-05-20, so the requested “old but recently modified” drift signal does not trigger from filesystem timestamps alone.

## 5. Data caches

Commands run: `ls -la data/cache/*.parquet data/raw/*.csv`, `find data/cache -maxdepth 2 -type f | sort`, `ls -la data/cache/portfolio_returns results/rl results/rl/n29`, `find data -name '*features*' -o -name '*58strategies*' -o -name '*31strategies*' | sort`.

Top-level raw/cache files include `prices_29_ohlcv_2003_2026.parquet` 4.6 MB, `returns_29_2003_2026.parquet` 1.6 MB, `prices_29.parquet` 1.2 MB, `regime_signals_2003_2026.parquet` 11 KB, `regime_conditional_sharpe_29.parquet` 6 KB, and raw `data/raw/prices_29_ohlcv_2003_2026.csv` 10.4 MB. Expected `features_29_2003_2026.parquet` is missing.

Strategy caches exist at `data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet` and `31strategies_vmp_29assets_2003_2026.parquet`, both May 15 12:01. `switch_v2a_oos_29assets.parquet`, ML caches, DL caches, and `full_comparison_with_rl.csv` are also present.

Old/odd cache names still linger: `data/cache/prices_30.parquet` remains from May 12, plus older portfolio return generations `8strategies_no_btc_2008_2026.parquet`, `14strategies_*`, `16strategies_*`, `20strategies_*`, `24strategies_*`, and `28strategies_*`. No `58strategies_*` files were found; the May 15 rename to `31strategies_*` appears complete for the canonical 29-asset caches.

Regime signals are present in both old and current forms: `regime_signals.parquet` and `regime_signals_2003_2026.parquet`. The duplicate old name is a reproducibility/backward-compatibility risk unless intentionally retained.

## 6. Tests

Command run: `PYTHONPATH=src pytest --collect-only -q`. Collection output listed 294 tests before interruption, then exited code 2 with 3 collection errors.

Collection errors are all missing optional dependency in the active interpreter: `ModuleNotFoundError: No module named 'cvxpy'` while importing `tests/test_black_litterman.py`, `tests/test_horse_race.py`, and `tests/test_max_sharpe.py`. `requirements.txt` does declare `cvxpy>=1.4`, so this looks like environment drift rather than an undeclared dependency.

Test files discovered: 33 `test_*.py` files, including 5 under `tests/rl/`. No test file had zero `def test_` functions by `rg -c`.

Mirror-completeness is good for high-risk modules: tests cover `strategy`, `features`, `evaluation`, `data/panel`, `data/intraday`, `ml`, `dl`, and `rl`. Gaps by structure: no direct test file for `src/aiam/data/universe.py`, `src/aiam/data/split.py`, `src/aiam/estimators/covariance.py`, `src/aiam/estimators/mean.py`, or `src/aiam/estimators/views.py` beyond integration through strategy tests.

## 7. Documentation & paper

Commands run: `ls -lT docs/results.pdf docs/results.md`, `find docs/handoff -maxdepth 1 -type f -name '*.md' -print0 | xargs -0 ls -lT`, `find docs/validation -maxdepth 1 -type f -name '*.md' -print0 | xargs -0 ls -lT`, `find docs/figures -maxdepth 1 -type f -name '*.png' -print0 | xargs -0 ls -lT`.

Paper files: `docs/results.pdf` is 6,006,925 bytes, modified May 15 23:00:22; `docs/results.md` is 77,262 bytes, modified May 15 22:58:51. No newer paper render is visible after Session 4 RL.

Handoff trail: `session_2_to_3.md` May 17, `notebook_05_findings.md` May 19 16:53, `session_3_findings.md` May 19 16:53, `session_4c_findings.md` May 19 18:25. This new review file will become the next handoff artifact.

Validation docs: `session_1.5b_feature_library.md` May 16 and `session_2_ml_strategies.md` May 17. There is no DL, Notebook 05, RL, or LLM validation document under `docs/validation/`.

Figure PNGs: 12 present: the canonical 11 from `scripts/generate_figures.py` plus `rf_permutation_importance.png`. Required audit examples `asset_class_allocation_timeline.png` and `rolling_sharpe_small_multiples.png` exist; no obvious missing canonical figure from the script’s 11 savefig targets.

## 8. CLAUDE.md health-check

Commands run: `rg -n '^## ' CLAUDE.md`, `sed -n '1,230p' CLAUDE.md`, and targeted `rg` for stale counts/cache names.

Current H2 headings: Project; Status; Environment; Architecture (locked decisions); Repository structure; Strategy inventory (31 base + 31 VMP = 62 total); Canonical commands; Claude Code prompt conventions; Non-negotiable rules; Reference repos; Roadmap; Known issues (defer to Session 1.5B); Validation Findings.

Staleness is significant. `CLAUDE.md` says current state is commit `edf88c9`, next is Session 3, test suite is 124 passing, docs mention only notebooks 01-03 in the status section, and repository structure says tests have 11 files. Actual HEAD is `4fd493c`, RL work exists, tests discovered are 33 files / 294 collected-before-error.

Architecture text mentions `SequentialStrategy.step(...)`, but no `SequentialStrategy` class exists in `src/aiam/strategy/base.py`; the current RL wrapper subclasses `PointInTimeStrategy`. Roadmap still says Session 4 is PPO/SAC via `notebooks/06_rl_strategies.ipynb`, while actual RL work is REINFORCE+baseline in `notebooks/04c`-`04f` and `src/aiam/rl/`.

Known-issues section includes `_PRICES_CACHE` pointing to `prices_30.parquet` but also says resolved; source now has `_PRICES_CACHE = Path("data/cache/prices_29.parquet")`, so the issue text is stale as a live warning.

## 9. Tech-debt audit (verify May 15 audit items shipped)

Commands run: targeted `rg` for `31strategies`, `58strategies`, family labels, `Hit%`, `_PRICES_CACHE`, figure saves, and SWITCH(v2a) numbers in `scripts/`, `notebooks/`, `src/`, and `data/cache/appendix_a_canonical.csv`.

| Item | Status | Evidence |
|---|---|---|
| Cache rename `58strategies_*` → `31strategies_*` and VMP variant, references updated | ✅ shipped | `build_all_strategies_29.py`, `build_switch_oos.py`, `generate_figures.py`, and `notebooks/_shared.py` all reference canonical `31strategies_*`; no `58strategies_*` files found. |
| Family labels unified to `"Regime Switch"` and `"Factor"` | ✅ shipped | CSV, figure script, and notebook shared helpers all use these exact labels. |
| Hit% blank cells filled with `—` for VMP rows | ✅ shipped | `appendix_a_canonical.csv` VMP rows have `—`; notebook display code maps missing/blank to `—`. |
| `_PRICES_CACHE` in `horse_race.py:12` updated to `prices_29.parquet` | ✅ shipped | `src/aiam/harness/horse_race.py:12` is `_PRICES_CACHE = Path("data/cache/prices_29.parquet")`. |
| 11 figures in `scripts/generate_figures.py` with docstring matching | ✅ shipped | Docstring says “Generate 11”; save targets are 11 PNGs and final print says all 11 generated. |
| Notebook §2.13 §5.3 SWITCH(v2a) numbers: Sharpe 1.514, Δ 0.434, z 2.05, p 0.040 | ⚠ partial | `build_all_strategies_29.py` comment references Sharpe 1.514; targeted grep did not find `0.434`, `2.05`, or `0.040` in the notebook/source paths checked. |

## 10. Session 4 (RL) status

Commands run: `find results/rl/n29 -maxdepth 2 -type f -print0 | xargs -0 ls -lT`, `find docs/handoff -maxdepth 1 -type f -name 'session_4*.md' -print0 | xargs -0 ls -lT`, `find notebooks -maxdepth 1 -type f \( -name '04*' -o -name '0?_rl_*.ipynb' \) -print0 | xargs -0 ls -lT`, and class greps under `src/aiam/rl`.

RL code exists locally: `src/aiam/rl/agent.py`, `env.py`, `policy.py`, `strategy.py`, `trainer.py`, and `walkforward.py`. Scaffolding includes `PortfolioEnv`, `SimplexPolicy`, `RLAgent`, `RLStrategy`, `TrainConfig`, `TrainHistory`, `ValueHead`, `WalkForwardRLEnsemble`, and `fit_walkforward_rl`.

RL notebooks and handoff: `04c_rl_sanity.ipynb` 7.6 KB, `04d_rl_random_n2.ipynb` 462 KB, `04e_rl_training.ipynb` 724 KB, `04f_rl_n29_training.ipynb` 309 KB, all modified May 19; `docs/handoff/session_4c_findings.md` is 16.9 KB, modified May 19 18:25.

Local N=29 artifacts are present but compact: `results/rl/n29/diagnostics_all.parquet` 9 KB plus four figures (`equity_curves_rl.png`, `seed_sharpe_dist_rl.png`, `turnover_over_time_rl.png`, `weight_heatmap_rl.png`). `data/cache/portfolio_returns/full_comparison_with_rl.csv` also exists.

No large serialized trained agents/checkpoints were found under `results/rl/n29`; G4-style full training artifacts appear summarized locally via diagnostics/figures, with model state either not saved or only available in the notebook/Colab runtime.

## 11. Session 5 (LLM views) readiness

Commands run: `sed -n '1,220p' src/aiam/estimators/views.py`, `sed -n '1,170p' src/aiam/strategy/black_litterman.py`, and function/class greps.

`src/aiam/estimators/views.py` exists and contains three function-based view generators: `equilibrium_only(returns, asof)`, `momentum_views(returns, asof, signal_lookback=252, view_uncertainty_scaler=0.05)`, and `mean_reversion_views(returns, asof, long_lookback=1260, view_uncertainty_scaler=0.05)`.

There is no `MomentumViews` class, but the existing function interface is already Callable-shaped: each generator receives `(returns: pd.DataFrame, asof: pd.Timestamp)` and returns `(P, Q, Omega)`. An `LLMViewGenerator` could slot in as a callable object if it preserves that signature and returns NumPy arrays with compatible dimensions.

`BlackLitterman.__init__` accepts `view_generator: Callable` as the first constructor argument, followed by `cov_estimator: Callable`, `lookback`, `tau`, `delta`, `prior_weights_method`, and `long_only`. It calls `P, Q, Omega = self.view_generator(returns, asof)`.

Session 5 can fire on existing BL infrastructure without changing the `BlackLitterman` contract. It still needs scaffolding for prompt/config management, provider clients, deterministic caching, offline tests/mocks, and leakage controls before any LLM-generated views should enter experiments.

## 12. Part VIII gap (NEW — LLM/agent layer)

Commands run: `find . -path './.venv' -prune -o -path './.git' -prune -o -type f \( -iname '*llm*.py' -o -iname '*agent*.py' -o -iname '*prompt*.py' -o -iname '*.yaml' -o -iname '*.yml' \) -print | sort`, and `rg -n --glob '!*.ipynb' --glob '!*.html' 'anthropic|openai|sentence-transformers|llm|agent|prompt' pyproject.toml requirements.txt src notebooks scripts docs CLAUDE.md README.md`.

No LLM-specific Python, prompt, or YAML config files were found outside virtualenv/git. The only matching project file is `src/aiam/rl/agent.py`, which is reinforcement-learning infrastructure, not an LLM/agent layer.

No non-notebook source references Anthropic/OpenAI API usage. Text references exist only at planning/documentation level: `CLAUDE.md` mentions LLM agents in the aspirational `SequentialStrategy` architecture, and `docs/results.md` mentions future reinforcement learning agents via `SequentialStrategy`.

`requirements.txt` and `pyproject.toml` contain no `anthropic`, `openai`, `sentence-transformers`, `langchain`, `llama`, or `tiktoken` entries. `requirements.txt` does contain `cvxpy>=1.4`, relevant to the test collection error.

Natural landing path: create `src/aiam/llm/` alongside `src/aiam/ml/`, `src/aiam/dl/`, and `src/aiam/rl/`, with a view-generator adapter likely imported by `src/aiam/estimators/views.py` or passed directly into `BlackLitterman`. A conservative first split would be `src/aiam/llm/views.py`, `prompts.py`, `cache.py`, and `schemas.py`, with provider-specific code isolated behind an interface.

## 13. Open questions / blockers

- Should `rl/foundation` be rebased/merged against local `main` before Session 5, given `main...rl/foundation` is `2 4` rather than a pure fast-forward?
- Should `CLAUDE.md` be updated before the next coding session so agents do not follow stale Session 3 / `SequentialStrategy` / test-count guidance?
- Is `features_29_2003_2026.parquet` intentionally obsolete, or should the expected feature cache be rebuilt/renamed?
- Are the old `8/14/16/20/24/28strategies_*` and `prices_30.parquet` caches intentionally retained, or should a cleanup prompt remove them from local cache only?
- Should Session 4 serialize RL model checkpoints locally, or are diagnostics/figures enough because the full trained artifacts live only in Colab/runtime output?
- Follow-up prompt warranted: fix environment drift so `PYTHONPATH=src pytest --collect-only -q` can import `cvxpy`, then rerun collection and optionally the focused strategy tests.
- Follow-up prompt warranted: add tests for `src/aiam/estimators/views.py` before introducing LLM views, because the BL view-generator contract is central to Session 5.
