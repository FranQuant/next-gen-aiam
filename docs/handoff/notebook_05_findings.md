# Notebook 05 Findings — Direct-Weight DL Portfolio Construction on SPY+IEF

**Status**: Pending. The notebook is committed and ready to execute on M4 locally.
This findings doc will be populated with results after the user runs the full
10-seed × 18-config experiment.

## Experimental Setup

- **Universe**: SPY.US + IEF.US (2 assets, equity + 10-year Treasury)
- **Period**: 2021-01-04 → 2026-04-30 (5.3 years, intraday-covered)
- **Splits** (chronological, non-overlapping):
  - Train: 2021-01 → 2023-12 (~750 days, 56%)
  - Validation: 2024-01 → 2024-06 (~125 days, 9%)
  - Test: 2024-07 → 2026-04 (~455 days, 35%)
- **Architectures**: MLP, LSTM, Transformer (all three)
- **Feature variants** (central ablation):
  - **Daily-only**: returns, mom_5, mom_21, mom_63, vol_60, vol_252 (6 features)
  - **Daily + realized power**: above + rp_daily (7 features, 1-day lagged)
- **Loss variants**: Sharpe, CRRA-γ5, CRRA+Shrinkage-to-risk-parity (sigmoid multiplier)
- **Total configs**: 3 archs × 2 feature variants × 3 losses = 18 configs
- **Seeds**: 10 per config → 180 policies total
- **Method**: Single fit, no walk-forward refit (consistent with Sessions 2-3 methodology)
- **Lookback (LSTM/Transformer)**: 21 days
- **Primary citation**: Brandt, Santa-Clara & Valkanov (2009)

## Benchmarks

- Risk Parity (21-day vol-weighted, daily rebalance)
- 60/40 static allocation
- Equal-weight 50/50

## Pending Results Sections

- Headline OOS Sharpe by config (table, 21 rows)
- Feature ablation effect: daily-only vs daily+RP (mean ΔSharpe across arch×loss)
- Architecture comparison: MLP vs LSTM vs Transformer
- Loss function comparison: Sharpe vs CRRA vs CRRA+Shrink
- Per-seed stability: 10-seed mean ± stdev by config
- Comparison vs benchmarks (risk parity, 60/40, equal-weight)
- Verdict: does intraday realized power improve OOS performance?

## Reproducibility

- Cached intraday data: `data/cache/intraday_5min_SPY_IEF_2021_2026.parquet` (5 MB, gitignored)
- Published CSV: `data/published/intraday_5min_SPY_IEF_2021_2026.csv` (17 MB, tracked)
- Notebook: `notebooks/05_dl_portfolio_construction_exploration.ipynb`
- Infrastructure modules:
  - `src/aiam/data/intraday.py`
  - `src/aiam/features/realized_power.py`
  - `src/aiam/dl/losses.py`, `policies.py`, `policy_workflow.py`
  - `src/aiam/strategy/dl_policy_strategies.py`
- Output dir: `results/notebook_05/` (comparison.csv, stability.csv, strategy_returns.parquet, 6 figures)
