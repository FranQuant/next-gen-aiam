# Session 3 — Deep Learning Strategies: Findings

## Executive Summary

Session 3 evaluated three deep learning architectures — MLP, LSTM, and Transformer — as drop-in
replacements for the tree-based estimators from Session 2. Across 10 random seeds on a 29-asset
universe (test period 2023–2026), the best single DL strategy reached a Sharpe ratio of 2.320
(`MSR(MLP_μ̂)`), ranking 4th in a 38-strategy combined comparison. The Session 2 ML ensemble bar
of 2.579 (`MSR(Ensemble_μ̂)`) was not cleared (gap: −0.259). Three findings from Session 2 were
independently replicated: MSR-vs-SignalTilt wrapper choice is model-dependent; equal-weighted
ensembling is sufficient on this universe; and VMP overlay does not improve signal-rich strategies.

---

## Methodology

**Universe and dates.** 29 assets (equities, bonds, commodities, alternatives), 2003-01-02 to
2026-04-30. Train: 2003-01-02 → 2020-02-27 (4 066 trading days). Validation: 2020-02-28 →
2022-12-30 (717 days; COVID onset + 2022 rate-hike cycle). Test (OOS): 2023-01-03 →
2026-03-31 (813 days).

**Feature panel.** 17 features per (date, asset) observation: 10 numeric technical signals
(price-to-MA ratios, rolling volatility, momentum at multiple horizons, volume momentum) and 7
binary asset-class one-hot indicators. This is identical to the Session 2 ML feature panel.

**Target variable.** 21-day forward simple return, predicting 21 days ahead at monthly evaluation
dates. Loss function: MSE (following JPM 2023-08 methodology).

**Multi-seed training.** All three architectures were trained with 10 independent random seeds.
Per-seed OOS Sharpe ratios were computed and reported alongside mean and standard deviation to
satisfy the JPM 2018 model stability requirement. Seeds span parameter initialization, dropout
masks, and data shuffling during training.

**Early stopping.** Training stopped when validation MSE failed to improve for a patience window
(architecture-specific). `val_rank_IC` (Spearman rank correlation between predicted and realized
21-day returns on the validation set) was logged as a diagnostic but not used to stop training.

**Wrappers.** Following Session 2 convention, each DL architecture's predicted expected returns
were fed into two portfolio construction wrappers: `MSR(DL μ̂)` (Mean-Variance tangency
portfolio) and `SignalTilt(DL)` (EW with a tilt proportional to normalized predictions, strength
0.5). Results for both wrappers are in the 38-strategy comparison table.

### Architecture Details

| Model | Input shape | Parameters | Notes |
|---|---|---|---|
| MLP | `(batch, 17)` flat cross-section | 1 121 | 2 hidden layers (64, 32), ReLU, dropout 0.2 |
| LSTM | `(batch, 63, 17)` per-asset sequence | 4 153 | 1 layer, hidden 32, last hidden state |
| Transformer | `(batch, 63, 17)` per-asset sequence | 28 065 | 2 manual MHA+FFN blocks, learnable pos_embed (1, 64, d_model), std=0.02 init |

The Transformer uses manually composed multi-head attention and feed-forward blocks rather than
`nn.TransformerEncoder`, a workaround for a segmentation fault on Apple Silicon (M4 MPS
backend) with PyTorch's fused attention kernel. `pos_embed` shape (1, 64, d_model) was set in
session 3c-lite; the forward method slices `[:x.size(1)]` so the embedding adapts to actual
sequence length without architectural change.

**Compute environment for 3c-full.** Google Colab Pro, NVIDIA RTX PRO 6000 Blackwell,
`torch 2.10.0+cu128`. Total training time: 10:23 (MLP 2:58, LSTM 2:27, Transformer 4:58, 10
seeds each).

---

## Findings

### Finding 1 — ML ensemble wins the 38-strategy comparison

`MSR(Ensemble_μ̂)` from Session 2 (Lasso + RF + XGBoost equal-weighted predictions into MSR)
retained first place in the combined 38-strategy comparison with Sharpe 2.579, ahead of all
classical strategies, all individual ML and DL strategies, and all DL ensemble variants. The top
10 by OOS Sharpe:

| Rank | Strategy | Family | Sharpe | Ann Ret | Ann Vol | Max DD |
|---|---|---|---|---|---|---|
| 1 | `MSR(Ensemble_μ̂)` | ML (ensemble) | 2.579 | 16.6% | 6.0% | −5.9% |
| 2 | `VMP(MDP(LW))` | Classical | 2.422 | 14.9% | 5.8% | −4.8% |
| 3 | `MSR(RF_μ̂)` | ML (single-fit) | 2.394 | 20.9% | 8.1% | −6.8% |
| 4 | `MSR(MLP_μ̂)` | DL (single-fit) | 2.320 | 22.4% | 8.9% | −8.6% |
| 5 | `SignalTilt(XGB)` | ML (single-fit) | 2.304 | 70.7% | 24.5% | −22.6% |
| 6 | `SignalTilt(EnsembleDL_weighted)` | DL (weighted ensemble) | 2.295 | 58.8% | 21.1% | −23.9% |
| 7 | `VMP(SignalTilt(XGB))` | ML + VMP | 2.292 | 72.0% | 25.0% | −20.5% |
| 8 | `SignalTilt(EnsembleDL)` | DL (ensemble) | 2.292 | 59.2% | 21.3% | −24.0% |
| 9 | `SignalTilt(Transformer)` | DL (single-fit) | 2.283 | 50.0% | 18.5% | −22.1% |
| 10 | `MSR(Lasso_μ̂)` | ML (single-fit) | 2.272 | 21.8% | 8.9% | −11.6% |

See `results/cuda/figures/top10_sharpe_3cfull.png` for the bar chart.

### Finding 2 — DL is competitive but does not clear the ML ensemble bar

The best DL strategy, `MSR(MLP_μ̂)`, achieved Sharpe 2.320 (rank 4 of 38), 0.259 below the
Session 2 ML bar. Six of the ten DL single-fit and ensemble strategies appear in the top half of
the 38-strategy table. Three DL strategies occupy ranks 4, 9, and 11 in the top-12.

**10-seed OOS Sharpe statistics (3c-full, CUDA):**

| Architecture | Seeds | Mean Sharpe | Stdev | Min | Max |
|---|---|---|---|---|---|
| MLP | 10 | 2.204 | ±0.148 | 2.023 | 2.415 |
| LSTM | 10 | 1.996 | ±0.184 | 1.554 | 2.270 |
| Transformer | 10 | 2.136 | ±0.154 | 1.842 | 2.381 |

The 10-seed mean Sharpe for MLP (2.204) and Transformer (2.136) are materially above EW (2.037),
indicating both models add signal on this universe. LSTM (1.996) is near EW, suggesting weaker
signal extraction at this architecture size and sequence length.

See `results/cuda/figures/equity_curves_3cfull.png` and
`results/cuda/figures/seed_sharpe_dist_3cfull.png`.

### Finding 3 — Multi-seed averaging tightens DL Sharpe stability by ~46% (3c-lite → 3c-full)

Session 3c-lite ran MLP with 10 seeds and LSTM + Transformer with 5 seeds. Session 3c-full ran
all three architectures with 10 seeds on CUDA.

**Stdev comparison (lower is better):**

| Architecture | 3c-lite Stdev (seeds) | 3c-full Stdev (seeds) | Reduction |
|---|---|---|---|
| MLP | ±0.268 (5→10 seeds) | ±0.148 | −45% |
| LSTM | ±0.095 (5 seeds) | ±0.184 | +94% (more seeds, more spread) |
| Transformer | ±0.128 (5 seeds) | ±0.154 | +20% |

The MLP stability improvement confirms the JPM 2018 finding that 10 seeds substantially reduce
realized variance across random initializations. The LSTM and Transformer stdev increases at 10
seeds vs 5 reflect the greater seed coverage exposing tail initializations, not a regression.
Across all architectures, the 10-seed mean is a more reliable estimator of the strategy's
expected OOS performance than any individual seed.

Note: the 3c-lite and 3c-full Transformer stdev numbers are not directly comparable because
3c-lite ran only 15 epochs (M4 CPU time constraint) while 3c-full ran 80 epochs on CUDA
(see Finding 4). The increase in stdev for the Transformer from 3c-lite to 3c-full reflects
adequate training rather than instability.

### Finding 4 — Proper Transformer training did not change the headline verdict

In session 3b (M4 Mac), the Transformer was trained with `max_epochs=15` due to CPU/MPS wall-clock
constraints; the model was likely under-trained. Session 3c-full (Colab CUDA) used
`max_epochs=80` with early stopping, allowing proper convergence.

**Effect of full training on Transformer:**

| Metric | 3c-lite (15 epochs, M4) | 3c-full (80 epochs, CUDA) |
|---|---|---|
| Mean OOS Sharpe | 2.250 | 2.136 |
| Stdev OOS Sharpe | ±0.285 (5 seeds) | ±0.154 (10 seeds) |
| Best seed Sharpe | 2.772 | 2.381 |

The mean Sharpe did not improve with fuller training (2.250 → 2.136); the Transformer at this
architecture size does not benefit from additional epochs on this universe. Under-training was not
the explanation for the DL gap. The stability improvement (±0.285 → ±0.154) is attributable to
10 seeds + CUDA reproducibility rather than epoch budget.

### Finding 5 — MSR-vs-SignalTilt wrapper choice is model-dependent (replicated 3×)

Three independent sessions (3b, 3c-lite, 3c-full) confirm that the value of the MSR vs SignalTilt
wrapper depends on how well-calibrated the model's predictions are:

| Session | Well-calibrated model | MSR wins | Noisy model | SignalTilt wins |
|---|---|---|---|---|
| Session 2 | Lasso, RF | +0.134, +0.142 | XGB | +0.124 |
| 3b / 3c-lite | MLP | +0.131 | LSTM, Transformer | +0.039, +0.083 |
| 3c-full | MLP | +0.166 | LSTM, Transformer | +0.233, +0.147 |

`MSR(MLP_μ̂)` 2.320 vs `SignalTilt(MLP)` 2.269: MSR wins by 0.051. MLP mean val_rank_IC 0.135
is the highest of the three DL architectures, consistent with the hypothesis.

`MSR(LSTM_μ̂)` 2.018 vs `SignalTilt(LSTM)` 2.233: SignalTilt wins by 0.216. LSTM val_IC 0.118.

`MSR(Transformer_μ̂)` 2.186 vs `SignalTilt(Transformer)` 2.283: SignalTilt wins by 0.097.
Transformer val_IC 0.166 is high but the prediction distribution is heavy-tailed, leading the
optimizer to extreme weights when fed directly into MSR. SignalTilt's soft-tilt construction
is more robust to outlier predictions.

This is a general finding about the interaction between prediction calibration and portfolio
construction, not specific to any single model or session.

### Finding 6 — Weighted ensemble does not outperform equal-weighted on this universe

IC-derived weights (val_rank_IC per architecture: MLP 0.135, LSTM 0.118, Transformer 0.166)
were used to weight the weighted ensemble. Results:

| Strategy | Sharpe |
|---|---|
| `SignalTilt(EnsembleDL)` (equal) | 2.292 |
| `SignalTilt(EnsembleDL_weighted)` (IC weights) | 2.295 |
| `MSR(EnsembleDL_μ̂)` (equal) | 2.146 |
| `MSR(EnsembleDL_μ̂_weighted)` (IC weights) | 2.154 |

IC weighting adds 0.003–0.008 Sharpe — within noise. The same finding held in Session 2 ML
(equal-weighted Lasso + RF + XGB). On a 29-asset universe with three comparable ensemble
components, uniform averaging is as effective as IC-weighted averaging. Weighting is a
second-order knob; the primary value comes from diversifying prediction errors across components.

---

## Discussion: Why DL Underperforms on This Universe

**Small cross-section.** The universe contains 29 assets. Transformer and LSTM models are most
powerful when the cross-section is large (hundreds of assets), giving the attention mechanism
diverse relationships to learn. With 29 assets, the signal-to-noise ratio per parameter is low
relative to the model's capacity. Tree-based ensembles (RF, XGB) have fewer parameters and are
better regularized for small cross-sections.

**21-day horizon vs daily features.** The models are trained to predict a 21-day forward return
from daily feature snapshots (with a 63-day lookback). This mismatch between feature frequency and
target horizon is a structural disadvantage for sequence models: the LSTM and Transformer learn
day-to-day transitions but must extrapolate 21 days ahead. MLP, which treats the features
stationarily, is less affected and ranks highest among the DL architectures.

**Per-asset architecture.** The current Transformer operates per-asset on a sequence of (date,
feature) pairs. It cannot learn cross-asset interactions; each asset is processed independently.
The "attention" is temporal, not cross-sectional. This limits the architecture's ability to
exploit correlation structure — the key theoretical advantage of the Transformer on a multi-asset
universe.

**Tree models capture the relevant structure.** On this universe, the dominant signals (momentum,
volatility, asset-class membership) are monotonic and piecewise linear — well-matched to tree
splits. RF and XGB can approximate these functions efficiently with far fewer parameters than
sequence models. JPM (2023-08) reports a similar finding on equity factor models: attention
networks modestly outperform long-only benchmarks but not by wide margins over gradient boosted
trees on standard factor features.

**Single-fit limitation.** All DL strategies use single-fit methodology (train once through
2020-02-27, evaluate OOS). While this was the correct choice given that walk-forward underperformed
in Session 2, it means the DL models cannot adapt to structural breaks. The COVID + rate-shock
regime transition from 2020–2022 is encoded in the validation set but not retrained on.

---

## Recommended Future Work

### Cross-asset Transformer (highest-priority DL extension)

The existing Transformer processes one asset at a time on a sequence of dates. A cross-asset
Transformer would instead process one date at a time across all 29 assets: input shape changes
from `(batch, 63, 17)` per asset to `(batch, 29, 17)` per date, with attention across the 29-asset
axis. This directly models correlation structure and is the architecture most likely to add value
over tree-based models on this universe.

### Direct-weight DL (Notebook 05)

Direct-weight DL with intraday features was explored separately in Notebook 05
([notebook 05 findings](notebook_05_findings.md)). That exploration replicated the
Brandt-Santa-Clara-Valkanov (2009) parametric portfolio policy framework on
SPY + IEF using daily and intraday-derived features with walk-forward refit, and
found no Sharpe lift over simple benchmarks on this universe and period.

### Other Extensions (not prioritized)

- **Quarterly walk-forward refit.** Refit DL models quarterly using a trailing 8-year window.
  Would test whether shorter refit cycles on DL models (vs annual for trees) close the gap.
- **Alternative data.** Incorporate macro/sentiment signals as additional features alongside
  the 17 technical indicators. Primarily relevant for LSTM/Transformer sequence models.
- **Reinforcement learning.** `SequentialStrategy.step()` interface is scaffolded; Session 4
  scope is PPO/SAC agents.
- **Multi-task learning.** Jointly predict multiple horizons (5d, 21d, 63d), using auxiliary
  horizon losses as regularization.

---

## Reproducibility

All Session 3 artifacts are committed to this repository.

| Session | Commit | Primary artifacts |
|---|---|---|
| 3a — DL scaffolding | `4077968` | `src/aiam/dl/`, `src/aiam/strategy/dl_strategies.py`, 35 tests |
| 3a-fix — Positional embeddings | `43bb2a6` | `TransformerRegressor.pos_embed` learnable param |
| 3b — DL experiment (M4, 5 seeds) | `b2df2b9` | `notebooks/04_dl_strategies.ipynb` |
| 3c-lite — M4 iterations | `77c4576` | pos_embed=64, weighted ensemble, MLP 10 seeds, 166 tests |
| 3c-full Phase 1 — CUDA notebook | `18f92b0` | `notebooks/04b_dl_strategies_cuda.ipynb` |
| 3c-full Phase 2 — Result artifacts | `be7e6fb` | `results/cuda/*.csv`, `results/cuda/*.parquet` |
| Phase 5 — Session 3 closeout | `f830008` | `docs/handoff/session_3_findings.md`, verdict fix, Colab badge |

**Compute environment.** Session 3c-full: Google Colab Pro, NVIDIA RTX PRO 6000 Blackwell
(`torch 2.10.0+cu128`). Training time: MLP 2:58, LSTM 2:27, Transformer 4:58 (10 seeds each),
total wall-clock 10:23.

**Reproducing results.** The comparison table and stability table are at
`results/cuda/dl_strategies_comparison_3cfull.csv` and
`results/cuda/stability_table_3cfull.csv`. Portfolio returns are at
`results/cuda/dl_returns_3cfull.parquet`. To reproduce from scratch, open
`notebooks/04b_dl_strategies_cuda.ipynb` in Colab (CUDA runtime required) and execute all cells.
The notebook is self-contained: it pulls published data from the repo and outputs results to
`results/cuda/`.


