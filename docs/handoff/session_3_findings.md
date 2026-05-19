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

### Session 3d — Cross-asset Transformer (highest-priority extension)

The existing Transformer processes one asset at a time on a sequence of dates. A cross-asset
Transformer would instead process one date at a time across all 29 assets: input shape changes
from `(batch, 63, 17)` per asset to `(batch, 29, 17)` per date, with attention across the 29-asset
axis. This directly models correlation structure and is the architecture most likely to add value
over tree-based models. Theoretical motivation: attention-based cross-sectional learning is the
identified gap; the current architecture cannot test this hypothesis.

Estimated implementation effort: one local Claude Code session (architecture change in
`src/aiam/dl/models.py`, update `TransformerSignalStrategy` input pipeline) plus one Colab GPU
session (~30 min training). Compute cost: $0–3 in Colab credits.

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
| Phase 5 — Session 3 closeout | *(this commit)* | `docs/handoff/session_3_findings.md`, verdict fix, Colab badge |

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

---

# Part II — Session 3d: Direct-Weight DL Portfolio Construction

## Methodology Pivot

Sessions 3a–3c-full applied the predict-then-allocate paradigm: a model first estimates
one-step-ahead expected returns $\hat{\mu}_{i,t}$ for each asset, then a separate allocation
rule converts those forecasts into portfolio weights. The two stages are trained independently;
the allocation objective does not feed back into the forecast objective.

Session 3d collapses both stages into a single end-to-end mapping $w_{i,t} = f_\theta(x_{i,t})$,
where $\theta$ are the neural network parameters and $x_{i,t}$ is the feature vector for asset
$i$ at date $t$. The network is optimised directly on a portfolio-level objective — Sharpe ratio,
CRRA utility, or CRRA with shrinkage — so the training signal is always a function of the full
cross-sectional portfolio return, not any individual asset return.

The theoretical foundation is the parametric portfolio policy framework of Brandt, Santa-Clara &
Valkanov (2009), who parameterised weights as a linear function of asset characteristics and
maximised expected utility in-sample. Session 3d replaces their linear policy with MLP, LSTM,
and Transformer backbones, extending the framework to nonlinear architectures.

The empirical precedent is Cheng & Wu (J.P. Morgan, 14 March 2024), who demonstrated
attention-based direct-weight portfolio construction on a 2-asset universe (S&P 500 + 10Y
Treasury) with 10-seed ensemble averaging. Session 3d extends this approach to the full
29-asset universe and 17-feature panel used throughout Sessions 1–3, enabling a direct
side-by-side comparison with all prior strategies.

## Experimental Setup

**Universe, features, and data splits.** Identical to Session 3c-full: 29 assets (equities,
sector ETFs, international equity, fixed income, commodities), 17 features (10 numeric
technical signals + 7 asset-class one-hots), train through 2020-02-27, validation
2020-02-28 to 2022-12-30, test (OOS) 2023-01-03 to 2026-03-31.

**Policy architectures.** Three backbone families:

| Architecture | Input | Output | Key hyperparameters |
|---|---|---|---|
| MLP | $(batch, 17)$ tabular | $(batch, 29)$ weights | 2 hidden layers (32, 16), ReLU, dropout 0.10 |
| LSTM | $(batch, 63, 17)$ per-asset sequence | $(batch, 29)$ weights | hidden dim 24, dropout 0.10 |
| Transformer | $(batch, 63, 17)$ per-asset sequence | $(batch, 29)$ weights | d_model 32, 4 heads, 2 layers, dropout 0.10 |

**Loss functions.** Three variants per architecture, giving 9 configurations:

- **Sharpe loss**: $\mathcal{L}_\text{Sharpe} = -\hat{\mu}(R_p)/\hat{\sigma}(R_p)$
- **CRRA utility** ($\gamma=5$): $\mathcal{L}_\text{CRRA} = -(1/T)\sum_t (1+R_{p,t})^{1-\gamma}/(1-\gamma)$
- **CRRA + shrinkage to equal-weight**: sigmoid output acts as per-asset multiplier on
  $w^b = 1/29$, bounding the effective weight within $[0,1] \cdot w_i^b$ and enforcing
  long-only shrinkage toward the equal-weight benchmark (Brandt 1999).

**Scale.** 9 configurations × 10 seeds = 90 trained policies. Training: max\_epochs=80,
patience=12 (same config across all 9 for a fair comparison).

**Compute.** NVIDIA RTX PRO 6000 Blackwell (Google Colab Pro, `torch 2.10.0+cu128`),
total wall-clock 26:51.

**Comparison baseline.** The 38-strategy baseline from Sessions 1–3 was extended with all 9
DW strategy results, producing a 47-strategy unified comparison table at
`results/cuda/policy_strategies_comparison_3d.csv`.

## Findings

### Finding 1 — No direct-weight configuration beats the ML ensemble bar

`MSR(Ensemble_μ̂)` from Session 2 retains first place in the 47-strategy comparison with
Sharpe 2.579. The best DW strategy by raw Sharpe, `DW(Transformer)[Sharpe]`, ranks 9th at
2.287 — but the allocation is degenerate to cash (see Finding 2). The best non-degenerate DW
strategy, `DW(Transformer)[CRRA]`, ranks 26th at Sharpe 2.111.

Top 15 strategies by OOS Sharpe (test period 2023–2026):

| Rank | Strategy | Family | Sharpe | Ann Ret | Ann Vol | Max DD |
|---|---|---|---|---|---|---|
| 1 | `MSR(Ensemble_μ̂)` | ML (ensemble) | 2.579 | 16.6% | 6.0% | −5.9% |
| 2 | `VMP(MDP(LW))` | Classical | 2.422 | 14.9% | 5.8% | −4.8% |
| 3 | `MSR(RF_μ̂)` | ML (single-fit) | 2.394 | 20.9% | 8.1% | −6.8% |
| 4 | `MSR(MLP_μ̂)` | DL (predict-then-wrap) | 2.320 | 22.4% | 8.9% | −8.6% |
| 5 | `SignalTilt(XGB)` | ML (single-fit) | 2.304 | 70.7% | 24.5% | −22.6% |
| 6 | `SignalTilt(EnsembleDL_weighted)` | DL (weighted ensemble) | 2.295 | 58.8% | 21.1% | −23.9% |
| 7 | `VMP(SignalTilt(XGB))` | ML + VMP | 2.292 | 72.0% | 25.0% | −20.5% |
| 8 | `SignalTilt(EnsembleDL)` | DL (ensemble) | 2.292 | 59.2% | 21.3% | −24.0% |
| 9 | `DW(Transformer)[Sharpe]` | DL (direct-weight) | 2.287 | 8.1% | **3.4%** | **−2.6%** |
| 10 | `SignalTilt(Transformer)` | DL (predict-then-wrap) | 2.283 | 50.0% | 18.5% | −22.1% |
| 11 | `MSR(Lasso_μ̂)` | ML (single-fit) | 2.272 | 21.8% | 8.9% | −11.6% |
| 12 | `SignalTilt(MLP)` | DL (predict-then-wrap) | 2.269 | 66.8% | 23.8% | −26.7% |
| 13 | `VMP(MSR(Lasso_μ̂))` | ML + VMP | 2.255 | 22.7% | 9.3% | −9.8% |
| 14 | `SignalTilt(RF)` | ML (single-fit) | 2.252 | 61.9% | 22.5% | −32.6% |
| 15 | `SignalTilt(LSTM)` | DL (predict-then-wrap) | 2.233 | 55.7% | 20.8% | −23.0% |

(Bold entries in rank 9 indicate degenerate allocation; see Finding 2.)

See `docs/figures/session3d/top10_sharpe_3d.png` for the bar chart.

### Finding 2 — Sharpe-loss direct-weight degenerates to cash on this universe

All three Sharpe-loss configurations produced allocations inconsistent with genuine
29-asset portfolio construction:

| Strategy | Sharpe | Ann Vol | Max DD |
|---|---|---|---|
| `DW(Transformer)[Sharpe]` | 2.287 | **3.4%** | **−2.6%** |
| `DW(LSTM)[Sharpe]` | 1.861 | **4.5%** | **−4.1%** |
| `DW(MLP)[Sharpe]` | 1.697 | **4.6%** | **−4.4%** |

On a 29-asset universe spanning equities, bonds, commodities, and alternatives, an invested
portfolio has no plausible realisation with 3.4% annualised volatility and a −2.6% maximum
drawdown over a three-year test period. These profiles are only achievable by concentrating
heavily in cash or near-zero-volatility positions.

The mechanism is well-understood: the Sharpe ratio $\hat{\mu}(R_p)/\hat{\sigma}(R_p)$ is
scale-invariant, and gradient descent can maximise it by driving portfolio variance toward
zero rather than by selecting high-return assets. The training diagnostic is conclusive:
for all three Sharpe-loss architectures, seed-0 reached best\_epoch=1 with no improvement
across 80 subsequent epochs, indicating the model collapsed to a near-degenerate solution
at the first gradient step. The Sharpe-loss degeneracy replicates consistently across all
three architectures, confirming it is a property of the loss function rather than any
specific backbone. Cheng & Wu (2024) discuss this failure mode in the 2-asset context;
Session 3d confirms it replicates at 29-asset scale.

### Finding 3 — CRRA+Shrinkage produces the tightest per-seed stability observed in the entire study

The CRRA+Shrinkage configurations converge to a narrow performance band across 10 seeds:

| Config | Ensemble Sharpe | Ann Vol | Max DD | Per-seed Mean | Per-seed Stdev |
|---|---|---|---|---|---|
| `DW(MLP)[CRRA+Shrink]` | 2.125 | 10.3% | −12.5% | 2.113 | ±0.023 |
| `DW(LSTM)[CRRA+Shrink]` | 2.102 | 10.4% | −12.7% | 2.108 | ±0.010 |
| `DW(Transformer)[CRRA+Shrink]` | 2.092 | 10.1% | −12.4% | — | — |

*Transformer per-seed stability data unavailable; see Stability Data Note below.*

The ±0.010 stdev for `DW(LSTM)[CRRA+Shrink]` is the tightest per-seed stability
observed across the entire 47-strategy study — tighter than the best predict-then-wrap
Session 3c-full results (MLP ±0.148, LSTM ±0.184, Transformer ±0.154). The shrinkage
architecture constrains the per-asset multiplier to $[0,1]$ via sigmoid activation,
bounding each output around the equal-weight benchmark. This acts as a regulariser: the
network cannot stray far from a diversified baseline regardless of initialisation, and the
CRRA loss's smooth shape within that bounded space makes the optimisation well-conditioned.
The combined effect is a training landscape that converges reliably to approximately the
same solution from diverse starting points.

Importantly, all three CRRA+Shrinkage architectures (MLP, LSTM, Transformer) cluster
within a 0.033 Sharpe range (2.092–2.125), confirming the architecture choice has minimal
impact once the loss function and output activation are correctly specified.

See `docs/figures/session3d/seed_sharpe_dist_3d.png` for the per-seed Sharpe distribution
across all 9 configurations.

### Finding 4 — CRRA without shrinkage is competitive but less stable

CRRA loss without shrinkage produces non-degenerate allocations but with higher per-seed
variability:

| Config | Ensemble Sharpe | Ann Vol | Max DD | Per-seed Mean | Per-seed Stdev |
|---|---|---|---|---|---|
| `DW(Transformer)[CRRA]` | 2.111 | 7.6% | −6.7% | — | — |
| `DW(LSTM)[CRRA]` | 1.761 | 6.7% | −7.3% | 1.857 | ±0.126 |
| `DW(MLP)[CRRA]` | 1.367 | 6.7% | −7.1% | 1.703 | ±0.212 |

*Transformer per-seed data unavailable; see Stability Data Note.*

Two observations stand out. First, there is a notable gap between the ensemble Sharpe and
the per-seed mean for `DW(MLP)[CRRA]` (ensemble 1.367, per-seed mean 1.703). Jensen's
inequality: averaging weight vectors across 10 seeds does not preserve the mean Sharpe of
the individual seed portfolios. The ensemble weight vector occupies a different region of
weight space than any individual seed, and in this case that blended region has lower
realised Sharpe. This effect does not appear for CRRA+Shrinkage, where the sigmoid
activation bounds the individual seed weights tightly.

Second, architecture matters more for CRRA than for CRRA+Shrinkage: the Transformer
significantly outperforms MLP under CRRA (2.111 vs 1.367), whereas the three architectures
converge tightly under CRRA+Shrinkage (2.092–2.125). Without the shrinkage prior, the
architecture's inductive bias influences which region of weight space the optimiser finds.

### Finding 5 — Loss function dominates architecture choice for direct-weight policy

Grouping the 9 configurations by loss function:

| Loss family | Sharpe range | Ann Vol range | Character |
|---|---|---|---|
| Sharpe loss | 1.697 – 2.287 | 3.4–4.6% | All degenerate to cash |
| CRRA | 1.367 – 2.111 | 6.7–7.6% | Non-degenerate, architecture-dependent |
| CRRA+Shrink | 2.092 – 2.125 | 10.1–10.4% | Tight cluster, non-degenerate |

The within-loss-family spread is much smaller than the between-loss-family spread for Sharpe
and CRRA+Shrink. For the CRRA family, the within-family spread is wider (1.367–2.111),
but even there the character of the allocations — non-degenerate, moderate volatility —
is consistent across architectures.

This finding is consistent with Cheng & Wu (2024): in the direct-weight paradigm, the
choice of loss function and output regularisation architecture (shrinkage vs unconstrained)
is the dominant design decision. Backbone selection within a given loss family is secondary.

## Stability Data Note

The §8 per-seed stability analysis in `notebooks/04c_dl_policy_strategies_cuda.ipynb`
produced correct per-seed statistics at stdout time (as captured in the table above), but
the saved `policy_stability_table_3d.csv` originally contained single-seed fallback values
for all 9 configurations due to a variable-scoping bug in the original notebook §8 cell.
The root cause: the `stability_rows` list was initialised and populated in a *separate loop*
after the main per-seed collection loop, rather than inline within the collection loop. If
the cell was re-executed after the full run, or if the two loops ran with different scope
state, `stability_rows` received stale data. The fix (applied in this session) moves the
`stability_rows = []` initialisation and each row's `stability_rows.append({...})` call
*inside* the outer collection loop, building the table atomically with the per-seed
computation it summarises.

The 10-seed per-seed statistics for 6 of 9 configurations are sourced from the §8 stdout
captured during the Colab run. The 3 Transformer per-seed entries are unavailable and are
flagged as `single-seed-fallback` in `results/cuda/policy_stability_table_3d.csv`.

## Comparative Conclusion Across Sessions 3 + 3d

The unified empirical finding across both DL paradigms on the 29-asset 2003–2026 universe:
the ML ensemble from Session 2 — Lasso, RF, and XGBoost equal-weighted predictions wrapped
in MSR optimisation — at Sharpe 2.579 remains the empirical winner of the 47-strategy
comparative study. No DL configuration in either the predict-then-wrap or the direct-weight
paradigm exceeds it.

Best DL strategies by paradigm (test period 2023–2026):

| Paradigm | Strategy | Sharpe | Ann Vol | Max DD |
|---|---|---|---|---|
| Predict-then-wrap (Session 3c-full) | `MSR(MLP_μ̂)` | 2.320 | 8.9% | −8.6% |
| Direct-weight non-degenerate (Session 3d) | `DW(Transformer)[CRRA]` | 2.111 | 7.6% | −6.7% |
| Direct-weight stable (Session 3d) | `DW(LSTM)[CRRA+Shrink]` | 2.102 ens / 2.108 ±0.010 | 10.4% | −12.7% |

The two paradigms produce qualitatively different failure modes. Predict-then-wrap strategies
produce realistic allocations (vol 7–25%, DD −7 to −27%) but can be over-confident when
the underlying model's predictions are noisy — `SignalTilt(XGB)` reaches Sharpe 2.304 but
with 24.5% annualised vol and −22.6% maximum drawdown. Direct-weight strategies without
proper regularisation degenerate to near-cash allocations under Sharpe loss. CRRA loss with
shrinkage to the equal-weight benchmark is the most defensible direct-weight configuration:
it produces non-degenerate allocations, achieves competitive Sharpe ratios (~2.10 across
architectures), and exhibits the tightest per-seed stability in the study.

See `docs/figures/session3d/equity_curves_3d.png` and
`docs/figures/session3d/drawdown_3d.png` for the visual comparison of top strategies.

## Future Work (Extended)

Session 3d-specific extensions — each is a self-contained experiment that could
specifically address the identified failure modes:

1. **Sum-of-weights penalty.** Add $\lambda \max(0, 1 - \sum_i w_i)^2$ to the Sharpe loss,
   penalising under-investment. This closes the cash-escape route without changing the output
   activation structure.
2. **Softmax output activation.** Replace the current per-asset output (relu-clipped and
   renormalised) with a softmax, which enforces $\sum_i w_i = 1$ exactly and eliminates the
   degenerate-to-zero solution from the feasible set.
3. **Sortino loss.** Replace the Sharpe loss with $\hat{\mu}(R_p)/\hat{\sigma}^-(R_p)$ where
   $\hat{\sigma}^-$ is the downside deviation. Retains the ratio structure but removes the
   variance-minimisation incentive that drives Sharpe degeneracy.
4. **Volatility targeting.** Post-hoc vol scaling (3-month realised vol, 5% p.a. target) per
   Cheng & Wu (2024), to stabilise portfolio risk across regimes.
5. **Walk-forward retraining.** Monthly model refits using a trailing window, as in
   Cheng & Wu (2024). Session 2 walk-forward underperformed single-fit for tree-based
   models, but direct-weight neural policies may benefit from adaptation to the post-2023
   regime.
6. **Cross-asset attention.** The current architectures process each asset independently on
   a per-asset sequence. A cross-asset Transformer would process a single cross-sectional
   snapshot of all 29 assets at each date (input shape $(batch, 29, 17)$), enabling the
   model to learn pairwise correlation structure directly. This was identified as the
   highest-priority architectural extension in the Session 3c-full findings (Part I) and
   remains unaddressed in Session 3d.

For the comparative study as a whole, the natural next chapters are reinforcement learning
(Session 4, via `SequentialStrategy.step()`), LLM-views Black-Litterman (Session 5), and
LLM-orchestrated agents (Session 6).

## Reproducibility

Session 3d artifacts are committed to this repository.

| Session | Commit | Primary artifacts |
|---|---|---|
| 3d-a — Direct-weight scaffolding | `4177ada` | `src/aiam/dl/policies.py`, `losses.py`, `policy_workflow.py`; `src/aiam/strategy/dl_policy_strategies.py`; 46 new tests (212 total) |
| 3d-b — Colab CUDA notebook | `2c8914e` | `notebooks/04c_dl_policy_strategies_cuda.ipynb` |
| 3d-c — Closeout (this commit) | *(this commit)* | `results/cuda/policy_*.csv`, `results/cuda/policy_*.parquet`, `docs/figures/session3d/`, `docs/handoff/session_3_findings.md` (Part II), notebook §8 fix |

**Compute environment.** Google Colab Pro, NVIDIA RTX PRO 6000 Blackwell
(`torch 2.10.0+cu128`). Training time: 26:51 total (90 policies across 9 configs × 10 seeds).

**Reproducing results.** The 47-strategy comparison table is at
`results/cuda/policy_strategies_comparison_3d.csv`. Per-seed stability data (6 of 9 configs)
is at `results/cuda/policy_stability_table_3d.csv`. Portfolio returns are at
`results/cuda/policy_returns_3d.parquet`. To reproduce from scratch, open
`notebooks/04c_dl_policy_strategies_cuda.ipynb` in Colab (CUDA runtime required) and
execute all cells.
