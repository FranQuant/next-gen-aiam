# Notebook 05 Findings — Direct-Weight DL Portfolio Construction (JPM-Spec Replication)

## Status: Complete

Notebook 05 ran on Google Colab (Blackwell GPU), executing 3,960 policy fits (18 configs × 10 seeds
× 22 walk-forward refit windows) in approximately 40 minutes of wall time. The headline result is a
clean negative: no DL configuration exceeded the simple Risk Parity benchmark (Sharpe 1.247) or
the Equal-Weight 50/50 benchmark (Sharpe 1.254) on SPY+IEF over the 2024-07 to 2026-04 OOS window.
The best DL strategy (`LSTM_dailyrp_crra_shrinkage`, Sharpe 1.240) is statistically indistinguishable
from Risk Parity (gap: −0.008). Realized power features had a mixed effect averaging to essentially
zero (+0.008 Sharpe mean across 9 matched arch×loss pairs, 4 positive / 5 negative). The JPM 2024
(Cheng & Wu) reported Sharpe advantage over risk parity does not reproduce on SPY+IEF with this
methodology.

---

## Experimental Setup

| Dimension | Setting |
|---|---|
| Universe | SPY.US + IEF.US (equity + 10-year Treasury ETF) |
| Full period | 2021-01-04 → 2026-04-30 |
| Train | 2021-01 → 2023-06 (rolling 24-month initial window) |
| Validation | 6 months following each train window |
| Test (OOS) | 2024-07 → 2026-04 (~460 trading days, 22 months) |
| Refit cadence | Monthly walk-forward; 22 refit windows total |
| Feature variants | Daily-only (6 features) vs Daily+realized-power (7 features) |
| Architectures | MLP, LSTM, Transformer |
| Loss functions | Sharpe, CRRA-γ5, CRRA+Shrinkage-to-risk-parity |
| Seeds | 10 per config |
| Total fits | 18 configs × 10 seeds × 22 refits = 3,960 |
| Compute | Google Colab Pro, Blackwell GPU |
| Framework | Brandt, Santa-Clara & Valkanov (2009) parametric portfolio policies |

**Methodology alignment with Cheng & Wu (2024):** feature spec (raw return history + realized power
lags), walk-forward refit with 24-month rolling window, monthly cadence, and 10-seed averaging are
all matched to the JPM 2024 methodology. The key untested differences are instrument (ETFs vs
ES/TY futures), OOS window length (~22 months vs ~7 years), overnight sessions (excluded vs
included for futures), and vol targeting (not applied here).

---

## Findings

### Finding 1 — No DL configuration exceeds simple benchmarks

The two benchmarks dominate the 21-strategy comparison. EW-50/50 (Sharpe 1.254) ranks first and
Risk Parity-21d (Sharpe 1.247) ranks second. The top-10 OOS Sharpe results are:

| Rank | Strategy | Arch | Features | Loss | Sharpe |
|---|---|---|---|---|---|
| 1 | EW-50/50 | Benchmark | — | — | 1.254 |
| 2 | RiskParity-21d | Benchmark | — | — | 1.247 |
| 3 | LSTM_dailyrp_crra_shrinkage | LSTM | daily+RP | CRRA+Shrink | 1.240 |
| 4 | MLP_daily_crra_shrinkage | MLP | daily-only | CRRA+Shrink | 1.239 |
| 5 | LSTM_daily_crra_shrinkage | LSTM | daily-only | CRRA+Shrink | 1.237 |
| 6 | MLP_dailyrp_crra_shrinkage | MLP | daily+RP | CRRA+Shrink | 1.237 |
| 7 | 60/40 | Benchmark | — | — | 1.197 |
| 8 | MLP_dailyrp_sharpe | MLP | daily+RP | Sharpe | 1.131 |
| 9 | Transformer_daily_crra_shrinkage | Transformer | daily-only | CRRA+Shrink | 1.127 |
| 10 | Transformer_dailyrp_crra_shrinkage | Transformer | daily+RP | CRRA+Shrink | 1.125 |

The gap from the best DL (Sharpe 1.240) to Risk Parity (1.247) is −0.008. Note that 60/40
static (1.197) ranks only 7th on this period; EW-50/50 dominates 60/40 due to IEF's strong
contribution in the test window.

See `results/notebook_05/figures/top_strategies_05.png`.

### Finding 2 — Realized power feature has mixed, essentially neutral effect

For each of the 9 matched arch×loss pairs, ΔSharpe(daily+RP − daily-only):

| Arch | Loss | ΔSharpe |
|---|---|---|
| MLP | Sharpe | +0.130 |
| MLP | CRRA | +0.057 |
| MLP | CRRA+Shrink | −0.002 |
| LSTM | Sharpe | +0.012 |
| LSTM | CRRA | −0.007 |
| LSTM | CRRA+Shrink | +0.002 |
| Transformer | Sharpe | −0.075 |
| Transformer | CRRA | −0.043 |
| Transformer | CRRA+Shrink | −0.003 |
| **Mean** | | **+0.008** |

Four of nine pairs favor daily+RP; five favor daily-only. The mean delta is +0.008, driven
primarily by MLP_sharpe (+0.130), but that config is an unstable outlier (10-seed stdev 0.356
for daily+RP vs 0.267 for daily-only). Among the stable CRRA+Shrinkage configs, the realized
power feature produces changes of ±0.003 — effectively zero. Among Transformer with unshrunk
losses the effect is slightly negative (−0.043 to −0.075).

The central question — does intraday realized power improve OOS Sharpe? — is answered: **not
consistently and not materially on SPY+IEF with 22 months of OOS data**.

See `results/notebook_05/figures/feature_ablation_05.png`.

### Finding 3 — Walk-forward refit did not change the qualitative conclusion

An earlier smoke-test version of Notebook 05 (commit `4336479`, single-fit, 1 seed × 5 epochs)
found best DL Sharpe ≈ 1.16 vs Risk Parity ≈ 1.17 — a small negative gap. The full
walk-forward JPM-spec run (10 seeds × 22 refits) reaches best DL Sharpe 1.240 vs Risk
Parity 1.247. Absolute levels improved with proper training, but the DL position relative to
the benchmark did not: in both versions, DL strategies trail simple benchmarks by a margin
consistent with noise. Walk-forward refit is not the missing ingredient.

### Finding 4 — CRRA+Shrinkage produces tightest per-seed stability (replicates Session 3d finding)

Per-seed Sharpe stability (10 seeds) for all six CRRA+Shrinkage configs:

| Config | Arch | Features | Mean Sharpe | Stdev | Min | Max |
|---|---|---|---|---|---|---|
| LSTM_dailyrp_crra_shrinkage | LSTM | daily+RP | 1.240 | 0.005 | 1.232 | 1.248 |
| MLP_daily_crra_shrinkage | MLP | daily-only | 1.239 | 0.008 | 1.227 | 1.256 |
| LSTM_daily_crra_shrinkage | LSTM | daily-only | 1.237 | 0.005 | 1.229 | 1.244 |
| MLP_dailyrp_crra_shrinkage | MLP | daily+RP | 1.237 | 0.009 | 1.223 | 1.248 |
| Transformer_daily_crra_shrinkage | Transformer | daily-only | 1.123 | 0.030 | 1.077 | 1.167 |
| Transformer_dailyrp_crra_shrinkage | Transformer | daily+RP | 1.120 | 0.031 | 1.060 | 1.160 |

All six CRRA+Shrinkage configs have per-seed stdev ≤ 0.031. MLP and LSTM shrinkage configs
achieve stdev of 0.005–0.009. This directly replicates the Session 3d result on the 29-asset
universe, where CRRA+Shrinkage was the only loss function with tight cross-seed stability. The
shrinkage mechanism is a robust implicit regularizer regardless of universe size or methodology.

### Finding 5 — Unshrunk losses have severe per-seed instability

The 12 non-shrinkage configs (Sharpe and CRRA losses) show dramatically higher per-seed variance:

| Metric | CRRA+Shrink (6 configs) | Non-shrink (12 configs) |
|---|---|---|
| Stdev range | 0.005 – 0.031 | 0.109 – 0.486 |
| Min seed Sharpe range | 1.060 – 1.256 | −0.404 – 1.035 |

The worst cases are `MLP_dailyrp_crra` (stdev 0.486, min seed Sharpe −0.404) and
`MLP_dailyrp_sharpe` (stdev 0.356, min seed Sharpe −0.016). Direct-weight Sharpe and CRRA
losses without regularization admit degenerate solutions that some random seeds converge to
and others avoid. In a production setting, any non-shrinkage direct-weight policy requires
either weight constraints or softmax output normalization to avoid this degeneracy.

See `results/notebook_05/figures/seed_sharpe_dist_05.png`.

### Finding 6 — Weight time series reveals CRRA+Shrink allocations closely track the benchmark

The Transformer weight time series panels in `results/notebook_05/figures/weight_timeseries_05.png`
illustrate the loss function's effect on allocation behavior. Sharpe-loss Transformer configs
exhibit visible time variation in SPY/IEF weights. CRRA-loss configs show moderate variation.
CRRA+Shrink configs produce nearly flat allocations hovering close to risk-parity weighting
throughout the test period.

This explains the stability finding mechanically: the shrinkage prior dominates the learned
signal, causing the policy to remain near the risk-parity benchmark regardless of seed or
realized data. The tight Sharpe and low drawdown of CRRA+Shrink configs follow directly from
this near-benchmark behavior, not from superior signal extraction.

### Finding 7 — Loss function explains more variance in OOS Sharpe than architecture

Grouping by loss function across all arch × feature combinations:

| Loss | OOS Sharpe range | Mean |
|---|---|---|
| CRRA+Shrink | 1.125 – 1.240 | 1.201 |
| CRRA | 0.937 – 1.041 | 0.996 |
| Sharpe | 0.914 – 1.131 | 0.986 |

Within each loss family, architecture (MLP / LSTM / Transformer) and feature variant (daily-only
vs daily+RP) explain modest additional variance. Choosing CRRA+Shrinkage over either alternative
is the single most impactful design decision in the BSV-style direct-weight framework on this
universe.

---

## Reconciling with Cheng & Wu (2024)

Cheng & Wu (JPM 2024) reported approximately Sharpe 0.85 for their best DL direct-weight policy
versus a risk parity benchmark at approximately Sharpe 0.63 on ES+TY futures — a gap of roughly
+0.22 in favor of DL. On SPY+IEF with our methodology, the gap is −0.008 in favor of the
benchmark.

**Methodology dimensions tested — and found not to be the explanation:**

| Dimension | This study | Cheng & Wu (2024) | Matched? |
|---|---|---|---|
| Feature spec (raw return + RP history) | 252-day raw history, 1-day lag | Same | Yes |
| Walk-forward refit window | 24 months rolling | 24 months rolling | Yes |
| Refit cadence | Monthly | Monthly | Yes |
| Seed averaging | 10 seeds | 10 seeds | Yes |
| Training window length | 24 months | 24 months | Yes |
| BSV parametric policy framework | Yes | Yes | Yes |

**Methodology dimensions not tested — remaining future work:**

| Dimension | This study | Cheng & Wu (2024) |
|---|---|---|
| Instruments | SPY/IEF ETFs | ES/TY futures |
| OOS window length | ~22 months | ~7 years |
| Trading hours | Regular hours only | 24-hour futures sessions |
| Vol targeting | None | 5% annual vol target post-process |

Three of the untested dimensions plausibly explain the gap:

1. **Instruments and overnight signal.** ES and TY futures trade 24 hours. Overnight moves —
   particularly gap opens following macro events — are a primary channel for intraday-derived
   signals. SPY and IEF ETFs trade only during regular session hours (09:30–16:00 ET); realized
   power computed from 5-minute bars excludes all overnight price action. If Cheng & Wu's signal
   derives primarily from the overnight session, it is structurally absent from our feature set.

2. **OOS window length.** Seven years of OOS observations versus 22 months. Statistical power
   to detect a Sharpe lift of 0.22 over a benchmark requires substantially more data than 22
   months; our test window may be too short to distinguish a real edge from noise.

3. **Vol targeting.** A 5% annual vol target applied as a post-process scales allocations
   during low-vol periods and scales them down otherwise. This scaling can generate meaningful
   Sharpe lift independently of the DL signal quality.

The most likely single explanation is the combination of instrument-specific overnight signal
(absent in ETF data) and OOS window length (too short to detect small edges).

---

## Contribution to the Literature

**1. Methodology-controlled non-replication.** Six specific Cheng & Wu (2024) methodology
choices were matched and found insufficient to reproduce the reported Sharpe lift on SPY+IEF.
This narrows the set of plausible explanations to the four untested dimensions above.

**2. Cross-universe consistency of the CRRA+Shrinkage finding.** The shrinkage loss variant
produces tight per-seed stability (stdev ≤ 0.031) and near-benchmark allocations in both the
Session 3d 29-asset experiment and this 2-asset experiment. The finding is robust across
universe size, methodology variant, and instrument class. CRRA+Shrinkage is the most
reproducible and interpretable BSV-style loss specification studied to date.

**3. Loss function primacy.** On both universe sizes, loss function choice explains more
cross-configuration Sharpe variance than architecture or feature engineering. Practitioners
designing BSV-style DL portfolios should invest in loss specification before architecture search.

---

## Limitations and Future Work

1. **Futures replication.** The obvious follow-up is to run the identical methodology on
   ES + TY futures, which match the Cheng & Wu instrument choice and include overnight sessions.
   This is the test most likely to isolate whether the overnight signal explains the gap.

2. **Short OOS window.** 22 months limits statistical power. A Sharpe lift of 0.10–0.20 over
   a risk parity benchmark is undetectable at conventional significance levels with fewer than
   3 years of data.

3. **No vol targeting.** Adding a 5% annual vol target post-process (per Cheng & Wu) would
   scale any strategy's Sharpe uniformly; it may also change relative ordering.

4. **Sharpe loss instability.** A sum-of-weights penalty or softmax output layer would
   eliminate degenerate solutions without changing the objective substantially.

5. **Architecture limits at 2 assets.** The representational capacity required to learn
   meaningful departures from a 2-asset risk parity benchmark may exceed what MLP/LSTM/Transformer
   can learn given the amount of training data available.

---

## Reproducibility

| Artifact | Path |
|---|---|
| Notebook | `notebooks/05_dl_portfolio_construction_exploration.ipynb` |
| Comparison table (21 rows) | `results/notebook_05/comparison.csv` |
| Stability table (18 rows) | `results/notebook_05/stability.csv` |
| Config timing (18 rows) | `results/notebook_05/config_progress.csv` |
| Portfolio returns (460 × 21) | `results/notebook_05/strategy_returns.parquet` |
| Per-config weights (18 files) | `results/notebook_05/weights_*.parquet` |
| Equity curves | `results/notebook_05/figures/equity_curves_05.png` |
| Drawdown | `results/notebook_05/figures/drawdown_05.png` |
| Weight time series | `results/notebook_05/figures/weight_timeseries_05.png` |
| Seed Sharpe distribution | `results/notebook_05/figures/seed_sharpe_dist_05.png` |
| Feature ablation | `results/notebook_05/figures/feature_ablation_05.png` |
| Top strategies | `results/notebook_05/figures/top_strategies_05.png` |
| Intraday bars (published CSV) | `data/published/intraday_5min_SPY_IEF_2021_2026.csv` |
| Intraday bars (cached parquet) | `data/cache/intraday_5min_SPY_IEF_2021_2026.parquet` (gitignored) |
| Policy infrastructure | `src/aiam/dl/losses.py`, `policies.py`, `policy_workflow.py` |
| Strategy wrapper | `src/aiam/strategy/dl_policy_strategies.py` |
| Intraday data module | `src/aiam/data/intraday.py` |
| Realized power features | `src/aiam/features/realized_power.py` |

To reproduce: open `notebooks/05_dl_portfolio_construction_exploration.ipynb` in Google Colab
with a GPU runtime and execute all cells. The notebook fetches the published intraday CSV
directly and writes results to `results/notebook_05/`.

---

## Academic References

**Primary framework:**
Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009). Parametric portfolio policies:
Exploiting characteristics in the cross-section of equity returns. *Review of Financial
Studies*, 22(9), 3411–3447.

**Supporting — return characteristics and portfolio choice:**
Brandt, M. W. (1999). Estimating portfolio and consumption choice: A conditional Euler
equations approach. *Journal of Finance*, 54(5), 1609–1645.

**Replication target:**
Cheng, H., & Wu, Y. (2024). Deep learning portfolio construction with intraday features.
*Journal of Portfolio Management*, 50(5).

---

## Status in the Comparative Paper

The DL track is complete. The main comparative paper's empirical core remains the 38-strategy
comparison on 29 assets (Sessions 1, 1.5B, 2, 3a-3c-full), with `MSR(Ensemble_μ̂)` at Sharpe
2.579 as the empirical winner. Notebook 05 is a focused methodology exploration — a 2-asset
replication study — that sits alongside the main comparison rather than competing with it.

The Notebook 05 result strengthens the paper's implicit claim that simple classical benchmarks
(risk parity, equal weight) are hard to beat: even with 3,960 policy fits, intraday features,
and a faithful replication of a published JPM methodology, a BSV direct-weight policy cannot
materially improve on a 21-day volatility-weighted risk parity allocation over a 22-month
test window on SPY+IEF.

**Next: Session 4 — Reinforcement Learning.** `notebooks/06_rl_strategies.ipynb`, PPO/SAC
agents via `SequentialStrategy.step()`.
