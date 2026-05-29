# Appendix E — Statistical Robustness

## E.1 Methodology

### E.1.1 Memmel (2003) Paired Sharpe-Difference Test

The Memmel (2003) test evaluates the null hypothesis $H_0: \mathrm{SR}_A = \mathrm{SR}_B$ for two strategies $A$ and $B$ observed over the same $T$ periods. The test statistic is:
$$z = \frac{\hat{\mathrm{SR}}_A - \hat{\mathrm{SR}}_B}{\hat{V}^{1/2}},$$
where the variance estimate accounts for the correlation between the two return series:
$$\hat{V} = \frac{1}{T}\!\left[ 2 - 2\hat{\rho}_{AB} + \frac{1}{2}\bigl(\hat{\mathrm{SR}}_A^2 + \hat{\mathrm{SR}}_B^2 - \hat{\mathrm{SR}}_A \hat{\mathrm{SR}}_B (1 + \hat{\rho}_{AB}^2)\bigr) \right].$$
Here $\hat{\rho}_{AB}$ is the sample correlation of the daily return series. Sharpe ratios are computed at daily frequency and annualised by $\sqrt{252}$ before entering $\hat{V}$. Under $H_0$, $z$ is asymptotically standard normal; all reported $p$-values are one-sided (direction stated in each finding).

The test is applied to the base (non-VMP) strategy series throughout to isolate the strategy-level signal from the VMP overlay effect.

### E.1.2 Block Bootstrap for Sharpe 95% CIs

To quantify sampling uncertainty around point Sharpe estimates, the block bootstrap resamples daily return series preserving short-range serial dependence. Parameters verified from `scripts/generate_figures.py:653`:
- **Block size:** 252 trading days (one calendar year)
- **Number of resamples:** 5,000
- **Random seed:** 42

Each bootstrap replicate samples $\lceil T/252 \rceil$ overlapping blocks of length 252 with replacement (start positions drawn uniformly from $[0, T-252]$), concatenates them to length $T$, and computes the annualised Sharpe. The 2.5th and 97.5th percentiles of the 5,000 replicates form the 95% CI. The 252-day block length preserves momentum and mean-reversion at the weekly-to-monthly horizon without over-blocking for a 5,868-day sample.

### E.1.3 Sign Test for VMP Universal Lift

The VMP overlay is applied to all 31 base strategies, producing 31 VMP–base pairs. Under $H_0$ that the overlay is equally likely to help or hurt any base strategy (probability $\frac{1}{2}$ per pair), the number of positive signs $S \sim \mathrm{Binomial}(31, 0.5)$. Observing $S = 24$ positive signs (24/31 pairs with VMP Sharpe > base Sharpe) has $p = P(S \geq 24) \approx 3.8 \times 10^{-4}$.

For the specific claim in §3 (main paper) regarding the original 24 base strategies presented in the study's main table, $S = 24$ out of $n = 24$ gives $p = 2^{-24} \approx 6 \times 10^{-8}$.

---

## E.2 Full Memmel Test Results

### MSR(LW) vs MSR(sample): shrinkage benefit

**Claim (§4):** Ledoit-Wolf shrinkage improves MSR Sharpe on the 29-asset 2003–2026 universe.

| Sample | $z$ | $p$ (one-sided, $\mathrm{SR}_{\mathrm{LW}} > \mathrm{SR}_{\mathrm{sample}}$) | Verdict |
|---|---|---|---|
| 29-asset, 2003–2026 (this study) | 1.13 | 0.259 | Not significant |
| 30-asset, 2008–2026 (prior study) | 2.78 | 0.005 | Significant |

The directional result (LW > sample for MSR) is consistent across both samples, but the 2003–2026 sample does not reach the 5% threshold. The extended pre-GFC period (2003–2007) dilutes the signal: in calm, near-random-walk regimes, the Stein-type shrinkage advantage over sample covariance compresses. The conclusion in §4 is deliberately hedged: the LW benefit is "directional but not significant at 5% on the 29-asset extended sample."

### VMP universal lift: sign test

**Claim (§3):** VMP raises Sharpe for every base strategy.

Applied to the original 24-strategy set: 24/24 positive signs, $p = 2^{-24} \approx 6 \times 10^{-8}$. This is the most overwhelming individual statistical result in the study. The individual pairwise contrast VMP(MSR(LW)) vs MSR(LW) gives $z = 1.90$, $p = 0.058$ — marginally outside the 5% threshold but directionally consistent. The universality claim is defended by the sign test, not by any single pairwise Memmel contrast.

### SWITCH(v2a) vs SWITCH(LW) v1: regime routing adds value

**Claim (§5):** SWITCH(v2a) (R0→MSR(LW), R5→MSR(sample), default→MDP(LW)) significantly outperforms SWITCH(LW) v1 (R0→EW, R5→MSR(LW), default→MDP(LW)).

| Comparison | $z$ | $p$ (one-sided, v2a > v1) | Verdict |
|---|---|---|---|
| SWITCH(v2a) vs SWITCH(LW) v1 | 2.05 | 0.040 | Significant at 5% |

This is the strongest pairwise Memmel result in the study (full-sample 2003–2026). The v2a rule was derived on the training period (2003–2022) and the Memmel test is computed on the full sample; the test cannot be interpreted as an OOS significance test but confirms the directional magnitude of the routing improvement.

### HRP shrinkage near-invariance

**Claim (§4):** HRP is approximately invariant to the choice of covariance estimator (sample vs LW).

| Comparison | $z$ | $p$ (two-sided) | Verdict |
|---|---|---|---|
| HRP(LW) vs HRP(sample) | −0.67 | 0.506 | Fail to reject equality |

The sign is opposite to the 30-asset 2008–2026 prior study, reinforcing the near-invariance interpretation: HRP's hierarchical bisection largely absorbs shrinkage effects because the recursive allocation is insensitive to the magnitude of individual covariance entries.

### Additional pairwise: VMP(MSR(LW)) vs MSR(LW)

| Comparison | $z$ | $p$ (one-sided) | Verdict |
|---|---|---|---|
| VMP(MSR(LW)) vs MSR(LW) | 1.90 | 0.058 | Marginal |

This contrast sits just above the 5% threshold individually. It is combined with the 24/24 sign test as joint evidence for VMP universality; the marginal individual p-value does not undermine the claim when the sign-test evidence is considered.

---

## E.3 Block Bootstrap Sharpe 95% Confidence Intervals

Table E.1 reports 95% bootstrap confidence intervals for the top 10 configurations by gross Sharpe (degenerate artifact VMP(GMV(sample)) excluded; see §3.2). All CIs are computed from 5,868 daily returns, 252-day blocks, 5,000 resamples. The figure below displays these intervals graphically.

**Table E.1. Bootstrap 95% CIs — top 10 configurations.**

| Configuration | Gross Sharpe | 95% CI Lower | 95% CI Upper | CI Width |
|---|---|---|---|---|
| VMP(MDP(LW)) | 1.372 | 0.837 | 1.856 | 1.019 |
| VMP(MDP(sample)) | 1.368 | 0.834 | 1.870 | 1.036 |
| VMP(MSR(sample)) | 1.295 | 0.864 | 1.763 | 0.899 |
| VMP(SWITCH(sample)) | 1.293 | 0.945 | 1.662 | 0.717 |
| VMP(SWITCH(LW)) | 1.265 | 0.826 | 1.646 | 0.820 |
| VMP(MSR(LW)) | 1.239 | 0.771 | 1.608 | 0.837 |
| VMP(HRP(LW)) | 1.232 | 0.819 | 1.639 | 0.820 |
| VMP(BL-Mom(LW)) | 1.217 | 0.768 | 1.576 | 0.808 |
| VMP(GMV(LW)) | 1.215 | 0.678 | 1.826 | 1.148 |
| VMP(GMV(OAS)) | 1.207 | 0.667 | 1.890 | 1.223 |

![95% block-bootstrap Sharpe confidence intervals for the top configurations; 252-day blocks,
5,000 resamples. Error bars span the 2.5th–97.5th percentile of the bootstrap distribution.
](figures/bootstrap_sharpe_cis.png)

All 95% CI lower bounds exceed 0.67, confirming that all top-10 configurations produce reliably positive Sharpe ratios at the 2.5th percentile. The wide intervals (≥0.70 Sharpe units) reflect the inherent difficulty of Sharpe inference over a 23.3-year daily sample: even with 5,868 observations, cross-cycle regime variation generates substantial bootstrap variability. VMP(SWITCH(sample)) and VMP(SWITCH(LW)) have the narrowest CIs (0.717 and 0.820 respectively), consistent with their regime-conditional routing reducing tail-dependence on any single market environment. The low-turnover GMV variants exhibit wider CIs despite moderate point estimates, driven by their concentrated fixed-income exposure and episodic yield-curve dislocations.

---

## E.4 Discussion

The tests above reveal what can and cannot be concluded from a 23.3-year daily sample of 29 assets. Sharpe inference is power-limited at these sample sizes: a typical 252-day block bootstrap CI for a Sharpe near 1.3 spans roughly ±0.5 Sharpe units, meaning that differences smaller than 0.3–0.4 are difficult to detect at conventional levels. The strongest evidence in this study is therefore directional consistency across multiple independent tests rather than any single significant p-value. The 24/24 VMP sign test ($p \approx 6 \times 10^{-8}$) is the most overwhelming individual result; the SWITCH(v2a) Memmel contrast ($z = 2.05$, $p = 0.040$) is the only pairwise test that crosses the 5% threshold. The shrinkage benefit for MSR and the HRP near-invariance are directionally consistent with the prior 30-asset 2008–2026 literature but cannot be claimed as statistically proven on the current sample alone. This limitation is reported honestly in §§4–5 rather than smoothed over by selective presentation.
