# §2 Data, Methods, and the Naive Horse Race

Before attributing any performance differences to a specific mechanism, the first question to
answer is simpler: what does the raw horse race say? Put all 62 configurations on the same
starting line — same universe, same harness, same evaluation window — and read the leaderboard
before imposing any analytical lens. This section establishes the data, the harness rules, and
the model zoo, then reports that raw ranking and catches the one degenerate entry that must be
excluded before interpretation.

## §2.1 Universe and harness

The evaluation universe spans 29 instruments across six asset groups: eight large-cap US equity
single names (AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, WMT); six US sector ETFs (XLK, XLF, XLE,
XLV, XLP, XLU); two broad US equity ETFs (SPY, IWM); three international equity ETFs (EFA, EEM,
FXI); five fixed-income ETFs (**SHY**, IEF, TLT, AGG, HYG); and five commodity and FX instruments
(GLD, SLV, DBC, USO, EURUSD). Daily OHLCV prices are sourced from EODHD; the sample runs from 2003-01-02 to
2026-04-30 (23.3 years, approximately 5,870 NYSE trading days). BTC-USD is excluded entirely for
survivorship hygiene. Four tickers with post-2003 inception dates (GOOGL 2004-08-19, FXI
2004-10-08, GLD 2004-11-18, HYG 2007-04-11) are handled through a variable universe size
$N(t)$: pre-inception assets carry zero weight and are excluded from the covariance estimation
window, eliminating the forward-fill bias present in earlier published comparisons. The full
ticker list, class assignments, and BTC exclusion rationale are in Appendix A.

All Sharpe ratios throughout are annualized excess-return Sharpe ratios at $r_f = 0$:
$$S = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}.$$

All 62 configurations are evaluated through a single walk-forward harness
(`aiam.harness.run_horse_race`) with the following fixed parameters:

- **Rebalancing:** monthly, on NYSE business days.
- **Prices:** close-of-day; weights summing to 1.0, long-only.
- **Covariance lookback:** 252 trading days for all mean-variance-family estimations.
- **Risk-free rate:** $r_f = 0$ throughout.
- **Benchmark:** Equal-weight (EW, 1/N) portfolio \citep{demiguel2009optimal}.

The 252-day lookback is the standard calendar-year convention. It is long enough to capture a
full annual return cycle and the horizon at which shrinkage estimators exert their largest
eigenvalue-regularization benefit; it is short enough to respond to shifting correlation regimes
without locking in stale GFC-era covariance structure.

**Train/test split.** The sample is partitioned at 2022-12-31. All strategy selection and
parameterization decisions — including the regime-switching routing rule — are derived
exclusively from the 2003–2022 training window and never revisited. The held-out test period
(2023-01-01 to 2026-04-30, 3.3 years) is evaluated exactly once. Full-sample Sharpe figures in
§§3–5 use the complete 2003–2026 window; out-of-sample figures report the 2023–2026 test set.

**Backtesting hygiene.** Six practices govern look-ahead prevention throughout. (1) Weights at
date $t$ use only data observable up to close$(t)$; they apply to realized returns at $t+1$.
(2) All signals — momentum, volatility, regime classification — are constructed from lagged price
history only. (3) `held_weights = target_weights.shift(1)` is enforced at the harness level;
unlagged application manufactures spurious performance. (4) Log returns are converted to simple
returns before portfolio aggregation; mixing the two is an implementation error. (5) Pre-inception
assets carry zero weight and are excluded from covariance estimation, eliminating forward-fill
bias. (6) Every reported number is one backtest on one 29-ticker universe; Appendix E provides
block-bootstrap confidence intervals (252-day blocks, 5,000 resamples) and Memmel paired tests
around the headline findings.

## §2.2 Strategy families

The 62 configurations arise from 31 base-level configurations (24 core estimator–family
combinations + 7 expanded variants), each evaluated with and without the VMP overlay → 31×2 = 62.
Fuller derivations, the complete family inventory, and per-family specification tables are in
Appendix B. The survey below gives the operative formula and one-line intuition for each family.

Before the per-family detail, Table 1 summarises the framework families on their objective, estimator sensitivity, and required inputs.

| Framework | Objective | Estimator sensitivity | Required inputs |
|-----------|-----------|----------------------|-----------------|
| EW | Equal capital across all assets at each rebalance | None | None |
| GMV | Minimize portfolio variance subject to fully invested | High (Σ inversion) | Σ |
| MSR | Maximize portfolio Sharpe ratio | Very high (Σ inversion + μ) | Σ, μ |
| MDP | Maximize diversification ratio (weighted vol / portfolio vol) | Moderate | Σ |
| RP / ERC | Equalize asset risk contributions to portfolio variance | Moderate | Σ |
| HRP | Hierarchical clustering + recursive inverse-variance bisection | Low (no inversion) | Σ (diagonal + correlations) |
| BL | Bayesian blend of equilibrium prior and investor views | Variable (vanishes when P=0) | Σ, π, P, Ω, q |
| TSMOM | Long-only volatility-scaled trend-following on own time series | None | Past returns, realized vol |
| FF3 | Cross-sectional ranking on momentum / low-vol / quality / multi | Low | Signal scores, Σ diagonal |
| SWITCH | Regime-conditional routing across base strategies | Inherits from routed targets | Regime classification + sub-strategies |
| VMP | Time-series exposure scaling by clipped inverse realized vol | None | Realized vol only |
| MSR\_C / MVO\_C | Constrained mean-variance (box constraints on weights) | Very high | Σ, μ |
| L/S extensions | Long-short versions of TSMOM, BL-Mom, FF3-Mom | Inherits from base | Base inputs + short availability |

**Equal-weight (EW)** — the DeMiguel–Garlappi–Uppal benchmark \citep{demiguel2009optimal}:
$w_i = 1/N$ at each rebalance date, monthly. Cheapest to implement; hardest to beat persistently.

**Global Minimum Variance (GMV)** \citep{markowitz1952portfolio} — find the lowest-variance
portfolio subject to full investment and long-only constraints:
$$w = \arg\min_w\; w^\top \Sigma w, \qquad \mathbf{1}^\top w = 1,\; w \geq 0.$$
Three estimators: GMV(sample), GMV(LW) \citep{ledoit2004honey}, GMV(OAS)
\citep{chen2010shrinkage}. The sample version is susceptible to a degenerate cash corner (§2.3);
Ledoit-Wolf regularization remedies it.

**Maximum Sharpe Ratio (MSR)** \citep{sharpe1964capital} — maximize the portfolio Sharpe over
the weight vector:
$$w = \arg\max_w\; \frac{\mu^\top w}{\sqrt{w^\top \Sigma w}},$$
with $\mu$ estimated from the 252-day rolling sample mean. Two estimators: MSR(sample) and
MSR(LW). MSR(sample) is the canonical demonstration of Michaud-style error maximization
\citep{michaud1989markowitz}: the optimizer concentrates on whichever low-volatility asset had
the highest sample Sharpe in the estimation window, and the concentration unwinds out of sample.
Ledoit-Wolf regularization substantially closes this gap (§4).

**Maximum Diversification Portfolio (MDP)** \citep{choueifaty2008toward} — maximize the
diversification ratio, the weighted sum of individual volatilities divided by portfolio
volatility:
$$w = \arg\max_w\; \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}},$$
where $\sigma$ is the vector of asset-level standard deviations. Two estimators: MDP(sample) and
MDP(LW).

**Risk Parity (RP)** \citep{maillard2010properties} — equalize each asset's marginal risk
contribution to total portfolio variance:
$$w_i \cdot (\Sigma w)_i = w_j \cdot (\Sigma w)_j \qquad \forall\, i, j.$$
Two estimators: RP(sample) and RP(LW).

**Hierarchical Risk Parity (HRP)** \citep{lopezdeprado2016building} — a three-step procedure:
(i) compute the asset correlation matrix and apply Ward-linkage clustering to obtain a
dendrogram; (ii) quasi-diagonalize the covariance matrix according to the leaf ordering; (iii)
assign inverse-variance weights by recursive bisection of the dendrogram. No matrix inversion is
required, making HRP robust to near-singular covariance estimates. As shown in §4, this
structural robustness also makes HRP approximately invariant to the choice of covariance
estimator — a qualitative contrast to the MSR family. Two estimators: HRP(sample) and HRP(LW).

**Black-Litterman (BL)** \citep{black1991global,he1999litterman} — blend an equilibrium prior
$\pi$ with investor views via the Bayesian posterior:
$$E[r \mid q] = \bigl[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\bigr]^{-1}
               \bigl[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} q\bigr],$$
where $P$ is the view matrix and $\Omega$ is view uncertainty. Four variants: BL-Eq (null view,
$P = \mathbf{0}$, reduces algebraically to the equal-weight prior — the BL circularity lemma,
§4), with sample and LW estimators; BL-Mom(LW) (12-month trailing momentum views);
BL-Rev(LW) (mean-reversion views).

**Time-Series Momentum (TSMOM)** \citep{moskowitz2012time} — weight each asset in proportion
to the sign of its trailing return, scaled by target volatility:
$$w_{i,t} \propto \text{sign}\bigl(r_{i,t-k,t}\bigr) \cdot \sigma_{\text{target}} / \hat\sigma_{i,t},$$
with a long-only constraint applied. Lookback periods $k$: 12 months (TSMOM(12m)) and 6 months
(TSMOM(6m)). The long-only constraint eliminates the short leg that drives the original results
of \citet{moskowitz2012time}; the cross-asset implications are analyzed in §6.

**Fama-French factor tilts (FF3)** \citep{fama1993common} — rank assets by a cross-sectional
signal (momentum, inverse realized volatility, quality composite, or multi-factor average),
select the top tertile, and assign inverse-volatility weights within the selection. Four
variants: FF3-Mom, FF3-LowVol, FF3-Quality, FF3-Multi.

**Regime-conditional switching (SWITCH)** — route to one of three base strategies at each
rebalance date according to the prevailing macro regime, where the routing rule is derived
entirely from training data. The regime classification pipeline and the v2a routing rule
(R0→MSR(LW), R5→MSR(sample), others→MDP(LW)) are detailed in §5 and Appendix D. Two
variants are included in the main 62-configuration comparison: SWITCH(sample) and SWITCH(LW).
The v2a rule, when combined with the VMP overlay, produces the study's highest full-sample
Sharpe at 1.608 — a result whose derivation is examined in §5.

**Volatility-managed portfolio overlay (VMP)** \citep{moreira2017volatility} — scale each
strategy's daily exposure by a clipped inverse-volatility multiplier:
$$w_t^{\text{VMP}} = \text{clip}\!\left(\frac{\bar{\sigma}}{\hat{\sigma}_t},\;0.25,\;1.5\right)
                     \cdot w_t^{\text{base}},$$
where $\bar{\sigma}$ is the strategy's long-run annualized volatility, $\hat{\sigma}_t$ is the
21-day realized volatility lagged one day, and the clip bounds keep leverage in
$[0.25\times,\,1.50\times]$ of the base weight. The VMP overlay is applied to all 31 base configurations, yielding 62 total; Figure 1 (in §3)
illustrates the exposure multiplier mechanism for a representative strategy.

Seven additional configurations round out the comparison: four constrained mean-variance
variants (MSR\_C and MVO\_C, each with sample and LW estimators) and three long-short extensions
(TSMOM-LS(12m), BL-Mom-LS(LW), FF3-Mom-LS), all assuming zero borrow cost and unlimited short
availability. The long-short results are analyzed in §6.

## §2.3 The naive horse race

Across all 62 configurations, the gross-Sharpe leaderboard is led by VMP(MDP(LW)) at 1.372,
VMP(MDP(sample)) at 1.368, and — at rank 3 — VMP(GMV(sample)) at 1.345. The third entry
requires immediate attention.

![Cumulative wealth curves on a log scale, 2003–2026. Shaded regions mark the dot-com recovery (2003), GFC (2008–09), COVID (2020–02 to 2020–04), and the 2022 rate shock. VMP(BL-Mom(LW)) leads on total return; VMP(GMV(sample)) is labelled as a degenerate artifact. The regime-conditional SWITCH(v2a) strategy is also shown; its derivation is detailed in §5.](figures/cumulative_wealth.png)

![Underwater drawdown paths for the top five non-degenerate strategies plus the EW benchmark, 2003–2026. SWITCH(v2a) recovers fastest from the GFC; VMP(MDP(LW)) and VMP(MDP(sample)) exhibit the shallowest sustained drawdowns through the 2022 rate shock.](figures/underwater_drawdown.png)

**Top 10 by gross Sharpe — raw (all 62 configurations, artifact included):**

| Rank | Configuration | Sharpe | Note |
|-----:|:--------------|-------:|:-----|
|    1 | **VMP(MDP(LW))**        | 1.372 | |
|    2 | VMP(MDP(sample))    | 1.368 | |
|    3 | VMP(GMV(sample))    | 1.345 | (†) degenerate artifact |
|    4 | VMP(MSR(sample))    | 1.295 | |
|    5 | VMP(SWITCH(sample)) | 1.293 | |
|    6 | VMP(SWITCH(LW))     | 1.265 | |
|    7 | VMP(MSR(LW))        | 1.239 | |
|    8 | VMP(HRP(LW))        | 1.232 | |
|    9 | VMP(BL-Mom(LW))     | 1.217 | |
|   10 | VMP(GMV(LW))        | 1.215 | |

**(†) The SHY-concentration artifact.** **VMP(GMV(sample))** at rank 3 is not a genuine portfolio
result. The unconstrained sample covariance matrix has extreme eigenvalues: **SHY** (iShares 1–3
Year Treasury Bond, a near-cash instrument) emerges as the near-zero-variance anchor, and the
GMV(sample) optimizer corners the portfolio in it — base vol 3.16%, Sharpe 0.958, effectively
short-duration Treasury carry. VMP then scales this near-cash position up to the $1.5\times$
cap to match the target volatility; the VMP'd variant reaches ann vol 2.31% and gross
Sharpe 1.345, claiming the Sharpe credit. It earns short-duration Treasury carry, not a
diversified risk premium. Ledoit-Wolf regularization breaks the corner: GMV(LW) is a genuinely
multi-asset solution (base vol 4.01%, Sharpe 0.954; VMP(GMV(LW)) Sharpe 1.215). All subsequent
analysis excludes VMP(GMV(sample)) from comparative claims; it is retained in Appendix C for
completeness.

**Top 10 by gross Sharpe — artifact excluded:**

| Rank | Configuration | Sharpe |
|-----:|:--------------|-------:|
|    1 | VMP(MDP(LW))        | 1.372 |
|    2 | VMP(MDP(sample))    | 1.368 |
|    3 | VMP(MSR(sample))    | 1.295 |
|    4 | VMP(SWITCH(sample)) | 1.293 |
|    5 | VMP(SWITCH(LW))     | 1.265 |
|    6 | VMP(MSR(LW))        | 1.239 |
|    7 | VMP(HRP(LW))        | 1.232 |
|    8 | VMP(BL-Mom(LW))     | 1.217 |
|    9 | VMP(GMV(LW))        | 1.215 |
|   10 | VMP(GMV(OAS))       | 1.207 |

All ten entries are VMP variants. The highest-ranking base configuration — without the overlay —
is MDP(LW) at Sharpe 1.167. The full 62-strategy table with annualized return, volatility, Sharpe,
hit ratio, maximum drawdown, Calmar ratio, turnover, and net-of-cost Sharpe is in Appendix C.

A few entries at the tails of the base-strategy distribution merit brief notice here. TSMOM(12m)
(Sharpe 0.801) is the weakest long-only base configuration: the long-only constraint eliminates
the short leg that produces the original momentum anomaly in futures universes, and the 12-month
lookback in a mixed-asset universe is too noisy to compensate. The VMP overlay lifts it by
+0.258 Sharpe points, rescuing it to 1.059, near the median of all 62 configurations. BL-Mom(LW)
is the highest-returning base strategy at 12.57% annualized — its momentum-tilted
views rotate into high-return assets during trending markets — but carries a −21.34% maximum
drawdown that the VMP overlay cannot fully compress (the worst momentum-reversal periods do not
coincide with high realized volatility). The low-volatility factor portfolio FF3-LowVol achieves
a competitive risk-adjusted Sharpe (1.021) at gross return of only 4.34% per year, a level too
low for most institutional mandates without leverage.

**Out-of-sample test period (2023–2026).** On the held-out test window the family ordering is
preserved. Excluding the SHY artifact, the test-period leaders are VMP(MDP(LW)) at Sharpe 2.422,
VMP(MDP(sample)) at 2.416, and MDP(LW) at 2.304. The VMP overlay continues to improve every
base configuration on the test period, replicating the full-sample sign-test result
($24/24$ directional improvements, probability $2^{-24} \approx 6 \times 10^{-8}$ under $H_0$)
on the OOS window.

---

The headline of the naive ranking is stark: **every entry in the genuine top 10 is a VMP
variant**. This pattern holds on the held-out test period and survives transaction-cost
adjustment for all but the highest-turnover configurations (§6). Section 3 asks what the VMP
overlay is actually doing — documenting the universal 24/24 lift, its interaction with
estimator choice, and the partial redundancy with regime conditioning — before Sections 4 and
5 isolate the estimator and regime dimensions separately.
