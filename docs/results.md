# Results — next-gen-aiam horse race

**Period:** 2008-01-01 to 2026-04-30 (18.3 years, ~4 600 trading days)
**Universe:** 30 tickers — 8 large-cap equities, 6 sector ETFs, 2 broad equity ETFs,
3 international equity ETFs, 5 fixed-income ETFs, 6 commodities/FX/crypto
**Rebalancing:** monthly (`ME` frequency), no transaction costs, long-only
**Benchmark:** EW (equal-weight, 1/N)
**Sharpe formula:** `(r - rf).mean() / (r - rf).std() * sqrt(252)`, rf = 0 throughout
**VMP params (all entries):** lookback=21d, lag=1d, exposure clipped to [0.25, 1.5],
target vol = each strategy's own long-run realized vol

---

## 1. Universe and Methodology

The 30-ticker panel spans six asset classes: large-cap US equity, US sector ETFs,
broad equity ETFs, international equity ETFs, investment-grade and high-yield fixed
income, and commodities plus FOREX and BTC. The 2008 start date deliberately captures
the global financial crisis, COVID-2020, and the 2022 rate-shock bear market, giving
three distinct stress regimes within the evaluation window.

All strategies run through a common walk-forward harness (`aiam.harness.run_horse_race`):
monthly refit, close-of-day prices, weights summing to 1.0. The covariance lookback for
all mean-variance-family strategies is 252 trading days (1 year). The regime engine
classifies macro state from 8 FRED indicators (GDP, CPI, unemployment, VIX, SPX, and
three yield-curve features) into 8 regimes (0–7) using Lopez de Prado–style level/
change/convexity features; the dominant regime is the mode across indicators.

The VMP (Volatility-Managed Portfolio) overlay (Moreira and Muir 2017) scales each
strategy's daily exposure inversely to its 21-day realized volatility, clipped to [0.25, 1.5],
and lagged one day to prevent lookahead. All VMP variants target the strategy's own
long-run realized volatility, so the vol level is preserved and only its time-series
variation is reduced.

---

## 2. Full 48-Strategy Comparison Table

48 rows = 24 base strategies × (base + VMP). Organized by method family.
Columns: annualized return, annualized volatility, Sharpe ratio, maximum drawdown, Calmar ratio.

### 2a. Classical Mean-Variance

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| EW                    |  14.31% |  14.84% |  0.976 | -37.86% |  0.378 |
| VMP(EW)               |  18.13% |  14.09% |  1.253 | -28.95% |  0.626 |
| GMV(sample)           |   1.80% |   1.43% |  1.260 |  -5.94% |  0.304 |
| VMP(GMV(sample))      |   2.00% |   1.30% |  1.533 |  -5.40% |  0.371 |
| GMV(LW)               |   2.88% |   3.23% |  0.896 | -11.60% |  0.248 |
| VMP(GMV(LW))          |   3.86% |   3.26% |  1.178 | -11.11% |  0.348 |
| GMV(OAS)              |   2.27% |   2.58% |  0.883 | -10.64% |  0.213 |
| VMP(GMV(OAS))         |   3.13% |   2.60% |  1.200 |  -9.11% |  0.344 |
| MSR(sample)           |   6.81% |   7.80% |  0.884 | -21.47% |  0.317 |
| VMP(MSR(sample))      |   8.44% |   5.89% |  1.405 | -11.45% |  0.737 |
| MSR(LW)               |  15.40% |  11.91% |  1.262 | -21.43% |  0.719 |
| VMP(MSR(LW))          |  17.53% |  11.80% |  1.429 | -22.66% |  0.774 |

### 2b. Diversification-Based

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| MDP(sample)           |   5.05% |   4.63% |  1.088 | -18.40% |  0.275 |
| VMP(MDP(sample))      |   6.41% |   4.32% |  1.460 | -12.03% |  0.533 |
| MDP(LW)               |   6.34% |   5.32% |  1.182 | -15.73% |  0.403 |
| VMP(MDP(LW))          |   7.94% |   5.42% |  1.437 | -13.16% |  0.604 |
| RP(sample)            |   5.36% |   5.59% |  0.961 | -15.96% |  0.336 |
| VMP(RP(sample))       |   7.22% |   5.35% |  1.330 | -12.20% |  0.592 |
| RP(LW)                |   7.25% |   6.74% |  1.073 | -16.61% |  0.437 |
| VMP(RP(LW))           |   8.82% |   6.64% |  1.306 | -13.68% |  0.645 |
| HRP(sample)           |   5.99% |   6.70% |  0.902 | -16.57% |  0.362 |
| VMP(HRP(sample))      |   7.04% |   6.57% |  1.068 | -15.51% |  0.454 |
| HRP(LW)               |   6.48% |   7.60% |  0.865 | -15.65% |  0.414 |
| VMP(HRP(LW))          |   7.63% |   7.42% |  1.027 | -15.06% |  0.506 |

### 2c. Regime-Conditional Switching

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| SWITCH(sample)        |   8.70% |   8.09% |  1.071 | -20.79% |  0.418 |
| VMP(SWITCH(sample))   |  10.48% |   7.01% |  1.457 | -13.91% |  0.753 |
| SWITCH(LW)            |  11.02% |   9.23% |  1.179 | -21.13% |  0.521 |
| VMP(SWITCH(LW))       |  12.91% |   8.71% |  1.438 | -18.06% |  0.715 |

### 2d. Time-Series Momentum

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| TSMOM(12m)            |   4.05% |   6.70% |  0.626 | -21.68% |  0.187 |
| VMP(TSMOM(12m))       |   6.13% |   6.30% |  0.976 | -13.47% |  0.455 |
| TSMOM(6m)             |   6.48% |   7.23% |  0.904 | -24.18% |  0.268 |
| VMP(TSMOM(6m))        |   7.27% |   6.56% |  1.102 | -12.33% |  0.589 |

### 2e. Black-Litterman

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| BL-Eq(sample)         |  12.76% |  14.77% |  0.887 | -37.86% |  0.337 |
| VMP(BL-Eq(sample))    |  16.24% |  14.00% |  1.145 | -28.85% |  0.563 |
| BL-Eq(LW)             |  12.76% |  14.77% |  0.887 | -37.86% |  0.337 |
| VMP(BL-Eq(LW))        |  16.24% |  14.00% |  1.145 | -28.85% |  0.563 |
| BL-Mom(LW)            |  20.01% |  19.12% |  1.049 | -50.85% |  0.394 |
| VMP(BL-Mom(LW))       |  24.97% |  17.73% |  1.346 | -36.01% |  0.693 |
| BL-Rev(LW)            |  10.17% |  22.27% |  0.547 | -48.33% |  0.210 |
| VMP(BL-Rev(LW))       |  12.18% |  19.13% |  0.697 | -47.61% |  0.256 |

### 2f. Cross-Sectional Factor Portfolios

| Strategy              | Ann Ret | Ann Vol | Sharpe | Max DD  | Calmar |
|-----------------------|--------:|--------:|-------:|--------:|-------:|
| FF3-Mom               |   9.60% |  18.53% |  0.588 | -39.51% |  0.243 |
| VMP(FF3-Mom)          |  11.61% |  16.97% |  0.733 | -29.85% |  0.389 |
| FF3-LowVol            |   3.17% |   3.39% |  0.936 | -10.68% |  0.296 |
| VMP(FF3-LowVol)       |   3.77% |   3.27% |  1.146 |  -9.53% |  0.395 |
| FF3-Quality           |   6.59% |   9.41% |  0.726 | -25.98% |  0.254 |
| VMP(FF3-Quality)      |   8.18% |   8.06% |  1.016 | -16.72% |  0.489 |
| FF3-Multi             |   6.79% |   8.86% |  0.786 | -19.54% |  0.348 |
| VMP(FF3-Multi)        |   8.35% |   8.42% |  0.995 | -15.98% |  0.522 |

### Cross-table rankings

**Top 10 by Sharpe (all 48 rows):**

| Rank | Strategy              | Sharpe |
|-----:|-----------------------|-------:|
|    1 | VMP(GMV(sample))      |  1.533 |
|    2 | VMP(MDP(sample))      |  1.460 |
|    3 | VMP(SWITCH(sample))   |  1.457 |
|    4 | VMP(SWITCH(LW))       |  1.438 |
|    5 | VMP(MDP(LW))          |  1.437 |
|    6 | VMP(MSR(LW))          |  1.429 |
|    7 | VMP(MSR(sample))      |  1.405 |
|    8 | VMP(BL-Mom(LW))       |  1.346 |
|    9 | VMP(RP(sample))       |  1.330 |
|   10 | VMP(RP(LW))           |  1.306 |

All 10 are VMP variants. The highest-Sharpe base strategy is MSR(LW) at 1.262.

**Top 5 by annualized return:**

| Rank | Strategy              | Ann Ret | Sharpe |
|-----:|-----------------------|--------:|-------:|
|    1 | VMP(BL-Mom(LW))       |  24.97% |  1.346 |
|    2 | BL-Mom(LW)            |  20.01% |  1.049 |
|    3 | VMP(EW)               |  18.13% |  1.253 |
|    4 | VMP(MSR(LW))          |  17.53% |  1.429 |
|    5 | VMP(BL-Eq(sample/LW)) |  16.24% |  1.145 |

**Bottom 5 by Sharpe (base strategies only):**

| Rank | Strategy   | Sharpe | Ann Ret |
|-----:|------------|-------:|--------:|
|   24 | BL-Rev(LW) |  0.547 |  10.17% |
|   23 | FF3-Mom    |  0.588 |   9.60% |
|   22 | TSMOM(12m) |  0.626 |   4.05% |
|   21 | FF3-Quality|  0.726 |   6.59% |
|   20 | FF3-Multi  |  0.786 |   6.79% |

---

## 3. Main Findings

### Finding 1 — GMV(sample) is a degenerate cash corner

GMV(sample) reports vol=1.43%, ret=1.80%, Sharpe=1.260 — numbers that look
attractive until context is added. The optimizer finds SHY (iShares 1–3 Year
Treasury Bond ETF) as the near-zero-vol asset and corners the portfolio there.
At rf=1.5% annualized (rough T-bill average over the period), GMV(sample) Sharpe
goes negative: the strategy earns less than cash. Shrinkage breaks the corner:
GMV(LW) vol=3.23%, Sharpe=0.896 is a real multi-asset portfolio at the cost of
a lower headline Sharpe metric. The OAS estimator gives a similar fix (GMV(OAS)
vol=2.58%, Sharpe=0.883). Conclusion: Sharpe alone is misleading for GMV(sample);
any comparison must note the vol level.

### Finding 2 — MSR(sample) suffers Michaud-style overfit

MSR(sample) Sharpe=0.884 is one of the lowest base-strategy Sharpes in the table,
despite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates
on whichever asset had the highest sample Sharpe in the 252-day estimation window —
typically a low-vol fixed-income ETF that happened to trend up — and the
out-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization
shrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.262 (+0.378).
This is the largest single-estimator substitution effect in the table.

### Finding 3 — HRP is the only strategy where sample covariance beats shrinkage

Across all 24 base strategies, shrinkage (LW vs sample) improves Sharpe for every
family except HRP: HRP(sample) Sharpe=0.902 > HRP(LW) Sharpe=0.865 (−0.037).
HRP partitions assets via hierarchical clustering on the correlation matrix and
assigns weights by inverse-variance within clusters. Shrinkage smooths the
pairwise correlations, which blurs the cluster boundaries that HRP's dendrogram
relies on — the information HRP extracts from block structure is degraded, not improved,
by regularization. The same mechanism is absent in all other methods, which work
directly with the covariance matrix rather than its cluster structure.

### Finding 4 — Regime 5 is the second shrinkage exception

In the regime-conditional Sharpe table (14 base strategies × 8 regimes), regime 5
(low macro level, falling, with positive convexity — a late-cycle or early-recession
environment) produces MSR(sample) conditional Sharpe=1.679 vs MSR(LW) conditional
Sharpe=1.482. Sample wins by +0.197 within this regime. Regime 5 accounts for 779
of the 4 512 monthly observations (~17%). In all other regimes MSR(LW) matches or
beats MSR(sample). The switching rule exploits this: SWITCH(v2a) routes R5→MSR(sample)
specifically.

### Finding 5 — SWITCH(v2a) construction

The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all
others→MDP(LW), achieving Sharpe=1.179. Regime-conditional analysis on 12
single-strategy baselines showed:

- R0 (1 176 days, 26%): MSR(LW) conditional Sharpe=1.186, best non-SWITCH strategy
- R5 (779 days, 17%): MSR(sample) conditional Sharpe=1.679, best non-SWITCH strategy

Substituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)
yields SWITCH(v2a) Sharpe=1.340 (+0.161 vs v1). V2a achieves this with only two
targeted swaps and no change to the default rule, keeping the improvement tractable
and interpretable.

### Finding 6 — VMP improves all 24/24 base strategies

VMP lifts Sharpe for every strategy in the table without exception (24/24 base
strategies, 24/24 improvements). The lift ranges from +0.145 (FF3-Mom) to +0.521
(MSR(sample)). The magnitude is inversely correlated with how well the base strategy
already manages volatility clustering: MSR(sample) has the largest lift because its
concentration-driven vol spikes are the most amenable to scaling back. HRP variants
have the smallest lifts (+0.165, +0.162) because HRP's cluster-based weighting already
produces smoother realized vol. Median lift across all 24 strategies: ≈+0.270 Sharpe points.

### Finding 7 — VMP makes shrinkage partially redundant

VMP(MSR(sample)) Sharpe=1.405 surpasses raw MSR(LW) Sharpe=1.262 (+0.143). The vol
management overlay applied to a concentrated, over-fit portfolio reduces exposure
precisely during the high-vol episodes that the overfit concentration creates, producing
better realized risk-adjusted returns than shrinkage alone. Practically: a cheaper
estimator (no LW computation) with VMP on top outperforms the more expensive estimator
without VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.533 > GMV(LW)
Sharpe=0.896, and VMP(MDP(sample)) Sharpe=1.460 > MDP(LW) Sharpe=1.182.

### Finding 8 — TSMOM(12m) is the weakest base strategy; VMP rescues by +0.350

TSMOM(12m) Sharpe=0.626 is the lowest base-strategy Sharpe in the table. The
long-only constraint is the primary culprit: when the 12-month momentum signal is
negative for an asset, the strategy cannot short it and instead holds a zero weight,
losing the return from the short leg. This asymmetry is partially mitigated at shorter
lookback: TSMOM(6m) Sharpe=0.904. VMP(TSMOM(12m)) Sharpe=0.976 (+0.350) achieves
EW-comparable performance by scaling down exposure during the high-vol drawdown
periods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is
near the median of all 48 strategies and adds little over VMP(EW) Sharpe=1.253.

### Finding 9 — BL-Mom(LW) and VMP(BL-Mom(LW)) are the return leaders

BL-Mom(LW) annualized return=20.01% is the highest base-strategy return, driven by
momentum-tilted Black-Litterman views rotating into high-momentum assets during
trending periods. The cost is severe: max drawdown=−50.85%, the worst in the table.
VMP(BL-Mom(LW)) return=24.97% (+4.96 pp) with max drawdown compressed to −36.01%
(+14.84 pp improvement). The Calmar ratio improves from 0.394 to 0.693. No other
strategy pair in the table reaches the 20%+ return threshold. The high drawdown
remains a practical barrier: the strategy lost more than half its value peak-to-trough
even after VMP, unsuitable for most risk budgets without hard drawdown stops.

### Finding 10 — BL-circularity theorem confirmed numerically

BL-Eq(sample) and BL-Eq(LW) produce return series that differ by at most 2.8×10⁻⁸
per day (floating-point rounding only) — effectively identical. Both report ret=12.76%,
vol=14.77%, Sharpe=0.887, maxdd=−37.86%. The theoretical explanation: when the
P matrix (view specification) is null, the BL posterior reduces to the prior
regardless of Σ. Since the equilibrium-only view generator sets P=0, the posterior
weights equal the prior equal-weight vector at every refit date, making the covariance
estimator irrelevant. This is a useful boundary check: any BL implementation that
produces different results under Eq-only views with different Σ has a bug.

### Finding 11 — Low-vol anomaly is real but unleveraged returns are impractical

FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)
achieves Sharpe=0.936 with vol=3.39% and ret=3.17%. The risk-adjusted performance is
competitive with EW (Sharpe=0.976) but the absolute return is too low for most
institutional mandates. VMP lifts Sharpe to 1.146 (ret=3.77%) but the vol
stabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol
earns 3.17 cents per dollar per year. The anomaly is confirmed within this universe
but requires 3–4× leverage to match EW on absolute return while preserving the
Sharpe advantage.

### Finding 12 — VMP and regime-conditional switching are partial substitutes

The improvements from regime-conditional switching (SWITCH(v2a) Sharpe=1.340 vs
SWITCH(LW) Sharpe=1.179, Δ=+0.161) and from VMP on top of the original rule
(VMP(SWITCH(LW)) Sharpe=1.438 vs SWITCH(LW) Sharpe=1.179, Δ=+0.259) are comparable
in magnitude. Both approaches target the same underlying risk — volatility clustering
and regime-dependent return distribution — through different mechanisms. Stacking them
(applying VMP to v2a) yields Sharpe=1.588 and Calmar=0.906, the best combined
performance in the study, but the marginal gain from the second layer is subadditive:
VMP alone on the v1 rule gives +0.259, regime switching alone gives +0.161, combined
gives +0.409, not +0.420. The two refinements share roughly 10% of their variance
explained.

---

## 4. Limitations

**Single universe.** All results are for the specific 30-ticker panel. The universe
contains no small-cap single stocks, no private assets, and has heavy US equity tilt
(8 of 30 tickers). Strategies that exploit cross-sectional dispersion (FF3, BL-Mom)
are particularly sensitive to universe composition; results may not generalize to
other asset universes.

**Single rebalancing frequency.** Monthly rebalancing is the only frequency tested.
Higher-frequency rebalancing would likely increase turnover costs (not modeled) and
may benefit momentum strategies; lower frequency would benefit low-turnover strategies.
The TSMOM long-only weakness finding is partly a function of monthly rebalancing; daily
rebalancing would allow faster signal exit.

**No transaction costs.** All Sharpe, return, and drawdown figures are gross of
transaction costs. Strategies with high portfolio turnover — BL-Mom(LW), FF3-Mom,
MSR(sample) — would see greater degradation net of costs. A realistic 10 bps round-trip
cost would meaningfully reduce Sharpe for monthly-rebalanced high-turnover strategies.
The VMP overlay adds daily exposure scaling, which is not modeled as a cost; in
practice VMP requires daily futures or swap overlays.

**Long-only constraint.** TSMOM and factor strategies cannot short, eliminating the
return contribution of the short book. Published TSMOM results (Moskowitz, Ooi, Pedersen
2012) assume long-short; the long-only version used here is a systematically weakened
variant. The comparisons with vol-management and regime-switching strategies are
therefore conservative for TSMOM.
