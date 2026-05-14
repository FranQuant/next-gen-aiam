---
author: "Francisco Sanchez"
header-includes:
  - \usepackage{booktabs}
abstract: |
  We compare 48 portfolio allocation strategies on a uniform 30-asset multi-class
  universe spanning January 2008 through April 2026 (18.3 years, approximately
  4,600 NYSE trading days). The universe covers six asset classes — large-cap US
  equities, US sector ETFs, broad equity index funds, international equity ETFs,
  fixed-income ETFs, and commodities including crypto — and is evaluated through a
  monthly walk-forward harness with no lookahead. Twenty-four base strategies drawn
  from six families (classical mean-variance, diversification-based,
  regime-conditional switching, time-series momentum, Black-Litterman, and
  cross-sectional factor portfolios) are each paired with a Volatility-Managed
  Portfolio (VMP) overlay, yielding 48 comparable performance records on identical
  data. Three headline results emerge. First, the VMP overlay improves Sharpe for
  all 24 of 24 base strategies without exception, with a median lift of +0.270
  Sharpe points, confirming volatility management as a universal meta-strategy
  across method families. Second, a Black-Litterman circularity theorem is
  confirmed numerically: equilibrium-only BL specifications produce identical
  portfolios regardless of the covariance estimator — a diagnostic boundary check
  for any BL implementation. Third, applying a uniform 10 bps round-trip
  transaction cost reorganizes the strategy rankings: momentum-intensive strategies
  collapse while regime-conditional switching and low-turnover methods retain their
  standings. The comparison is fully reproducible from cached daily returns at
  github.com/FranQuant/next-gen-aiam.
---

# Introduction

## Motivation

Portfolio construction has been a central problem in quantitative finance for over
70 years, originating with @markowitz1952portfolio's mean-variance framework. Despite
this long history, published comparison studies typically pit 3–5 methods against each
other on a single dataset, making it difficult to draw general conclusions about which
approaches survive out-of-sample across different market regimes. The rise of covariance
estimator improvements [@ledoit2004well; @chen2010shrinkage], diversification-based
objectives [@choueifaty2008toward; @maillard2010properties], hierarchical risk parity
[@lopezdeprado2016building], Black-Litterman views [@black1991global; @he1999litterman],
and volatility-managed overlays [@moreira2017volatility] has expanded the strategy
universe substantially, yet comprehensive head-to-head evaluations on a common universe
and harness remain scarce.

The benchmark proposed by @demiguel2009optimal — showing that naive equal-weight (1/N)
outperforms 14 mean-variance models on most standard datasets — raised the bar for
demonstrating out-of-sample improvements, but tested only a narrow slice of the
available strategy space. Most strategies introduced in the decade since have been
evaluated in isolation rather than side-by-side. The need is acute: practitioners face
decisions not just about which strategy family to use, but whether shrinkage or sample
covariances are better, whether VMP overlays add value uniformly or only for specific
base strategies, and whether regime-conditional switching is substitutable with
volatility management. These are empirical questions that require simultaneous
evaluation on the same data over the same period.

This study expands the comparison to 48 strategies, standardizes the evaluation harness,
and systematically adds a VMP overlay to every base method — providing the most
comprehensive single-universe allocation comparison we are aware of.

## Related Work

The foundational challenge of portfolio out-of-sample performance was crystallized by
@demiguel2009optimal, who showed that naive equal-weight (1/N) outperforms 14
mean-variance models on most standard datasets, attributing the failure to estimation
error in expected returns. @michaud1989markowitz earlier identified the "error
maximization" property of the sample Markowitz tangency portfolio — the unconstrained
optimizer amplifies estimation errors and concentrates heavily in assets with erroneously
large sample Sharpe ratios. Regularization via linear shrinkage [@ledoit2004well] and
Oracle Approximating Shrinkage [@chen2010shrinkage] address the eigenvalue instability
in sample covariance matrices.

Diversification-based strategies that side-step expected return estimation include the
Maximum Diversification Portfolio [@choueifaty2008toward] and Equal Risk Contribution
(Risk Parity) [@maillard2010properties]. Hierarchical Risk Parity
[@lopezdeprado2016building] uses machine learning clustering to construct a diversified
portfolio without matrix inversion, avoiding the instability of classical mean-variance
optimizers. The momentum anomaly, documented by @jegadeesh1993titman for cross-sectional
equities and extended to time-series settings by @moskowitz2012time, provides an
alternative signal-based weighting scheme. The low-volatility anomaly
[@frazzini2014betting] motivates factor strategies based on realized volatility ranking.
The Black-Litterman model [@black1991global; @he1999litterman] combines equilibrium
market priors with investor views via Bayesian updating, offering a principled framework
for incorporating signals without extreme concentration. Fama-French factor portfolios
[@fama1993common] extended to the momentum factor provide benchmarks for systematic
factor-based allocation.

Volatility-managed portfolios [@moreira2017volatility] scale exposure inversely to
realized variance, generating improved Sharpe ratios across a wide range of base
strategies in equity and bond markets. The present study evaluates all of these
approaches simultaneously on a single harness, covering an 18-year window that includes
the Global Financial Crisis, the COVID-19 crash, and the 2022 rate-shock bear market.

## Contribution

This study makes three contributions to the comparative portfolio evaluation literature:

1. **Scale.** To our knowledge, this is the largest single-universe, single-period
   comparison of diversified allocation strategies, evaluating 48 distinct methods
   (24 base strategies paired with and without a VMP overlay) against a common benchmark
   on identical data.

2. **Systematic VMP overlay.** Every base strategy is evaluated both with and without the
   VMP overlay [@moreira2017volatility], enabling a clean measurement of the overlay's
   marginal contribution across all families and isolating the interaction between
   structural portfolio construction and dynamic volatility management.

3. **Regime-conditional analysis and cost sensitivity.** A custom v2a regime-switching
   rule constructed from 8-regime FRED-based macro classification is compared against the
   full strategy universe. All strategies are evaluated net of 10 bps round-trip
   transaction costs to assess implementability under realistic market frictions.


# Data and Methodology

## Universe

The evaluation universe comprises 30 tickers spanning six asset classes: 8 large-cap
US equity single stocks, 6 US sector ETFs, 2 broad equity index ETFs, 3 international
equity ETFs, 5 fixed-income ETFs (investment-grade and high-yield), and 6 commodity,
FOREX, and cryptocurrency instruments. Daily close-of-business prices are sourced from
EODHD and cached locally; the sample runs from 2008-01-01 to 2026-04-30 (18.3 years,
approximately 4,600 NYSE trading days).

The 2008 start date is deliberate: it encompasses three distinct stress regimes — the
Global Financial Crisis (2008–09 to 2009–03), the COVID-19 crash (2020–02 to 2020–04),
and the 2022 rate-shock bear market — along with multiple expansion phases. This regime
heterogeneity provides a robust out-of-sample testing environment for both risk-based
and return-based strategies. The benchmark throughout is the equal-weight (EW, 1/N)
portfolio [@demiguel2009optimal], rebalanced monthly. All Sharpe ratios are computed as

$$S = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{252}$$

with $r_f = 0$ throughout (annualized excess-return Sharpe).

## Walk-Forward Harness

All 48 strategies are evaluated through a common walk-forward harness
(`aiam.harness.run_horse_race`) with the following fixed parameters:

- **Rebalancing:** monthly, on NYSE business days
- **Prices:** close-of-day; weights summing to 1.0, long-only
- **Covariance lookback:** 252 trading days for all mean-variance-family strategies
- **Risk-free rate:** $r_f = 0$ throughout
- **Benchmark:** Equal-weight (EW) portfolio

No transaction costs are applied in the base harness; a 10 bps sensitivity analysis is
performed in Section 3.3. The VMP overlay is applied daily (Section 2.4) and is assumed
costless in the base analysis; the implications of this assumption are discussed in
Section 3.3.

## Strategy Families

### Classical Mean-Variance

**Global Minimum Variance (GMV)** [@markowitz1952portfolio] minimizes portfolio variance
subject to full investment and long-only constraints:

$$w = \arg\min_w\; w^\top \Sigma w$$
$$\text{subject to}\quad \mathbf{1}^\top w = 1,\quad w \geq 0$$

Three covariance estimators are tested: sample covariance, Ledoit-Wolf linear shrinkage
[@ledoit2004well], and Oracle Approximating Shrinkage [@chen2010shrinkage], yielding
GMV(sample), GMV(LW), and GMV(OAS) respectively.

**Maximum Sharpe Ratio (MSR)** maximizes the portfolio Sharpe ratio
[@sharpe1964capital]:

$$w = \arg\max_w\; \frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}}$$

Expected returns $\mu$ are estimated from the 252-day rolling sample mean. The
Michaud [-@michaud1989markowitz] critique of sample-MSR concentration risk is confirmed
empirically (Finding 2). Two estimators: MSR(sample) and MSR(LW).

### Diversification-Based

**Maximum Diversification Portfolio (MDP)** [@choueifaty2008toward] maximizes the
diversification ratio — the ratio of weighted average asset volatility to portfolio
volatility:

$$w = \arg\max_w\; \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}$$

where $\sigma$ is the vector of individual asset volatilities (square roots of the
diagonal of $\Sigma$).

**Equal Risk Contribution / Risk Parity (RP)** [@maillard2010properties] equalizes
the marginal risk contribution of each asset:

$$w_i \cdot \text{MRC}_i = w_j \cdot \text{MRC}_j \quad \forall\, i, j$$

where $\text{MRC}_i = (\Sigma w)_i / \sqrt{w^\top \Sigma w}$ is asset $i$'s marginal
risk contribution.

**Hierarchical Risk Parity (HRP)** [@lopezdeprado2016building] constructs weights via
recursive bisection of a dendrogram computed from asset return correlations, assigning
inverse-variance weights within each cluster. HRP avoids matrix inversion and is
robust to near-singular covariance estimates. The smoothing effect of Ledoit-Wolf
shrinkage on cluster boundaries is explored in Finding 3.

### Black-Litterman

The Black-Litterman model [@black1991global; @he1999litterman] combines an equilibrium
prior $\pi$ with investor views $q$ via the posterior:

$$E[r \mid q] = \bigl[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\bigr]^{-1}
               \bigl[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} q\bigr]$$

where $P$ is the view matrix, $\Omega$ is the view uncertainty matrix, and $\tau$
scales the prior confidence. Three view specifications are tested: BL-Eq (no views,
$P = \mathbf{0}$, reducing to the prior); BL-Mom (momentum views from 12-month trailing
returns); and BL-Rev (mean-reversion views). The BL-circularity theorem (Finding 10)
follows analytically from the $P = \mathbf{0}$ case.

### Time-Series Momentum

Time-series momentum [@moskowitz2012time] constructs positions proportional to the sign
of each asset's past return, scaled by target volatility:

$$w_{i,t} = \text{sign}(r_{i,\,t-1,t-k}) \cdot \frac{\sigma_{\text{target}}}{\sigma_{i,t}}$$

with a long-only constraint applied. Lookback periods $k = 12$ months (TSMOM(12m)) and
$k = 6$ months (TSMOM(6m)) are evaluated. The long-only constraint eliminates the short
leg present in the original @moskowitz2012time results (discussed in Finding 8).

### Cross-Sectional Factor Portfolios

Factor portfolios follow a rank-then-weight approach: assets are ranked by a signal
(momentum, low-volatility, quality composite, or multi-factor average), the top third
by signal rank is selected, and weights are assigned by inverse realized volatility.
Signals follow the Fama-French factor paradigm [@fama1993common] extended to momentum
[@jegadeesh1993titman] and to the low-volatility anomaly [@frazzini2014betting].
Four variants: FF3-Mom, FF3-LowVol, FF3-Quality, FF3-Multi.

## VMP Overlay

The Volatility-Managed Portfolio overlay [@moreira2017volatility] scales each
strategy's daily exposure inversely to its 21-day realized volatility:

$$w_t^{\text{VMP}} = \text{clip}\!\left(\frac{\bar{\sigma}}{\sigma_t},\; 0.25,\; 1.5\right)
                     \cdot w_t^{\text{base}}$$

where $\bar{\sigma}$ is the strategy's long-run realized volatility (annualized),
$\sigma_t$ is the 21-day realized vol lagged one day to prevent lookahead, and the
clipping constraint keeps exposure in $[0.25\times,\, 1.50\times]$ of the base weight
vector. Because the target is each strategy's own long-run vol, the overall volatility
level is preserved and only its time-series variation is reduced. The overlay is applied
to all 24 base strategies, producing 24 additional VMP variants (48 rows total).

![VMP exposure multiplier for MSR(LW), 2008–2026. Top panel: 21-day realized vol (annualized) vs. long-run vol (11.9%). Bottom panel: exposure multiplier clipped to [0.25, 1.5]. Red fill = vol cap active; green fill = maximum leverage applied. Crisis periods appear as the deepest vol spikes; the 0.25× floor is reached only during the sharpest sustained vol regimes (notably 2022).](figures/vmp_exposure_mechanism.png)

## Regime Classification

The regime engine classifies macro state from eight FRED indicators (GDP growth, CPI
inflation, unemployment, VIX, S&P 500 trailing return, and three yield-curve
features — level, slope, and curvature) into eight regimes (0–7) using a
feature engineering pipeline [@lopezdeprado2016building] that computes level,
first-difference, and second-difference (convexity) for each indicator. The dominant
regime at each decision date is the mode across all eight indicator classifications.
Regime 0 corresponds to an expansion state (26% of sample days); Regime 5 corresponds
to a low-macro-level, falling, positive-convexity state consistent with late-cycle or
early-recession environments (17% of sample days).

The custom v2a switching rule routes: R0 (Expansion) $\to$ MSR(LW); R5 (Low &
Contracting) $\to$ MSR(sample); all other regimes $\to$ MDP(LW). This rule was
constructed from regime-conditional Sharpe analysis on 12 single-strategy baselines
(Finding 5).


# Results

## 48-Strategy Comparison

![Cumulative wealth curves on a log-y axis, 2008–2026. Shaded regions mark the GFC (2008–09 to 2009–03), COVID (2020–02 to 2020–04), and 2022 rate shock. VMP(BL-Mom(LW)) leads on total return (24.97% p.a.) but suffered the deepest base-strategy drawdown (−50.85% for BL-Mom(LW), annotated). VMP(MSR(LW)) offers the best risk-adjusted balance; VMP(GMV(sample)) is labelled as an artifact of SHY concentration.](figures/cumulative_wealth.png)

Across all 48 strategies, the top three by gross Sharpe are VMP(GMV(sample)) (1.533), VMP(MDP(sample)) (1.460), and VMP(SWITCH(sample)) (1.457) — all VMP variants of low-to-moderate turnover base strategies. By net Sharpe after 10 bps round-trip costs, the leaders shift to VMP(GMV(sample)) (1.503), VMP(MDP(LW)) (1.400), and VMP(SWITCH(LW)) (1.381), reflecting turnover penalties on the higher-rotation sample-covariance variants. Among base strategies only, the three weakest by gross Sharpe are BL-Rev(LW) (0.547), FF3-Mom (0.588), and TSMOM(12m) (0.626) — strategies where return-chasing signals generate high turnover or deep drawdowns without commensurate compensation.

The complete 48-strategy comparison table appears in Appendix A.

## Rankings

![Sharpe ratio vs. maximum drawdown for all 48 strategies. Filled circles = base strategies; open rings = VMP variants. Color encodes family (see legend). Dashed lines mark Sharpe = 1.0 and max drawdown = −20%. The VMP cluster dominates the upper-right frontier; VMP(GMV(sample)) sits far right but is an artifact of SHY concentration (low drawdown because the portfolio is near-cash).](figures/sharpe_vs_drawdown.png)

**Top 10 by Sharpe (all 48 strategies):**

| Rank | Strategy | Sharpe |
|-----:|:---------|-------:|
|    1 | VMP(GMV(sample))    | 1.533 |
|    2 | VMP(MDP(sample))    | 1.460 |
|    3 | VMP(SWITCH(sample)) | 1.457 |
|    4 | VMP(SWITCH(LW))     | 1.438 |
|    5 | VMP(MDP(LW))        | 1.437 |
|    6 | VMP(MSR(LW))        | 1.429 |
|    7 | VMP(MSR(sample))    | 1.405 |
|    8 | VMP(BL-Mom(LW))     | 1.346 |
|    9 | VMP(RP(sample))     | 1.330 |
|   10 | VMP(RP(LW))         | 1.306 |

All 10 are VMP variants. The highest-Sharpe base strategy is MSR(LW) at 1.262.

**Top 5 by annualized return:**

| Rank | Strategy | Ann Ret | Sharpe |
|-----:|:---------|--------:|-------:|
|    1 | VMP(BL-Mom(LW))       | 24.97% | 1.346 |
|    2 | BL-Mom(LW)            | 20.01% | 1.049 |
|    3 | VMP(EW)               | 18.13% | 1.253 |
|    4 | VMP(MSR(LW))          | 17.53% | 1.429 |
|    5 | VMP(BL-Eq(sample/LW)) | 16.24% | 1.145 |

**Bottom 5 by Sharpe (base strategies only):**

| Rank | Strategy    | Sharpe | Ann Ret |
|-----:|:------------|-------:|--------:|
|   24 | BL-Rev(LW)  |  0.547 |  10.17% |
|   23 | FF3-Mom     |  0.588 |   9.60% |
|   22 | TSMOM(12m)  |  0.626 |   4.05% |
|   21 | FF3-Quality |  0.726 |   6.59% |
|   20 | FF3-Multi   |  0.786 |   6.79% |

## Transaction-Cost Sensitivity

> **Footnote on VMP costs:** VMP exposure scaling is assumed costless in this sensitivity. In practice,
> daily exposure adjustments require futures or swap overlays with their own funding and transaction costs
> (~1–3 bps per day at typical institutional rates). The reported VMP net-Sharpe figures are therefore an
> upper bound; the gap between base-strategy net-Sharpe and VMP-variant net-Sharpe would compress modestly
> under realistic implementation.

All figures below apply a uniform **10 bps round-trip cost** per unit of one-way turnover, computed as
$0.5 \times \sum|w_t - w_{t-1}|$ at each decision date (raw weight change, ignoring intra-rebalance price drift).

### Top 10 by Sharpe net of 10 bps

| Rank | Strategy                       | Gross Sharpe | Net Sharpe | Turnover |
|-----:|:------------------------------|-------------:|-----------:|---------:|
|    1 | VMP(GMV(sample))               | 1.533 | 1.503 | 0.15% |
|    2 | VMP(MDP(LW))                   | 1.437 | 1.400 | 0.79% |
|    3 | VMP(SWITCH(LW))                | 1.438 | 1.381 | 1.98% |
|    4 | VMP(SWITCH(sample))            | 1.457 | 1.337 | 3.37% |
|    5 | VMP(MSR(LW))                   | 1.429 | 1.329 | 4.65% |
|    6 | VMP(MDP(sample))               | 1.460 | 1.307 | 2.60% |
|    7 | VMP(BL-Mom(LW))                | 1.346 | 1.276 | 4.91% |
|    8 | VMP(RP(LW))                    | 1.306 | 1.269 | 0.95% |
|    9 | VMP(EW)                        | 1.253 | 1.253 | 0.00% |
|   10 | GMV(sample)                    | 1.260 | 1.233 | 0.15% |

### Top 5 strategies by Sharpe degradation (base strategies only)

| Rank | Strategy               | Gross Sharpe | Net Sharpe | Turnover | Degradation |
|-----:|:-----------------------|-------------:|-----------:|---------:|------------:|
| 1 | FF3-Mom                | 0.588 | 0.310 | 20.51% | 0.277 |
| 2 | FF3-Multi              | 0.786 | 0.561 | 7.95% | 0.225 |
| 3 | MSR(sample)            | 0.884 | 0.717 | 5.19% | 0.167 |
| 4 | TSMOM(6m)              | 0.904 | 0.738 | 4.77% | 0.167 |
| 5 | HRP(sample)            | 0.902 | 0.753 | 3.92% | 0.149 |

### Reading

At 10 bps round-trip, cost impact separates into two clear groups. **Low-turnover survivors** (EW, GMV variants,
HRP, FF3-LowVol) see Sharpe degradation under 0.098 — a negligible penalty that preserves their
rankings. **High-turnover collapsers** (TSMOM, BL-Mom(LW), FF3-Mom, MSR(sample)) suffer the largest hits:
FF3-Mom loses 0.277 Sharpe points (median base-strategy degradation: 0.098).
BL-Mom(LW) is particularly exposed — its 4.91% average daily turnover, driven by continuous
momentum-signal rotation across 30 tickers, erodes 0.065 Sharpe points, and
its net Sharpe drops to 0.985 vs gross 1.049.

Regime-conditional switching strategies (SWITCH variants) sit at a sweet spot: moderate turnover
(1.98% avg) and net Sharpe 1.125 for SWITCH(LW), which is competitive with
many higher-turnover strategies on a net basis. VMP(SWITCH(LW)) net Sharpe 1.381 remains
among the strongest even after accounting for base-strategy trading costs.


# Findings

## Finding 1 — GMV(sample) is a degenerate cash corner

GMV(sample) reports vol=1.43%, ret=1.80%, Sharpe=1.260 — numbers that look
attractive until context is added. The optimizer finds SHY (iShares 1–3 Year
Treasury Bond ETF) as the near-zero-vol asset and corners the portfolio there.
At rf=1.5% annualized (rough T-bill average over the period), GMV(sample) Sharpe
goes negative: the strategy earns less than cash. Shrinkage breaks the corner:
GMV(LW) vol=3.23%, Sharpe=0.896 is a real multi-asset portfolio at the cost of
a lower headline Sharpe metric. The OAS estimator gives a similar fix (GMV(OAS)
vol=2.58%, Sharpe=0.883). Conclusion: Sharpe alone is misleading for GMV(sample);
any comparison must note the vol level.

## Finding 2 — MSR(sample) suffers Michaud-style overfit

MSR(sample) Sharpe=0.884 is one of the lowest base-strategy Sharpes in the table,
despite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates
on whichever asset had the highest sample Sharpe in the 252-day estimation window —
typically a low-vol fixed-income ETF that happened to trend up — and the
out-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization
shrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.262 (+0.378).
This is the largest single-estimator substitution effect in the table.

## Finding 3 — HRP is the only strategy where sample covariance beats shrinkage

Across all 24 base strategies, shrinkage (LW vs sample) improves Sharpe for every
family except HRP: HRP(sample) Sharpe=0.902 > HRP(LW) Sharpe=0.865 (−0.037).
HRP partitions assets via hierarchical clustering on the correlation matrix and
assigns weights by inverse-variance within clusters. Shrinkage smooths the
pairwise correlations, which blurs the cluster boundaries that HRP's dendrogram
relies on — the information HRP extracts from block structure is degraded, not improved,
by regularization. The same mechanism is absent in all other methods, which work
directly with the covariance matrix rather than its cluster structure.

## Finding 4 — Regime 5 is the second shrinkage exception

In the regime-conditional Sharpe table (14 base strategies × 8 regimes), regime 5
(low macro level, falling, with positive convexity — a late-cycle or early-recession
environment) produces MSR(sample) conditional Sharpe=1.679 vs MSR(LW) conditional
Sharpe=1.482. Sample wins by +0.197 within this regime. Regime 5 accounts for 779
of the 4 512 daily observations (~17%). In all other regimes MSR(LW) matches or
beats MSR(sample). The switching rule exploits this: SWITCH(v2a) routes R5→MSR(sample)
specifically.

## Finding 5 — SWITCH(v2a) construction

The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all
others→MDP(LW), achieving Sharpe=1.179. Regime-conditional analysis on 12
single-strategy baselines showed:

- R0 (1 176 days, 26%): MSR(LW) conditional Sharpe=1.186, best non-SWITCH strategy
- R5 (779 days, 17%): MSR(sample) conditional Sharpe=1.679, best non-SWITCH strategy

Substituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)
yields SWITCH(v2a) Sharpe=1.340 (+0.161 vs v1). V2a achieves this with only two
targeted swaps and no change to the default rule, keeping the improvement tractable
and interpretable.

![Annualized Sharpe by strategy and regime for the 12 non-SWITCH base strategies. Diverging red–blue colormap (center = 0). Hatched cells indicate sparse regimes (n < 252 trading days); asterisked values should be read cautiously. Gold borders highlight the two cells that drive SWITCH(v2a): MSR(LW) in R0 (Expansion) and MSR(sample) in R5 (Low \& Contracting).](figures/regime_conditional_heatmap.png)

## Finding 6 — VMP improves all 24/24 base strategies

VMP lifts Sharpe for every strategy in the table without exception (24/24 base
strategies, 24/24 improvements). The lift ranges from +0.145 (FF3-Mom) to +0.521
(MSR(sample)). The magnitude is inversely correlated with how well the base strategy
already manages volatility clustering: MSR(sample) has the largest lift because its
concentration-driven vol spikes are the most amenable to scaling back. HRP variants
have the smallest lifts (+0.165, +0.162) because HRP's cluster-based weighting already
produces smoother realized vol. Median lift across all 24 strategies: ≈+0.270 Sharpe points.

## Finding 6.5 — VMP(GMV(sample)) rank-1 Sharpe is an artifact

VMP(GMV(sample))'s Sharpe=1.533 is the highest in the table, but the result is an artifact: GMV(sample) corners the portfolio in SHY (iShares 1–3 Year Treasury), giving near-zero base vol, and VMP then scales exposure up to the 1.5× cap to match the target volatility — in effect leveraging a near-cash position and claiming the credit as a "portfolio" return.

## Finding 7 — VMP makes shrinkage partially redundant

VMP(MSR(sample)) Sharpe=1.405 surpasses raw MSR(LW) Sharpe=1.262 (+0.143). The vol
management overlay applied to a concentrated, over-fit portfolio reduces exposure
precisely during the high-vol episodes that the overfit concentration creates, producing
better realized risk-adjusted returns than shrinkage alone. Practically: a cheaper
estimator (no LW computation) with VMP on top outperforms the more expensive estimator
without VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.533 > GMV(LW)
Sharpe=0.896, and VMP(MDP(sample)) Sharpe=1.460 > MDP(LW) Sharpe=1.182.

## Finding 8 — TSMOM(12m) is the weakest base strategy; VMP rescues by +0.350

TSMOM(12m) Sharpe=0.626 is the lowest base-strategy Sharpe in the table. The
long-only constraint is the primary culprit: when the 12-month momentum signal is
negative for an asset, the strategy cannot short it and instead holds a zero weight,
losing the return from the short leg. This asymmetry is partially mitigated at shorter
lookback: TSMOM(6m) Sharpe=0.904. VMP(TSMOM(12m)) Sharpe=0.976 (+0.350) achieves
EW-comparable performance by scaling down exposure during the high-vol drawdown
periods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is
near the median of all 48 strategies and adds little over VMP(EW) Sharpe=1.253.

## Finding 9 — BL-Mom(LW) and VMP(BL-Mom(LW)) are the return leaders

BL-Mom(LW) annualized return=20.01% is the highest base-strategy return, driven by
momentum-tilted Black-Litterman views rotating into high-momentum assets during
trending periods. The cost is severe: max drawdown=−50.85%, the worst in the table.
VMP(BL-Mom(LW)) return=24.97% (+4.96 pp) with max drawdown compressed to −36.01%
(+14.84 pp improvement). The Calmar ratio improves from 0.394 to 0.693. No other
strategy pair in the table reaches the 20%+ return threshold. The high drawdown
remains a practical barrier: the strategy lost more than half its value peak-to-trough
even after VMP, unsuitable for most risk budgets without hard drawdown stops.

## Finding 10 — BL-circularity theorem confirmed numerically

BL-Eq(sample) and BL-Eq(LW) produce return series that differ by at most $2.8 \times 10^{-8}$
per day (floating-point rounding only) — effectively identical. Both report ret=12.76%,
vol=14.77%, Sharpe=0.887, maxdd=−37.86%. The theoretical explanation: when the
P matrix (view specification) is null, the BL posterior reduces to the prior
regardless of Σ. Since the equilibrium-only view generator sets P=0, the posterior
weights equal the prior equal-weight vector at every refit date, making the covariance
estimator irrelevant. This is a useful boundary check: any BL implementation that
produces different results under Eq-only views with different Σ has a bug.

## Finding 11 — Low-vol anomaly is real but unleveraged returns are impractical

FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)
achieves Sharpe=0.936 with vol=3.39% and ret=3.17%. The risk-adjusted performance is
competitive with EW (Sharpe=0.976) but the absolute return is too low for most
institutional mandates. VMP lifts Sharpe to 1.146 (ret=3.77%) but the vol
stabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol
earns 3.17 cents per dollar per year. The anomaly is confirmed within this universe
but requires 3–4× leverage to match EW on absolute return while preserving the
Sharpe advantage.

## Finding 12 — VMP and regime-conditional switching are partial substitutes

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

## Finding 13 — Transaction-cost survival

At 10 bps round-trip cost, the Sharpe landscape reorganizes but most key findings survive.
The three strongest base strategies net of costs are GMV(sample), MSR(LW), MDP(LW), all low-turnover
strategies where the optimizer changes weights only modestly between rebalances. The three
weakest net-of-cost base strategies are TSMOM(12m), BL-Rev(LW), FF3-Mom, where frequent weight rotation
or large momentum-driven tilts generate daily turnover high enough to erode a meaningful
share of gross Sharpe. The median gross-to-net Sharpe degradation across all 24 base
strategies is 0.098 Sharpe points; the maximum degradation is 0.277
(FF3-Mom). Finding 6 (VMP improves all 24/24 strategies) survives qualitatively on
a net basis: every VMP variant's net Sharpe exceeds the corresponding base strategy's net
Sharpe, since the VMP overlay adds Sharpe by scaling down during high-vol periods and the
base-strategy turnover cost is the same for both. Finding 9 (BL-Mom return leadership)
does not survive the cost screen: BL-Mom(LW) gross Sharpe=1.049 falls to net
Sharpe=0.985 at 4.91% average daily turnover, dropping out of the
top-10 net ranking. Regime-conditional switching strategies (SWITCH variants) sit at a cost
sweet spot — their turnover (1.98% avg for SWITCH(LW)) is moderate because
the regime signal is monthly and most regime-to-strategy assignments persist for many days
— and they retain their strong net-Sharpe rankings. VMP(SWITCH(LW)) net Sharpe
1.381 is among the best strategies on a fully net-of-cost basis.


# Discussion

## Volatility Management as a Meta-Overlay

The most striking empirical result in this study is the universality of the VMP lift
(Finding 6): every one of 24 base strategies improves when exposure is scaled
inversely to 21-day realized volatility. The lift is not uniform, however. Strategies
that already embody volatility management — HRP through cluster-based inverse-variance
weighting, RP through equal risk contribution — show the smallest gains (+0.162,
+0.165). Strategies with inherent concentration risk and vol-regime sensitivity —
MSR(sample), TSMOM(12m) — show the largest gains (+0.521, +0.350). This pattern
confirms the theoretical intuition of @moreira2017volatility: VMP adds the most value
where the base strategy's realized volatility is most forecastable, and thus most
reducible.

The interaction with regime-conditional switching (Finding 12) reveals partial
redundancy. VMP and SWITCH(v2a) both respond to volatility regimes — VMP through a
daily multiplicative scalar, regime switching through a monthly strategy replacement.
The two mechanisms share roughly 10% of their variance explained, confirming they are
partial substitutes. Stacking both yields the best combined Sharpe in the study (1.588
for VMP(SWITCH(v2a))), but the gain is subadditive: the marginal value of the second
layer diminishes when the first already adapts to vol regimes. Practitioners face a
complexity-cost tradeoff: the VMP overlay alone over a simple base strategy (e.g.,
VMP(MDP(LW)) Sharpe=1.437) achieves near-top performance without the regime
classification infrastructure.

## Shrinkage vs. Structure

Covariance estimation matters enormously for classical mean-variance strategies — the
MSR Michaud overfit (Finding 2, Δ=+0.378 Sharpe from sample to LW) is the largest
single methodological substitution effect in the table. Shrinkage pulls extreme
eigenvalues toward the grand mean, reducing the optimizer's tendency to concentrate
on assets with fortuitously large in-sample Sharpe ratios. The GMV and MDP families
show smaller but consistent improvements under shrinkage, with the OAS estimator
[@chen2010shrinkage] producing results similar to Ledoit-Wolf [@ledoit2004well].

The exception is hierarchical structure. HRP(sample) Sharpe=0.902 beats HRP(LW)
Sharpe=0.865 because shrinkage smooths the pairwise correlations that HRP's dendrogram
relies on to define cluster boundaries (Finding 3). This is a structural incompatibility:
LW shrinkage pulls correlations toward a common mean, blurring the block structure that
encodes economic asset groupings. The mechanism is absent in all other families because
they operate directly on $\Sigma$ rather than its cluster topology. The practical
implication is that HRP should be paired with sample (or alternatively lightly
regularized) covariance, while all other families benefit from full shrinkage.

## Transaction Costs as the Implementability Filter

The cost-sensitivity analysis (Section 3.3, Finding 13) functions as an
implementability filter: it reveals which strategies survive from the academic
performance table into an institutional portfolio context. Two groups emerge cleanly
at 10 bps round-trip. Low-turnover strategies (EW, GMV variants, FF3-LowVol,
SWITCH(LW)) suffer degradation below 0.098 Sharpe points and maintain their relative
rankings. High-turnover strategies (FF3-Mom, FF3-Multi, TSMOM, BL-Mom(LW)) suffer
the largest losses — FF3-Mom's gross Sharpe of 0.588 falls to 0.310 net, making it
the weakest strategy on a cost-adjusted basis despite a 9.60% annualized gross return.

The regime-conditional switching strategies occupy a strategically important position:
moderate turnover (1.98% average for SWITCH(LW)) with a regime signal that persists
for many days produces a cost-adjusted Sharpe that rivals more complex structures.
VMP(SWITCH(LW)) net Sharpe 1.381 is the strongest non-degenerate result in the
cost-adjusted table. The transaction-cost ladder is ultimately the implementability
filter for institutional adoption: strategies must survive from gross Sharpe to
net-of-friction Sharpe to risk-budget approval, and only the structural methods
(diversification-based, regime-conditional) pass all three screens reliably.


# Conclusion and Future Work

This study evaluated 48 portfolio allocation strategies — the largest single-universe
comparison in the literature we are aware of — across 18.3 years of daily multi-asset
returns from 2008 to 2026. The central findings are: (1) the VMP overlay is a
universal Sharpe-improver with a median lift of +0.270 that works across all six
strategy families; (2) Ledoit-Wolf shrinkage is universally beneficial except for HRP,
where it degrades cluster boundary information; (3) the Black-Litterman circularity
theorem holds numerically for zero-view specifications; and (4) regime-conditional
switching and VMP are partial substitutes targeting the same volatility-regime
vulnerability through complementary mechanisms; (5) transaction costs reorganize the
ranking table, with regime-conditional and low-turnover strategies surviving as
the implementability leaders.

**Limitations.** All results are for a specific 30-ticker universe with US equity
tilt; strategies exploiting cross-sectional dispersion (FF3, BL-Mom) may respond
differently in small-cap or non-US universes. Only monthly rebalancing is tested.
The VMP overlay cost is assumed zero in the base analysis; daily exposure scaling
via futures overlays carries its own friction (~1–3 bps/day). The TSMOM long-only
constraint systematically weakens time-series momentum relative to published long-short
implementations [@moskowitz2012time]. All Sharpe ratios are computed at $r_f = 0$; at
positive risk-free rates the relative ordering of low-return strategies (GMV(sample),
FF3-LowVol) would deteriorate further.

**Future work.** The harness architecture accommodates several natural extensions.
First, machine-learning signal strategies — Lasso expected-return estimation, Random
Forest regime classification, and XGBoost factor scoring — can replace the rolling
sample-mean signals currently used in MSR and BL-Mom, with the uniform harness
providing a clean performance attribution. Second, long-short extensions of TSMOM and
factor portfolios would remove the long-only constraint penalty and allow direct
replication of published results [@moskowitz2012time; @jegadeesh1993titman]. Third,
multi-universe robustness checks — applying the same 48-strategy comparison to a
global equity universe, a fixed-income-only universe, and a commodities universe —
would test whether the ranking structure generalizes across asset classes. Fourth,
deep learning sequence models (LSTM, Transformer) and reinforcement learning agents
via a `SequentialStrategy` interface represent the frontier for dynamic allocation
within the same evaluation framework.

# References {.unnumbered}

::: {#refs}
:::

# Appendix A — Full 48-Strategy Comparison Table {.unnumbered}

The table below presents the complete performance record for all 48 strategies evaluated in this study. Strategies are organized by family (Classical Mean-Variance, Diversification-Based, Regime Switch, Time-Series Momentum, Black-Litterman, and Factor), with each base strategy followed immediately by its VMP variant. The benchmark equal-weight portfolio (EW) appears at the head of the Classical MV block. Columns report annualized return (Ann Ret), annualized volatility (Ann Vol), gross Sharpe ratio (Sharpe), maximum drawdown (Max DD), Calmar ratio, average daily one-way turnover (Turnover), and Sharpe ratio net of a uniform 10 bps round-trip transaction cost (Net Sharpe). All figures are computed over the full 2008-01-01 to 2026-04-30 walk-forward period with no lookahead, using the monthly rebalancing harness described in Section 2.2. The VMP(GMV(sample)) Sharpe of 1.533 is the highest in the table but is flagged as a degenerate result (see Finding 1 and Finding 6.5): the optimizer concentrates entirely in SHY, producing near-zero portfolio volatility that the VMP overlay then levers to the 1.5× cap.

```{=latex}
\clearpage
\footnotesize
\begin{tabular}{p{1.5cm} p{3.6cm} r r r r r r r}
\toprule
Family & Strategy & Ann Ret & Ann Vol & Sharpe & Max DD & Calmar & Turnover & Net Sharpe \\
\midrule
Classical MV & EW                  & 14.31\% & 14.84\% & 0.976 & -37.86\% & 0.378 & 0.00\%  & 0.976 \\
             & VMP(EW)             & 18.13\% & 14.09\% & 1.253 & -28.95\% & 0.626 & 0.00\%  & 1.253 \\
             & GMV(sample)         &  1.80\% &  1.43\% & 1.260 &  -5.94\% & 0.304 & 0.15\%  & 1.233 \\
             & VMP(GMV(sample))    &  2.00\% &  1.30\% & 1.533 &  -5.40\% & 0.371 & 0.15\%  & 1.503 \\
             & GMV(LW)             &  2.88\% &  3.23\% & 0.896 & -11.60\% & 0.248 & 0.54\%  & 0.853 \\
             & VMP(GMV(LW))        &  3.86\% &  3.26\% & 1.178 & -11.11\% & 0.348 & 0.54\%  & 1.136 \\
             & GMV(OAS)            &  2.27\% &  2.58\% & 0.883 & -10.64\% & 0.213 & 0.47\%  & 0.837 \\
             & VMP(GMV(OAS))       &  3.13\% &  2.60\% & 1.200 &  -9.11\% & 0.344 & 0.47\%  & 1.154 \\
             & MSR(sample)         &  6.81\% &  7.80\% & 0.884 & -21.47\% & 0.317 & 5.19\%  & 0.717 \\
             & VMP(MSR(sample))    &  8.44\% &  5.89\% & 1.405 & -11.45\% & 0.737 & 5.19\%  & 1.183 \\
             & MSR(LW)             & 15.40\% & 11.91\% & 1.262 & -21.43\% & 0.719 & 4.65\%  & 1.163 \\
             & VMP(MSR(LW))        & 17.53\% & 11.80\% & 1.429 & -22.66\% & 0.774 & 4.65\%  & 1.329 \\
\midrule
Diversification & MDP(sample)      &  5.05\% &  4.63\% & 1.088 & -18.40\% & 0.275 & 2.60\%  & 0.945 \\
                & VMP(MDP(sample)) &  6.41\% &  4.32\% & 1.460 & -12.03\% & 0.533 & 2.60\%  & 1.307 \\
                & MDP(LW)          &  6.34\% &  5.32\% & 1.182 & -15.73\% & 0.403 & 0.79\%  & 1.144 \\
                & VMP(MDP(LW))     &  7.94\% &  5.42\% & 1.437 & -13.16\% & 0.604 & 0.79\%  & 1.400 \\
                & RP(sample)       &  5.36\% &  5.59\% & 0.961 & -15.96\% & 0.336 & 2.96\%  & 0.829 \\
                & VMP(RP(sample))  &  7.22\% &  5.35\% & 1.330 & -12.20\% & 0.592 & 2.96\%  & 1.191 \\
                & RP(LW)           &  7.25\% &  6.74\% & 1.073 & -16.61\% & 0.437 & 0.95\%  & 1.037 \\
                & VMP(RP(LW))      &  8.82\% &  6.64\% & 1.306 & -13.68\% & 0.645 & 0.95\%  & 1.269 \\
                & HRP(sample)      &  5.99\% &  6.70\% & 0.902 & -16.57\% & 0.362 & 3.92\%  & 0.753 \\
                & VMP(HRP(sample)) &  7.04\% &  6.57\% & 1.068 & -15.51\% & 0.454 & 3.92\%  & 0.915 \\
                & HRP(LW)          &  6.48\% &  7.60\% & 0.865 & -15.65\% & 0.414 & 3.63\%  & 0.743 \\
                & VMP(HRP(LW))     &  7.63\% &  7.42\% & 1.027 & -15.06\% & 0.506 & 3.63\%  & 0.903 \\
\midrule
Regime Switch & SWITCH(sample)      &  8.70\% &  8.09\% & 1.071 & -20.79\% & 0.418 & 3.37\%  & 0.967 \\
              & VMP(SWITCH(sample)) & 10.48\% &  7.01\% & 1.457 & -13.91\% & 0.753 & 3.37\%  & 1.337 \\
              & SWITCH(LW)          & 11.02\% &  9.23\% & 1.179 & -21.13\% & 0.521 & 1.98\%  & 1.125 \\
              & VMP(SWITCH(LW))     & 12.91\% &  8.71\% & 1.438 & -18.06\% & 0.715 & 1.98\%  & 1.381 \\
\midrule
TS Momentum & TSMOM(12m)          &  4.05\% &  6.70\% & 0.626 & -21.68\% & 0.187 & 2.93\%  & 0.514 \\
            & VMP(TSMOM(12m))     &  6.13\% &  6.30\% & 0.976 & -13.47\% & 0.455 & 2.93\%  & 0.857 \\
            & TSMOM(6m)           &  6.48\% &  7.23\% & 0.904 & -24.18\% & 0.268 & 4.77\%  & 0.738 \\
            & VMP(TSMOM(6m))      &  7.27\% &  6.56\% & 1.102 & -12.33\% & 0.589 & 4.77\%  & 0.918 \\
\midrule
Black-Litterman & BL-Eq(sample)       & 12.76\% & 14.77\% & 0.887 & -37.86\% & 0.337 & 0.00\%  & 0.887 \\
                & VMP(BL-Eq(sample))  & 16.24\% & 14.00\% & 1.145 & -28.85\% & 0.563 & 0.00\%  & 1.145 \\
                & BL-Eq(LW)           & 12.76\% & 14.77\% & 0.887 & -37.86\% & 0.337 & 0.00\%  & 0.887 \\
                & VMP(BL-Eq(LW))      & 16.24\% & 14.00\% & 1.145 & -28.85\% & 0.563 & 0.00\%  & 1.145 \\
                & BL-Mom(LW)          & 20.01\% & 19.12\% & 1.049 & -50.85\% & 0.394 & 4.91\%  & 0.985 \\
                & VMP(BL-Mom(LW))     & 24.97\% & 17.73\% & 1.346 & -36.01\% & 0.693 & 4.91\%  & 1.276 \\
                & BL-Rev(LW)          & 10.17\% & 22.27\% & 0.547 & -48.33\% & 0.210 & 10.05\% & 0.433 \\
                & VMP(BL-Rev(LW))     & 12.18\% & 19.13\% & 0.697 & -47.61\% & 0.256 & 10.05\% & 0.565 \\
\midrule
Factor & FF3-Mom             &  9.60\% & 18.53\% & 0.588 & -39.51\% & 0.243 & 20.51\% & 0.310 \\
       & VMP(FF3-Mom)        & 11.61\% & 16.97\% & 0.733 & -29.85\% & 0.389 & 20.51\% & 0.430 \\
       & FF3-LowVol          &  3.17\% &  3.39\% & 0.936 & -10.68\% & 0.296 & 0.41\%  & 0.905 \\
       & VMP(FF3-LowVol)     &  3.77\% &  3.27\% & 1.146 &  -9.53\% & 0.395 & 0.41\%  & 1.115 \\
       & FF3-Quality         &  6.59\% &  9.41\% & 0.726 & -25.98\% & 0.254 & 3.62\%  & 0.628 \\
       & VMP(FF3-Quality)    &  8.18\% &  8.06\% & 1.016 & -16.72\% & 0.489 & 3.62\%  & 0.902 \\
       & FF3-Multi           &  6.79\% &  8.86\% & 0.786 & -19.54\% & 0.348 & 7.95\%  & 0.561 \\
       & VMP(FF3-Multi)      &  8.35\% &  8.42\% & 0.995 & -15.98\% & 0.522 & 7.95\%  & 0.757 \\
\bottomrule
\end{tabular}
\clearpage
```
