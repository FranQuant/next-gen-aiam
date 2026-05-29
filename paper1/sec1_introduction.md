# §1 Introduction

When practitioners argue about classical portfolio allocation — sample mean-variance versus
Ledoit-Wolf shrinkage, mean-variance versus HRP, static versus regime-conditional — the
empirical evidence is surprisingly fragmented. \citet{demiguel2009optimal} famously showed that
naïve equal-weight (1/N) outperforms 14 optimized mean-variance portfolios on standard datasets,
raising the out-of-sample bar for every methodology that followed. The post-2009 literature
responded with genuine advances — hierarchical clustering \citep{lopezdeprado2016building},
volatility-managed overlays \citep{moreira2017volatility}, robust covariance estimators
\citep{ledoit2004honey} — yet these innovations have mostly been evaluated in isolation, on
bespoke universes, over varying sample periods. The practitioner is left with a menu of
seemingly plausible improvements and no reliable way to rank them. This paper does that ranking.

We evaluate the principal families of classical portfolio allocation — classical mean-variance
\citep{markowitz1952portfolio}, diversification objectives
\citep{choueifaty2008toward,maillard2010properties}, risk parity, hierarchical risk parity
\citep{lopezdeprado2016building}, regime-conditional switching, time-series momentum
\citep{moskowitz2012time}, Black-Litterman views \citep{black1991global,he1999litterman},
and Fama-French factor tilts \citep{fama1993common} — each tested across estimator choices
(sample, Ledoit-Wolf, OAS) and with and without a volatility-managed overlay
\citep{moreira2017volatility}, yielding 62 strategy configurations on a single 29-asset
multi-asset universe (US single names, US sector ETFs, broad and international equity ETFs,
fixed income, and commodities/FX) over 2003–2026 (23.3 years, approximately 5,870 NYSE trading
days). These comprise 31 base-level configurations (24 core estimator-family combinations plus seven
expanded variants), each evaluated with and without the VMP overlay — 31 × 2 = 62. All strategies share a common walk-forward harness, a single held-out test
period (2023 onwards), identical transaction-cost accounting, and a Sharpe ratio convention
following \citet{sharpe1966mutual}. Statistical robustness is assessed via
\citet{memmel2003performance} paired tests, block-bootstrap confidence intervals, and
calendar-year sub-period analysis.

Three patterns dominate, and they compose. Volatility management is the universal lever:
applying the VMP overlay of \citet{moreira2017volatility} improves the Sharpe ratio on every
one of the 24 base strategies, a result confirmed by a sign test at $p \approx 6 \times
10^{-8}$. Robust covariance estimation helps significantly where the optimizer amplifies sample
noise — the MSR family, where Ledoit-Wolf shrinkage \citep{ledoit2004honey} closes the gap
between sample-covariance overfit and the efficient frontier — but is near-irrelevant for HRP,
whose hierarchical clustering already smooths block correlations. Regime conditioning, derived
entirely from training data through a López de Prado-style feature-engineering pipeline on eight macro indicators,
adds a statistically significant and economically meaningful lift over the strongest
non-conditional strategy. The strongest classical strategy in the study combines all three:
**VMP** applied to the regime-conditional switcher **SWITCH(v2a)**, which reaches a Sharpe ratio of
1.608 in the train-only-derived configuration.

The uncomfortable footnote deserves its own paragraph. Within-strategy variation across calendar
years exceeds the cross-strategy variation in the full-sample headline table for most pairs. A
strategy that ranked first in 2003–2009 frequently ranked near the median in 2010–2022 and
vice versa. This does not invalidate the three structural patterns — VMP universality,
shrinkage selectivity, and regime-conditioning lift are robust across sub-periods — but it does
caution against treating any single headline Sharpe as a deployment signal. The paper's most
defensible claim is that these three *mechanisms* are reliable; the *ranking* of specific
parameterizations is not.

The remainder of the paper proceeds as follows. Section 2 introduces the universe, the walk-forward
methodology, and the full strategy zoo, and reports the naïve horse-race ranking alongside the
SHY-concentration artifact that must be excluded before interpretation. Section 3 establishes
VMP as the universal lever, documenting the sign-test result and the magnitude of improvement
across base families. Section 4 contrasts the MSR family — where shrinkage is a first-order
improvement — with HRP, where it is not, isolating the conditions under which estimator choice
matters. Section 5 documents the regime-conditional SWITCH(v2a) result: the strongest single
methodological refinement in the study and the only one whose training-only derivation
replicates the full-sample derivation. Section 6 stress-tests the rankings under realistic
transaction costs, on the held-out 2023–2026 test period, and across calendar sub-periods, where
the most uncomfortable finding lives. Section 7 names the three robust patterns, states what the
62-strategy harness cannot say, and points forward: to learned return predictors and
regime detectors (Paper 2) and to a multi-tool agentic research workflow that orchestrates data engineering, strategy
construction, backtesting, and risk review under a governance substrate (Paper 3).
