# §3 Volatility Management: The Universal Lever

Section §2 closed with a pattern that demands explanation: every entry in the genuine
top 10 is a VMP variant. Before asking why MSR and HRP sit so far apart on the base-strategy
leaderboard — the estimator question §4 addresses — there is a prior question to answer.
What is the overlay actually doing, and why does it help every strategy rather than just the
ones that need it most?

## §3.1 The 24/24 result

The most direct answer: the VMP overlay improves the gross Sharpe ratio on every one of the
24 original base configurations without exception (Finding 6). Lifts range from +0.119
(FF3-Mom) to +0.400 (MSR(sample)); the median improvement is +0.194 Sharpe points. This
24/24 record is the statistical anchor of the section. Under $H_0$ that the overlay is
equally likely to help or hurt any given strategy, the probability of observing 24 positive
signs out of 24 independent trials is $2^{-24} \approx 6 \times 10^{-8}$ (Finding R2). No
pairwise significance threshold is needed to defend the universality claim; the sign test
overwhelms it. The headline pairwise contrast — VMP(MSR(LW)) vs. MSR(LW), $\Delta = +0.180$
Sharpe — is individually marginal ($z = 1.90$, $p = 0.058$), but the directional consistency
across all 24 families is the evidence that matters.

The result extends to the 7 expanded configurations (4 constrained MV variants, 3 long-short
extensions): 6 of 7 improve under VMP. The sole exception is FF3-Mom-LS, where
VMP(FF3-Mom-LS) worsens an already near-zero Sharpe (0.103 → −0.045). With a base strategy
whose rolling returns are frequently negative, inverse-vol scaling amplifies the downside
during drawdown periods rather than dampening vol spikes. This exception is specific to
near-zero-expected-return strategies and does not qualify the universal finding for the
original 24 families.

The OOS test period (2023–2026) replicates the sign-test result exactly: all 24 original
base strategies improve under VMP on the held-out window, providing out-of-sample confirmation
that the universality is structural rather than sample-specific.

## §3.2 The mechanism: de-risking into vol spikes

The formula defined in §2.2 makes the mechanism precise, but the intuition is simpler. VMP
scales daily exposure inversely to the strategy's 21-day realized volatility, clipped to
$[0.25\times, 1.5\times]$ of the base weight vector. Because the target $\bar{\sigma}$ is
each strategy's own long-run realized vol, the overlay does not change the average volatility
level — it compresses its time-series variation around that level.

![VMP exposure multiplier for MSR(LW), 2003–2026. Top panel: 21-day realized vol (annualized) vs. long-run vol (10.6%). Bottom panel: exposure multiplier clipped to [0.25, 1.5]. Red fill = vol cap active; green fill = maximum leverage applied. Crisis periods appear as the deepest vol spikes; the 0.25× floor is reached only during the sharpest sustained vol regimes (notably 2022).](figures/vmp_exposure_mechanism.png)

Two features of Figure 1 define the mechanism in practice. First, the multiplier spends the
majority of its time between 0.8× and 1.5×, with the upper cap binding frequently during
expansionary low-vol periods (green fill). For MSR(LW), whose long-run vol is 10.6%
annualized, the 2003–2007 and 2012–2017 expansions are long enough to sustain near-maximum
leverage for extended stretches, compounding return. Second, the $0.25\times$ floor is
reached only during the sharpest, most sustained vol regimes. The 2022 rate shock is the
clearest example: a prolonged high-vol drawdown that pushes 21-day realized vol well above
the long-run target for multiple consecutive months, triggering maximum de-risking. GFC
spikes are sharper but shorter, producing briefer floor contacts.

The net effect on realized returns is straightforward: the numerator of the Sharpe ratio
(expected daily return) scales approximately linearly with exposure, while the denominator
(daily vol) is highest precisely when exposure is lowest. Vol is most forecastable at short
horizons — high-vol begets high-vol — so the 21-day lookback captures the persistence that
drives the mechanism. The overlay de-risks into vol spikes and re-risks into calm, trading
the tail of the return distribution for a smoother path. Risk-adjusted performance improves
not because VMP generates signal, but because it reduces the realized vol drag during the
episodes that dominate the denominator.

## §3.3 Asymmetric magnitude: the overlay lifts most where it is needed most

The +0.119 to +0.400 range is not noise. It reflects a structural gradient that is
inversely related to how well the base strategy already manages vol clustering.

The strategies with the smallest lifts among traditional families are HRP(sample) (+0.130)
and HRP(LW) (+0.139). HRP's cluster-based inverse-variance weighting already allocates away
from assets with extreme realized vol at each rebalance; the VMP overlay adds a second
layer over a first layer that is partially doing the same job. Risk Parity sits just above:
RP(sample) and RP(LW) lift by +0.181 and +0.183 respectively. Equal-risk-contribution
objectives equalize individual volatility contributions across assets, producing realized
portfolio vol that is less subject to single-asset spikes than equal-weight or MV
constructions. The vol-management already embedded in the base strategy leaves less room
for the overlay.

MSR(sample) sits at the opposite extreme (+0.400, 0.895 → 1.295). The unconstrained
optimizer concentrates the portfolio in whichever asset had the highest in-sample Sharpe,
and that concentration is itself a source of vol instability — when the concentrated asset's
vol spikes, the whole portfolio spikes. VMP's inverse scaling responds to exactly these
concentration-driven episodes, reducing exposure and then restoring it as vol normalizes.
The overlay is most valuable where the base strategy is least self-protective.

TSMOM(12m) offers the most instructive tail case. It is the weakest base-strategy Sharpe
in the table (0.801), and VMP rescues it by +0.258 Sharpe points to 1.059 — near the median
of all 62 configurations. The lift comes from scaling down exposure during the high-vol
drawdown periods that dominate TSMOM(12m)'s poor record, not from manufacturing signal. Even
after VMP rescue, TSMOM(12m) adds little over VMP(EW) (1.133): the lever lifts from below,
it does not distinguish between signal quality. VMP makes a weak strategy mediocre, not a
winner.

## §3.4 Partial redundancy with shrinkage and regime conditioning

The most counterintuitive VMP result is that the overlay applied to a noisy estimator can
exceed shrinkage applied alone — or even shrinkage combined with VMP. VMP(MSR(sample))
achieves Sharpe 1.295, exceeding both raw MSR(LW) at 1.059 (+0.236, Finding 7) and
VMP(MSR(LW)) at 1.239. The same pattern holds for VMP(GMV(sample)) Sharpe 1.345 exceeding
GMV(LW) at 0.954, and VMP(MDP(sample)) Sharpe 1.368 exceeding MDP(LW) at 1.167.

The mechanism: Ledoit-Wolf shrinkage regularizes the covariance matrix at the estimation
stage, reducing concentration before the optimizer runs. VMP reduces concentration at the
execution stage, scaling back exposure when the over-concentrated portfolio is most volatile.
Both refinements attack the same problem — noisy covariance leads to concentration leads to
vol spikes — through different paths. For MSR, the two paths are partially substitutable,
and the execution-stage intervention (VMP) turns out to be sufficient to not only recover
the shrinkage advantage but to exceed it: the concentrated MSR(sample) portfolio generates
larger vol spikes than MSR(LW), which gives VMP more to compress, overshooting the shrinkage
effect in realized Sharpe terms. Adding shrinkage on top of VMP (i.e., VMP(MSR(LW)) 1.239)
actually reduces rather than raises the final Sharpe, because the LW-regularized portfolio
has smaller vol spikes and thus less for VMP to exploit.

This does not mean shrinkage is dominated by VMP across the board — for other family choices
(HRP, RP) the covariance estimator matters far less and the VMP lift is smaller. The point
is narrower: for the MSR family, VMP and LW shrinkage both target the same source of
underperformance (concentration-driven vol), and the execution-stage intervention is
sufficient to dominate the estimation-stage one.

VMP and regime-conditional switching exhibit the same partial redundancy (Finding 12). Both
respond to volatility regimes: VMP through a daily multiplicative scalar, regime switching
through a monthly strategy replacement. Stacking them yields Sharpe 1.608 for
VMP(SWITCH(v2a)), the highest in the study, but the gain is subadditive: VMP alone on the
v1 rule adds +0.184; regime switching alone adds +0.434; combined adds +0.580, not the
additive +0.618. The two refinements share roughly 6% of their incremental Sharpe. For
practitioners facing a complexity–cost tradeoff, VMP over a simple base — VMP(MDP(LW)) at
1.372 or VMP(MDP(sample)) at 1.368 — achieves near-top performance without the regime
classification infrastructure.

---

VMP lifts every base strategy, by different amounts, through the same inverse-vol mechanism.
What it cannot explain is why MSR and HRP sit so far apart on the base-strategy leaderboard,
or why Ledoit-Wolf shrinkage adds 0.164 Sharpe points to MSR but is statistically
indistinguishable from zero for HRP. For that we turn to Section §4, where estimator choice
is the independent variable.
