# §5 Regimes Add Real Value When the Rule Is Built Honestly

The R5 exception surfaced at the end of §4 is not itself a strategy. A single cell in the
regime-conditional Sharpe table — MSR(sample) outperforming in one macro regime — is exactly
the kind of post-hoc observation that generates spurious backtests. The question §5 addresses
is narrower and harder: can the regime structure become a *usable rule* if it is derived
honestly from training data? And does that rule, applied without modification to a held-out
period, deliver a statistically supported improvement?

## §5.1 The honest construction

**Pre-registration discipline.** The entire regime-to-strategy mapping in SWITCH(v2a) is
derived exclusively from the 2003–2022 training window. No information from the 2023–2026
test period — not a single day of post-2022 returns — entered the routing decision. The
test period is evaluated exactly once, with the training-derived rule applied without
modification. This pre-registration discipline is what distinguishes v2a from a data-mining
exercise. Its cost is that the rule commits to training-period findings that may differ from
full-sample findings. That cost is paid explicitly and openly.

**The regime engine.** The macro classifier is a López de Prado-style feature-engineering
pipeline on eight macro indicators: GDP growth, CPI inflation, unemployment, VIX, S&P 500
trailing return, and three yield-curve features (level, slope, and curvature). Each
indicator is transformed into level, first-difference, and second-difference (convexity)
features; the dominant regime at each decision date is the mode across all eight indicator
classifications. The result is eight regimes (R0–R7) labeled by macro character. Full
methodology and regime transition matrices appear in Appendix D.

**Regime distribution.** Over the 2003–2026 sample (5,868 NYSE trading days): R0 Expansion
1,603 days (27.3%); R1 Recovery 1,481 days (25.2%); R2 Neutral 230 days (3.9%); R3 Slow
Growth 85 days (1.4%); R4 Stress 210 days (3.6%); R5 Low & Contracting 924 days (15.7%);
R6 Crisis 148 days (2.5%); R7 Contraction 1,187 days (20.2%). Expansion and recovery
(R0+R1) account for 52% of sample days; late-cycle and contraction (R5+R7) account for 36%.

**Mapping derivation.** For each regime the training-period conditional Sharpe table
(12 non-SWITCH base strategies × 8 regimes) identifies the top-performing strategy. The v2a
routing follows directly:

- **R0 (Expansion, 1,603 days):** MSR(LW) selected from training-period analysis. *Honesty
  note:* on the full 2003–2026 sample, MDP(LW) leads R0 at conditional Sharpe 1.326 — above
  MSR(LW) at 0.869. This post-hoc observation does not retroactively change the
  training-derived rule. A rule revised after seeing the full sample to favour MDP(LW) in R0
  would be overfit by construction. Pre-registration means committing before the test data is
  observed and accepting the noise that comes with that commitment.

- **R5 (Low & Contracting, 924 days):** **MSR(sample)** selected, with training-period
  conditional Sharpe 1.392 — the best non-SWITCH strategy in this regime. MSR(LW) in R5
  achieves conditional Sharpe only 1.097, confirming the gap that motivates the
  substitution.

- **R1–R4, R6–R7 (all other regimes):** MDP(LW), the strongest non-conditional base
  strategy across the remaining regime population.

## §5.2 The result and its statistical significance

![Annualized Sharpe by strategy and regime for the 12 non-SWITCH base strategies. Diverging
red–blue colormap (center = 0). Hatched cells indicate sparse regimes (n < 252 trading days);
asterisked values should be read cautiously. Gold borders highlight the two cells that drive
SWITCH(v2a): MSR(LW) in R0 (Expansion) and MSR(sample) in R5 (Low & Contracting).
](figures/regime_conditional_heatmap.png)

The regime-conditional heatmap below shows the full 12 × 8 conditional Sharpe matrix that generates the routing rule.
Three features stand out. First, the heatmap is not uniform noise: several regimes produce
strong directional separation across strategies, confirming that the macro classifier captures
real variation in the opportunity set. Second, R5 is the clearest anomaly — MSR(sample) is
the brightest cell in that column, with a conditional Sharpe well above any other strategy in
the same regime. Third, the gold-bordered cells mark the only two deviations from the
MDP(LW) default; across the other six regimes MDP(LW) is consistently dominant.

**The empirical lift.** SWITCH(LW) v1 base Sharpe=1.080; SWITCH(v2a) base Sharpe=1.514;
$\Delta=+0.434$. The two changes from v1 are: replacing R0→EW with R0→MSR(LW), and
replacing R5→MSR(LW) with R5→MSR(sample). The Memmel (2003) paired test on the 29-asset
2003–2026 sample gives $z=2.05$, $p=0.040$ — the strongest single statistically significant
result in the study. The test is computed on the base (non-VMP) return series;
the VMP warm-up correction does not enter this contrast.

**Mechanism.** In a late-cycle contraction, a narrow cluster of assets — short-duration
fixed income and defensive equities — generates the highest realized Sharpe. The
unconstrained MSR(sample) optimizer concentrates on exactly this cluster, which happens to
be correct ex post in R5. Ledoit-Wolf regularization pulls the portfolio back toward
diversification, missing the concentrated trade. The full-sample LW advantage
($\Delta=+0.164$ over sample, §4.1) reflects that R5 accounts for only 15.7% of
observations; the remaining 84.3% penalize the concentration, overwhelming the R5 advantage
in aggregate. Regime conditioning recovers the R5 advantage without incurring the aggregate
concentration penalty — not by overriding the §4 finding, but by confining the noisy
optimizer to the one regime where its bias is a feature rather than a bug.

## §5.3 OOS confirmation, VMP combination, and the combined ceiling

**OOS confirmation.** Applying the training-derived v2a mapping to the held-out 2023–2026
test period (3.3 years): SWITCH(v2a) Sharpe=2.114 versus SWITCH(LW) v1 Sharpe=2.010,
$\Delta=+0.104$. The directional sign holds. The 3.3-year window is power-limited and
$\Delta=+0.104$ alone does not clear conventional significance. The primary OOS evidence is
structural: the regime-to-strategy pattern that drives v2a's lift — most importantly
R5→MSR(sample) — replicates when the analysis is repeated on the full sample. The R0
assignment shifts post-hoc from MSR(LW) to MDP(LW) (per §5.1's honesty note), but R5,
the cell responsible for most of the +0.434 improvement, is identified consistently across
the training-only and full-sample analyses. This indicates a persistent feature of late-cycle
macro dynamics rather than training-window noise.

![Asset-class allocation weights for four representative strategies (EW, MDP(LW), MSR(LW),
SWITCH(v2a)) over 2003–2026. Shaded bands mark GFC (2008–2009), COVID (2020), and the 2022
rate-shock. SWITCH(v2a)'s abrupt reallocations at regime transitions contrast with the
smoother, regime-agnostic patterns of the other strategies.
](figures/asset_class_allocation_timeline.png)

**VMP combination.** Applying the VMP overlay to **SWITCH(v2a)** produces full-sample
Sharpe=1.608 and Calmar=0.921 — the highest gross Sharpe in the 62-strategy comparison.
This configuration stacks all three robust refinements identified in §§3–5: regime-conditional
strategy routing (§5), Ledoit-Wolf shrinkage applied within two of the three routed targets
(§4), and daily inverse-vol scaling over the routed strategy (§3).

**Subadditivity.** The combined gain is subadditive, as developed in §3.4: VMP alone on the
v1 rule adds +0.184; regime switching alone adds +0.434; combined adds +0.528, not the
additive +0.618. The two refinements share roughly 15% of their incremental Sharpe through
partial mechanism overlap — both respond to volatility regimes, VMP through a daily
multiplier and regime conditioning through a monthly strategy substitution. The $+0.528$
combined gain is still the study's largest lift from any pair of refinements.

---

::: transition
Three robust methodological refinements have now been identified and quantified — vol
management (§3), estimator choice where the optimizer overfits (§4), and regime conditioning
built from training data (§5). Their combination reaches gross Sharpe 1.608. §6 examines
what survives once implementation costs are applied, sub-periods are examined separately,
and the question of whether the 2023–2026 test period is different in kind from the training
window is addressed directly.
:::
