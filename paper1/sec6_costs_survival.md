# §6 Costs, Survival, and What the Rankings Hide

Gross Sharpe 1.608 is the best-case figure. It is measured before trading costs, over the
full 23.3-year in-sample window, without sub-period interrogation. Sections §3–§5 identified
three robust refinements — vol management, estimator regularisation, regime conditioning —
and established their combination as the strongest classical configuration in the study. This
section subjects those findings to three stress tests of escalating severity: transaction costs
(which strategies survive fees?), out-of-sample survival (which mechanisms replicate on the
held-out test period?), and sub-period sensitivity (does the headline ranking survive
calendar-year scrutiny?). The third test yields the paper's most defensible conclusion.

---

## §6.1 The implementability filter

Institutional adoption imposes a sequential cost screen: gross Sharpe must survive to
net-of-friction Sharpe, and net Sharpe must then pass a risk-budget approval. The analysis
applies a uniform 10 bps round-trip cost per unit of one-way daily turnover computed at each
decision date; Appendix F provides the asset-class-stratified cost ladder (2 bps fixed income,
3 bps broad/international ETFs, 5 bps sector ETFs and commodities/FX, 5 bps single names)
and the full stratified table.

At 10 bps flat, the 62 strategies divide into two clean groups. **Low-turnover survivors**
hold their rankings intact: EW (0.006% average daily turnover) loses fewer than 0.001 Sharpe
points; GMV variants (turnover under 0.5%) lose under 0.02; MDP(LW) (0.78% turnover) falls
from gross 1.167 to net 1.131; FF3-LowVol (0.39% turnover) from 1.021 to 0.998. These
strategies incur a negligible implementability penalty. **High-turnover collapsers** tell a
different story. FF3-Mom's 20.25% average daily turnover converts gross Sharpe 0.685 into net
Sharpe 0.394 — a loss of 0.291 Sharpe points, the largest single degradation in the table.
FF3-Mom-LS falls further to net −0.306, the weakest strategy in the study on a cost-adjusted
basis. BL-Mom(LW), with 5.11% turnover driven by continuous momentum-signal rotation across
29 tickers, loses 0.108 Sharpe points (gross 1.042 → net 0.934) and exits the top-10 net
ranking.

SWITCH(LW) occupies the strategically important intermediate position. Its 2.04% average
daily turnover — the regime signal fires monthly and most assignments persist for many
consecutive days — produces a net Sharpe of 1.020 (gross 1.080, loss 0.060). SWITCH(LW)
survives as a cost-adjusted competitor precisely because regime-based rebalancing is sparse by
construction. VMP(MDP(LW)) leads the cost-adjusted table at net Sharpe 1.336 (gross 1.372,
loss 0.037); VMP(SWITCH(LW)) follows at net Sharpe 1.201 (gross 1.265, loss 0.064). The
ordering at the top of the table is unchanged by costs.

![Scatter of net Sharpe under flat 10 bps round-trip (x-axis) vs. net Sharpe under
asset-class-stratified costs (y-axis) for all 62 strategies. Points above the diagonal benefit
from stratified pricing; FF3-Mom and FF3-Multi are the largest gainers (+0.193 and +0.150
Sharpe points respectively) as their frequent ETF rebalancing is materially cheaper than the
flat-10-bps baseline. Points on the diagonal are turnover-insensitive strategies. Filled
circles = base strategies; open rings = VMP variants.
](figures/stratified_vs_flat_costs.png)

Figure 7 confirms that the flat-10-bps assumption is conservative for equity-heavy factor
strategies: under stratified costs the top-5 net ranking (excluding artifacts) shifts to
VMP(MDP(LW)) 1.360, VMP(MDP(sample)) 1.327, VMP(SWITCH(sample)) 1.258, VMP(SWITCH(LW))
1.242, VMP(EW) 1.133. MDP-family strategies retain the leadership; the qualitative conclusion
— regime-conditional and low-turnover strategies as implementability leaders — survives both
cost assumptions.

Two strategies warrant specific notes at this point in the ranking. **FF3-LowVol** (gross
Sharpe 1.021, vol 4.25%, annualized return 4.34%) confirms the low-volatility anomaly on a
cross-asset universe — the anomaly is real and the OOS sub-period evidence (§6.2) corroborates
it — but the absolute return is too low for most institutional mandates without 3–4× leverage
to match EW on absolute return. The Sharpe is competitive; the unleveraged product is not.
**Long-short strategies**: activating the short leg in this heterogeneous universe is a
risk-profile transformation, not a Sharpe enhancement. BL-Mom-LS(LW) is the cleanest
illustration — Sharpe barely changes from long-only (0.904 vs. BL-Mom(LW) at 1.042) while
annualized vol collapses from 12.07% to 4.65% and max drawdown improves from −21.34% to
−11.87%. The L/S form is a qualitatively different instrument for risk-budgeted mandates. The
remaining long-short variants (TSMOM-LS, FF3-Mom-LS) produce no such risk-profile benefit:
they underperform their long-only counterparts on gross Sharpe and collapse further on net
Sharpe, consistent with the mixed-universe shorting problem — the short leg captures
mean-reverting asset classes (bonds, commodities in their respective drawdown periods) rather
than genuine momentum losers [@moskowitz2012time].

---

## §6.2 Out-of-sample survival of the mechanism claims

The held-out 2023–2026 test period (approximately 3.3 years, 830 trading days) provides the
cleanest available test of the three mechanisms identified in §§3–5. No strategy
hyperparameter, no SWITCH routing rule, no VMP target vol was touched after the
2022-12-31 training cutoff.

**VMP universality.** The VMP overlay improves gross Sharpe over the corresponding base
strategy for all 24 original base configurations on the 2023–2026 test period — replicating
the full-sample 24/24 result exactly. Under $H_0$ that the overlay is equally likely to help
or hurt, the sign-test $p$-value is again $2^{-24} \approx 6 \times 10^{-8}$. This is the
most powerful OOS statement the paper can make: the directional consistency is overwhelming at
any conventional significance level.

**MSR shrinkage benefit.** MSR(LW) maintains a Sharpe advantage over MSR(sample) on the test
period, consistent with the full-sample Michaud-overfit finding (§4.1). The
optimizer-concentration mechanism — aggressive sample-weighting toward whichever asset
recently trended up — is not peculiar to the training-period regime sequence; it operates
persistently across sub-periods.

**HRP near-invariance.** HRP(sample) and HRP(LW) produce similar OOS performance, consistent
with the Memmel (2003) Sharpe test (§4.2, Finding R4). The near-invariance is stable across
sub-periods; neither estimator dominates reliably on either horizon, confirming the finding is
about HRP's structural insensitivity to eigenvalue regularization rather than a training-sample
artifact.

**Regime conditioning.** On the held-out test period, SWITCH(v2a) produces OOS Sharpe 2.114
vs. SWITCH(LW) v1 at 2.010, $\Delta = +0.104$. The directional sign holds. The 3.3-year
test window limits statistical power and the gap alone does not clear conventional significance;
the primary OOS evidence for v2a is structural — the regime-to-strategy routing derived
exclusively from 2003–2022 training data produces the same R5→MSR(sample) assignment as the
full-sample analysis (§5.3). The R5 anomaly is a persistent feature of late-cycle macro
dynamics rather than a training-window artifact.

What does not survive OOS is the full-sample ordering of specific parameterisations within
families. Test-period leadership is held by the MDP family — VMP(MDP(LW)) OOS Sharpe 2.422,
VMP(MDP(sample)) 2.416, MDP(LW) 2.304 — which is directionally consistent with the
full-sample MDP ranking but involves Sharpe magnitudes two to three times the training-period
figures. The regime shift from a high-vol, rate-shock training environment to a low-vol,
post-shock test environment changes the return distribution, amplifying differences between
base strategies in a way that inflates test-period Sharpes uniformly. §6.3 addresses this
directly.

---

## §6.3 Sub-period sensitivity: the test the paper imposes on its own ranking

![Calendar-year returns for 24 representative strategies, 2003–2026 (* 2026 through April). Red
cells = negative annual return; green cells = positive. Broad-based losses in 2008 (GFC) and
2022 (rate shock) are unmistakable; 2003 is the only year with near-uniform positive returns
across all strategy families. TSMOM(12m) and BL-Mom(LW) exhibit the largest year-to-year
swings; SWITCH(LW) and HRP variants the smallest.
](figures/calendar_returns_heatmap.png)

Figure 10 is the visual core of the section. Each row is a single strategy; the colour range
within that row — how far from green to red a single strategy travels across years — is the
within-strategy variation. By inspection it is large. The cross-strategy variation — the
difference between the greenest and reddest row in any single year — is what the headline
Sharpe table ranks. The table below quantifies the comparison.

Table 2 reports annualized Sharpe ratios for 8 representative strategies across the five
non-overlapping sub-periods that span the full sample:

```{=latex}
\footnotesize
\begin{tabular}{l r r r r r}
\toprule
Strategy & 2003--2007 & 2008--2012 & 2013--2017 & 2018--2022 & 2023--2026 \\
\midrule
EW                   &  1.72 &  0.35 &  1.17 &  0.63 &  2.03 \\
MSR(LW)              &  1.42 &  0.53 &  1.51 &  0.41 &  1.80 \\
MSR(sample)          &  1.53 &  0.89 &  1.44 &  0.30 &  1.19 \\
MDP(LW)              &  1.61 &  0.82 &  1.27 &  0.26 &  2.30 \\
SWITCH(LW)           &  1.59 &  0.58 &  1.24 &  0.53 &  1.73 \\
SWITCH(v2a)          &  1.24 &  1.23 &  1.36 &  0.48 &  2.01 \\
VMP(MSR(LW))         &  1.57 &  0.70 &  1.46 &  0.50 &  2.18 \\
VMP(MDP(LW))         &  1.87 &  1.02 &  1.26 &  0.56 &  2.43 \\
\bottomrule
\end{tabular}
\normalsize
```

The core finding is numerical: within-strategy Sharpe variation across the five sub-periods
substantially exceeds the cross-strategy variation in the full-sample headline table. MSR(LW)
— gross full-sample Sharpe 1.059, one of the better non-degenerate base strategies — swings
from 0.41 in 2018–2022 to 1.51 in 2013–2017, a within-strategy range of 1.10 Sharpe points.
The full-sample cross-strategy spread, from the weakest non-degenerate base strategy
(BL-Rev(LW) at 0.663) to the strongest (MDP(LW) at 1.167), spans 0.50 Sharpe points. The
within-strategy variation is more than twice the cross-strategy spread that generates the
headline ranking. MDP(LW) itself illustrates the problem from the other direction: full-sample
Sharpe 1.167 is the best non-degenerate base result, yet MDP(LW) produces Sharpe 0.26 in
2018–2022 — the weakest result of any strategy in that window — and 2.30 in 2023–2026.
A practitioner who selected MSR(LW) in 2019 based on its 2013–2017 performance (Sharpe 1.51)
would have experienced its weakest sub-period (Sharpe 0.41) over the following four years.

SWITCH(v2a) is the partial exception that proves the rule. Its 2008–2012 sub-period Sharpe of
1.23 is the highest in Table 2 for that window — precisely the crisis period where most other
strategies underperform — and its five sub-period values (1.24, 1.23, 1.36, 0.48, 2.01) are
more stable than any MV strategy in the table. Regime conditioning adds the most visible
defensive value when the macro signal is sharpest; the GFC is exactly that window. Yet even
SWITCH(v2a) suffers a weak 2018–2022 (Sharpe 0.48), demonstrating that regime conditioning
reduces sub-period variance without eliminating it. The VMP overlay reduces variance further:
VMP(MDP(LW))'s worst sub-period is 0.56, compared to MDP(LW)'s 0.26, and VMP(MSR(LW))'s
worst sub-period of 0.50 is better than MSR(LW)'s 0.41. The three refinements survive as
mechanisms — vol management, shrinkage, regime conditioning each provide a consistent
directional benefit visible within each sub-period row — but they do not stabilise the
sub-period ranking.

![12-month rolling Sharpe for six representative strategies (EW, MDP(LW), MSR(LW), SWITCH(LW),
VMP(MDP(LW)), HRP(LW)), 2003–2026. Dashed line at Sharpe 1.0. Grey shading marks GFC
(2008–09), COVID (2020-02 to 2020-04), and the 2022 rate shock. VMP(MDP(LW)) sustains the
most consistently elevated rolling Sharpe across the full sample; MSR(LW) shows the widest
within-strategy variation. All strategies produce rolling Sharpe well below 1.0 during
sustained crisis periods.
](figures/rolling_sharpe_small_multiples.png)

The implication should be stated without softening. The full-sample ranking table is not a
stable ranking of strategy quality. It is a ranking of average performance over one specific
23.3-year sequence of macro regimes. The cross-strategy gap between, say, VMP(MDP(LW)) at
full-sample Sharpe 1.372 and VMP(MSR(LW)) at 1.239 does not constitute reliable evidence that
the first deserves a higher institutional allocation than the second: a single sub-period
reversal (2018–2022 shows VMP(MDP(LW)) at 0.56 vs. VMP(MSR(LW)) at 0.50, essentially tied)
can close the full-sample gap. The paper's strongest and most defensible claims are therefore
claims about mechanisms, not about specific configurations: vol management universally improves
Sharpe within every sub-period; Ledoit-Wolf shrinkage consistently benefits mean-variance
families; regime conditioning provides the most visible crisis-period protection. These three
patterns persist across sub-periods. The combined VMP(SWITCH(v2a)) configuration at 1.608 is
the best classical result this study has found; it is not a forecast of what will rank first
over any future five-year horizon.

---

The three stress tests produce a coherent verdict. Transaction costs reorganise the bottom half
of the ranking table but leave the structural leaders — diversification-based and
regime-conditional strategies — in place; VMP(MDP(LW)) at net 1.336 and VMP(SWITCH(LW)) at
net 1.201 survive the implementability filter. The three mechanisms from §§3–5 replicate
directionally on the held-out test period; the precise strategy ranking does not need to — its
instability is the expected and honest result of a finite sample. Sub-period analysis
establishes that within-strategy variation dwarfs cross-strategy variation, confirming that the
paper's strongest statements are about mechanisms and not about the 1.608 configuration as a
deployable signal.

§7 draws on these three findings to articulate what the horse race has actually established
about structural portfolio construction, states the study's limitations, and closes with the
forward pointer to the comparison with learned methods in Paper 2.
