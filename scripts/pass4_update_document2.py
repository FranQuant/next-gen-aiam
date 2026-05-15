#!/usr/bin/env python3
"""Pass-4: Update docs/results.md — remaining updates (precise text matching)."""

from pathlib import Path

MD = Path("docs/results.md")
text = MD.read_text()

def rep(old, new, n=1):
    global text
    if old not in text:
        print(f"NOT FOUND (skipping): {old[:80]!r}")
        return
    text = text.replace(old, new, n)
    print(f"OK: {old[:60]!r}")

# ── §1.2 Related Work ──────────────────────────────────────────────────────────
rep("covering an 18-year window that includes",
    "covering a 23.3-year window that includes")

# ── §3.1 lead paragraph ───────────────────────────────────────────────────────
rep("Across all 62 strategies, the top three by gross Sharpe are VMP(GMV(sample)) (1.345), VMP(MDP(sample)) (1.368), and VMP(MDP(LW)) (1.372) — all VMP variants of low-to-moderate turnover base strategies. VMP(GMV(sample)) is flagged as a degenerate artifact (see Findings 1 and 6.5 and Section 3.2). By net Sharpe after 10 bps round-trip costs, the leaders shift to VMP(MDP(LW)) (1.337), VMP(SWITCH(LW)) (1.203), and VMP(SWITCH(sample)) (1.178), reflecting turnover penalties on the higher-rotation sample-covariance variants. Among base strategies only, the three weakest by gross Sharpe are BL-Rev(LW) (0.663), FF3-Mom (0.685), and TSMOM(12m) (0.801) — strategies where return-chasing signals generate high turnover or deep drawdowns without commensurate compensation.",
    "Across all 62 strategies, the top three by gross Sharpe are VMP(MDP(LW)) (1.372), VMP(MDP(sample)) (1.368), and VMP(GMV(sample)) (1.345) — all VMP variants of low-to-moderate turnover base strategies. VMP(GMV(sample)) is flagged as a degenerate artifact (see Findings 1 and 6.5 and Section 3.2). By net Sharpe after 10 bps round-trip costs, the leaders shift to VMP(MDP(LW)) (1.336), VMP(MDP(sample)) (1.227), and VMP(SWITCH(LW)) (1.201), reflecting turnover penalties on the higher-rotation sample-covariance variants. Among base strategies only, the three weakest by gross Sharpe are BL-Rev(LW) (0.663), FF3-Mom (0.685), and TSMOM(12m) (0.801) — strategies where return-chasing signals generate high turnover or deep drawdowns without commensurate compensation.")

# ── §3.2 Top 10 raw ────────────────────────────────────────────────────────────
rep("""|    1 | VMP(GMV(sample))    | 1.533 | (†) degenerate artifact — SHY concentration |
|    2 | VMP(MDP(sample))    | 1.460 | |
|    3 | VMP(SWITCH(sample)) | 1.457 | |
|    4 | VMP(SWITCH(LW))     | 1.438 | |
|    5 | VMP(MDP(LW))        | 1.437 | |
|    6 | VMP(MSR(LW))        | 1.429 | |
|    7 | VMP(MSR(sample))    | 1.405 | |
|    8 | VMP(MSR_C(LW))      | 1.390 | |
|    9 | VMP(BL-Mom(LW))     | 1.346 | |
|   10 | VMP(RP(sample))     | 1.330 | |

(†) VMP(GMV(sample)) Sharpe=1.533 is not a genuine portfolio result: GMV(sample) corners the portfolio in SHY (iShares 1–3 Year Treasury), producing near-zero base volatility, and VMP then levers up to 1.5× of that near-cash position. The "Sharpe" reflects cash concentration, not diversified portfolio construction. Rankings 2–10 are genuine.""",
    """|    1 | VMP(MDP(LW))        | 1.372 | |
|    2 | VMP(MDP(sample))    | 1.368 | |
|    3 | VMP(GMV(sample))    | 1.345 | (†) degenerate artifact — SHY concentration |
|    4 | VMP(MSR(sample))    | 1.295 | |
|    5 | VMP(SWITCH(sample)) | 1.293 | |
|    6 | VMP(SWITCH(LW))     | 1.265 | |
|    7 | VMP(MSR(LW))        | 1.239 | |
|    8 | VMP(HRP(LW))        | 1.232 | |
|    9 | VMP(BL-Mom(LW))     | 1.217 | |
|   10 | VMP(GMV(LW))        | 1.215 | |

(†) VMP(GMV(sample)) Sharpe=1.345 is not a genuine portfolio result: GMV(sample) corners the portfolio in SHY (iShares 1–3 Year Treasury), producing near-zero base volatility, and VMP then levers up to 1.5× of that near-cash position. The "Sharpe" reflects cash concentration, not diversified portfolio construction. Rankings 1–2 and 4–10 are genuine.""")

# ── §3.2 Top 10 excluding artifact ────────────────────────────────────────────
rep("""|    1 | VMP(MDP(sample))    | 1.460 |
|    2 | VMP(SWITCH(sample)) | 1.457 |
|    3 | VMP(SWITCH(LW))     | 1.438 |
|    4 | VMP(MDP(LW))        | 1.437 |
|    5 | VMP(MSR(LW))        | 1.429 |
|    6 | VMP(MSR(sample))    | 1.405 |
|    7 | VMP(MSR_C(LW))      | 1.390 |
|    8 | VMP(BL-Mom(LW))     | 1.346 |
|    9 | VMP(RP(sample))     | 1.330 |
|   10 | VMP(RP(LW))         | 1.306 |

All 10 are VMP variants. The highest-Sharpe base strategy is MSR(LW) at 1.262.""",
    """|    1 | VMP(MDP(LW))        | 1.372 |
|    2 | VMP(MDP(sample))    | 1.368 |
|    3 | VMP(MSR(sample))    | 1.295 |
|    4 | VMP(SWITCH(sample)) | 1.293 |
|    5 | VMP(SWITCH(LW))     | 1.265 |
|    6 | VMP(MSR(LW))        | 1.239 |
|    7 | VMP(HRP(LW))        | 1.232 |
|    8 | VMP(BL-Mom(LW))     | 1.217 |
|    9 | VMP(GMV(LW))        | 1.215 |
|   10 | VMP(GMV(OAS))       | 1.207 |

All 10 are VMP variants. The highest-Sharpe base strategy is MDP(LW) at 1.167.""")

# ── §3.2 Top 5 return ─────────────────────────────────────────────────────────
rep("""|    1 | VMP(BL-Mom(LW))       | 24.97% | 1.346 |
|    2 | BL-Mom(LW)            | 20.01% | 1.049 |
|    3 | VMP(EW)               | 18.13% | 1.253 |
|    4 | VMP(MSR(LW))          | 17.53% | 1.429 |
|    5 | VMP(BL-Eq(sample/LW)) | 16.24% | 1.145 |""",
    """|    1 | VMP(EW)               | 15.31% | 1.133 |
|    2 | VMP(BL-Eq(LW))        | 15.12% | 1.118 |
|    3 | VMP(BL-Mom(LW))       | 14.65% | 1.217 |
|    4 | VMP(RP(LW))           | 14.10% | 1.108 |
|    5 | VMP(RP(sample))       | 14.09% | 1.110 |""")

# ── §3.2 Bottom 5 ────────────────────────────────────────────────────────────
rep("""|   24 | BL-Rev(LW)  |  0.547 |  10.17% |
|   23 | FF3-Mom     |  0.588 |   9.60% |
|   22 | TSMOM(12m)  |  0.626 |   4.05% |
|   21 | FF3-Quality |  0.726 |   6.59% |
|   20 | FF3-Multi   |  0.786 |   6.79% |""",
    """|   24 | BL-Rev(LW)  |  0.663 |  12.09% |
|   23 | FF3-Mom     |  0.685 |  11.03% |
|   22 | TSMOM(12m)  |  0.801 |   5.45% |
|   21 | FF3-Quality |  0.811 |   7.59% |
|   20 | FF3-Multi   |  0.907 |   7.95% |""")

# ── §3.2 Bottom 5 header ─────────────────────────────────────────────────────
rep("**Bottom 5 by Sharpe (base strategies only):**",
    "**Bottom 5 by Sharpe (base strategies, original 24 only):**")

# ── §3.3 Top 10 net table ─────────────────────────────────────────────────────
rep("""The net-cost ranking excludes VMP(GMV(sample)) (gross net Sharpe 1.503 after costs) as a degenerate artifact — see Section 3.2 and Findings 1 and 6.5. VMP(MDP(LW)) is the strongest genuine result net of costs.

| Rank | Strategy                       | Gross Sharpe | Net Sharpe | Turnover |
|-----:|:------------------------------|-------------:|-----------:|---------:|
|    1 | VMP(MDP(LW))                   | 1.437 | 1.400 | 0.79% |
|    2 | VMP(SWITCH(LW))                | 1.438 | 1.381 | 1.98% |
|    3 | VMP(SWITCH(sample))            | 1.457 | 1.337 | 3.37% |
|    4 | VMP(MSR(LW))                   | 1.429 | 1.329 | 4.65% |
|    5 | VMP(MDP(sample))               | 1.460 | 1.307 | 2.60% |
|    6 | VMP(BL-Mom(LW))                | 1.346 | 1.276 | 4.91% |
|    7 | VMP(RP(LW))                    | 1.306 | 1.269 | 0.95% |
|    8 | VMP(EW)                        | 1.253 | 1.253 | 0.00% |
|    9 | GMV(sample)                    | 1.260 | 1.233 | 0.15% |
|   10 | VMP(MSR_C(LW))                 | 1.390 | 1.275 | 5.34% |""",
    """The net-cost ranking excludes VMP(GMV(sample)) (gross Sharpe 1.345, net 1.326 after costs) as a degenerate artifact — see Section 3.2 and Findings 1 and 6.5. VMP(MDP(LW)) is the strongest genuine result net of costs.

| Rank | Strategy                       | Gross Sharpe | Net Sharpe | Turnover |
|-----:|:------------------------------|-------------:|-----------:|---------:|
|    1 | VMP(MDP(LW))                   | 1.372 | 1.336 | 0.78% |
|    2 | VMP(MDP(sample))               | 1.368 | 1.227 | 2.62% |
|    3 | VMP(SWITCH(LW))                | 1.265 | 1.201 | 2.04% |
|    4 | VMP(GMV(LW))                   | 1.215 | 1.177 | 0.54% |
|    5 | VMP(SWITCH(sample))            | 1.293 | 1.173 | 3.28% |
|    6 | VMP(GMV(OAS))                  | 1.207 | 1.169 | 0.48% |
|    7 | VMP(FF3-LowVol)                | 1.165 | 1.140 | 0.39% |
|    8 | VMP(EW)                        | 1.133 | 1.133 | 0.01% |
|    9 | MDP(LW)                        | 1.167 | 1.131 | 0.78% |
|   10 | VMP(MSR(LW))                   | 1.239 | 1.121 | 4.80% |""")

# ── §3.3 Top 5 degradation table ─────────────────────────────────────────────
rep("""| 1 | FF3-Mom                | 0.588 | 0.310 | 20.51% | 0.277 |
| 2 | FF3-Multi              | 0.786 | 0.561 | 7.95% | 0.225 |
| 3 | MSR(sample)            | 0.884 | 0.717 | 5.19% | 0.167 |
| 4 | TSMOM(6m)              | 0.904 | 0.738 | 4.77% | 0.167 |
| 5 | HRP(sample)            | 0.902 | 0.753 | 3.92% | 0.149 |""",
    """| 1 | FF3-Mom                | 0.685 | 0.394 | 20.25% | 0.291 |
| 2 | FF3-Multi              | 0.907 | 0.683 | 7.87% | 0.223 |
| 3 | MSR(sample)            | 0.895 | 0.728 | 5.12% | 0.167 |
| 4 | TSMOM(6m)              | 0.971 | 0.805 | 4.74% | 0.166 |
| 5 | BL-Mom(LW)             | 1.042 | 0.934 | 5.11% | 0.108 |""")

# ── §3.3 Reading paragraph ────────────────────────────────────────────────────
rep("At 10 bps round-trip, cost impact separates into two clear groups. **Low-turnover survivors** (EW, GMV variants,\nHRP, FF3-LowVol) see Sharpe degradation under 0.098 — a negligible penalty that preserves their\nrankings. **High-turnover collapsers** (TSMOM, BL-Mom(LW), FF3-Mom, MSR(sample)) suffer the largest hits:\nFF3-Mom loses 0.277 Sharpe points (median base-strategy degradation: 0.098).\nBL-Mom(LW) is particularly exposed — its 4.91% average daily turnover, driven by continuous\nmomentum-signal rotation across 29 tickers, erodes 0.065 Sharpe points, and\nits net Sharpe drops to 0.985 vs gross 1.049.\n\nRegime-conditional switching strategies (SWITCH variants) sit at a sweet spot: moderate turnover\n(1.98% avg) and net Sharpe 1.125 for SWITCH(LW), which is competitive with\nmany higher-turnover strategies on a net basis. VMP(SWITCH(LW)) net Sharpe 1.381 remains\namong the strongest even after accounting for base-strategy trading costs.",
    "At 10 bps round-trip, cost impact separates into two clear groups. **Low-turnover survivors** (EW, GMV variants,\nHRP, FF3-LowVol) see Sharpe degradation under 0.099 — a negligible penalty that preserves their\nrankings. **High-turnover collapsers** (TSMOM, BL-Mom(LW), FF3-Mom, MSR(sample)) suffer the largest hits:\nFF3-Mom loses 0.291 Sharpe points (median base-strategy degradation across 24 original strategies: 0.099).\nBL-Mom(LW) is particularly exposed — its 5.11% average daily turnover, driven by continuous\nmomentum-signal rotation across 29 tickers, erodes 0.108 Sharpe points, and\nits net Sharpe drops to 0.934 vs gross 1.042.\n\nRegime-conditional switching strategies (SWITCH variants) sit at a sweet spot: moderate turnover\n(2.04% avg) and net Sharpe 1.020 for SWITCH(LW), which is competitive with\nmany higher-turnover strategies on a net basis. VMP(SWITCH(LW)) net Sharpe 1.201 remains\namong the strongest even after accounting for base-strategy trading costs.")

# ── §3.3.4 Stratified costs ───────────────────────────────────────────────────
rep("Under stratified costs, virtually every strategy improves relative to the flat-10-bps\nbenchmark because most assets in the universe are cheaper than 10 bps. The largest\nbeneficiaries are high-turnover strategies concentrated in equities: FF3-Mom net Sharpe\nrises from 0.310 (flat 10 bps) to 0.485 (stratified), and FF3-Multi from 0.561 to\n0.704. Fixed-income-heavy strategies such as GMV(sample) improve marginally (1.233\n→ 1.249) because SHY's 2 bps cost is already far below the flat assumption.\nThe highest-equity-turnover strategies gain the most from stratified pricing:\nFF3-Mom improves by +0.182 Sharpe points (0.405 → 0.587) and FF3-Multi by +0.142\n(0.691 → 0.833) as their frequent 3-bps ETF rebalancing is materially cheaper than\nthe flat-10-bps baseline. The top-5 ranking by stratified net Sharpe (excluding the\nGMV(sample) artifact) shifts slightly: VMP(MDP(LW)) 1.421, VMP(SWITCH(sample)) 1.417,\nVMP(SWITCH(LW)) 1.414, VMP(MDP(sample)) 1.411, VMP(MSR(LW)) 1.384 — VMP(SWITCH(sample))\nmoves from rank 3 to rank 2 as its equity-ETF exposure benefits from the 3 bps ETF\nrate. The qualitative conclusion from §3.3 — that regime-conditional and low-turnover\nstrategies dominate on a cost-adjusted basis — survives unchanged under stratified costs.",
    "Under stratified costs, virtually every strategy improves relative to the flat-10-bps\nbenchmark because most assets in the universe are cheaper than 10 bps. The largest\nbeneficiaries are high-turnover strategies concentrated in equities: FF3-Mom net Sharpe\nrises from 0.394 (flat 10 bps) to 0.587 (stratified), and FF3-Multi from 0.683 to\n0.833. Fixed-income-heavy strategies such as GMV(sample) improve marginally (0.944\n→ 0.953) because SHY's 2 bps cost is already far below the flat assumption.\nThe highest-equity-turnover strategies gain the most from stratified pricing:\nFF3-Mom improves by +0.193 Sharpe points (0.394 → 0.587) and FF3-Multi by +0.150\n(0.683 → 0.833) as their frequent 3-bps ETF rebalancing is materially cheaper than\nthe flat-10-bps baseline. The top-5 ranking by stratified net Sharpe (excluding the\nGMV(sample) artifact) shifts to: VMP(MDP(LW)) 1.359, VMP(MDP(sample)) 1.327,\nVMP(SWITCH(sample)) 1.258, VMP(SWITCH(LW)) 1.242, VMP(EW) 1.133 — MDP-family\nstrategies remain the clear leaders. The qualitative conclusion from §3.3 — that\nregime-conditional and low-turnover strategies dominate on a cost-adjusted basis —\nsurvives unchanged under stratified costs.")

# ── Finding 1 ─────────────────────────────────────────────────────────────────
rep("GMV(sample) reports vol=1.43%, ret=1.80%, Sharpe=1.260 — numbers that look\nattractive until context is added. The optimizer finds SHY (iShares 1–3 Year\nTreasury Bond ETF) as the near-zero-vol asset and corners the portfolio there.\nAt rf=1.5% annualized (rough T-bill average over the period), GMV(sample) Sharpe\ngoes negative: the strategy earns less than cash. Shrinkage breaks the corner:\nGMV(LW) vol=3.23%, Sharpe=0.896 is a real multi-asset portfolio at the cost of\na lower headline Sharpe metric. The OAS estimator gives a similar fix (GMV(OAS)\nvol=2.58%, Sharpe=0.883). Conclusion: Sharpe alone is misleading for GMV(sample);\nany comparison must note the vol level.",
    "GMV(sample) reports vol=3.16%, ret=3.02%, Sharpe=0.958 — numbers that still look\nattractive until context is added. The optimizer finds SHY (iShares 1–3 Year\nTreasury Bond ETF) as the near-zero-vol asset and corners the portfolio there,\nproducing a portfolio that is essentially a cash surrogate. At rf=1.5% annualized\n(rough T-bill average over the period), GMV(sample) Sharpe goes negative: the\nstrategy earns less than cash. Shrinkage breaks the corner:\nGMV(LW) vol=4.01%, Sharpe=0.954 is a more diversified multi-asset portfolio.\nThe OAS estimator gives a similar result (GMV(OAS) vol=3.64%, Sharpe=0.925).\nConclusion: Sharpe alone is misleading for GMV(sample); any comparison must note the vol level.")

# ── Finding 2 ─────────────────────────────────────────────────────────────────
rep("MSR(sample) Sharpe=0.884 is one of the lowest base-strategy Sharpes in the table,\ndespite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates\non whichever asset had the highest sample Sharpe in the 252-day estimation window —\ntypically a low-vol fixed-income ETF that happened to trend up — and the\nout-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization\nshrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.262 (+0.378).\nThis is the largest single-estimator substitution effect in the table.",
    "MSR(sample) Sharpe=0.895 is among the lower base-strategy Sharpes in the table,\ndespite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates\non whichever asset had the highest sample Sharpe in the 252-day estimation window —\ntypically a low-vol fixed-income ETF that happened to trend up — and the\nout-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization\nshrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.059 (+0.164).\nThis is the largest single-estimator substitution effect in the table.")

# ── Finding 3 — complete rewrite ─────────────────────────────────────────────
rep("""## Finding 3 — HRP is the only strategy where sample covariance beats shrinkage

Across all 24 base strategies, shrinkage (LW vs sample) improves Sharpe for every
family except HRP: HRP(sample) Sharpe=0.902 > HRP(LW) Sharpe=0.865 (−0.037).
HRP partitions assets via hierarchical clustering on the correlation matrix and
assigns weights by inverse-variance within clusters. Shrinkage smooths the
pairwise correlations, which blurs the cluster boundaries that HRP's dendrogram
relies on — the information HRP extracts from block structure is degraded, not improved,
by regularization. The same mechanism is absent in all other methods, which work
directly with the covariance matrix rather than its cluster structure.""",
    """## Finding 3 — HRP is approximately invariant to shrinkage choice

In the 29-asset 2003–2026 sample, HRP(sample) Sharpe=1.045 and HRP(LW) Sharpe=1.093,
with a directional difference of −0.047 (LW ahead) that does not clear conventional
significance (Memmel z=−0.67, p=0.506; see §5.4 Finding R4). In the prior 30-asset
2008–2026 sample, the directional sign was opposite (HRP(sample) 0.902 vs HRP(LW)
0.865, Δ=+0.037 for sample over LW), placing the cross-sample reversal squarely within
sampling noise. The conservative conclusion: HRP is approximately invariant to shrinkage
choice in long-sample multi-asset universes, in contrast to the MSR family where
shrinkage produces a directional Sharpe advantage (Finding 2, Δ=+0.164). The structural
intuition — that LW shrinkage smooths the correlation block structure HRP uses for
cluster boundaries — remains plausible as a mechanism but is not empirically supported
at this sample size.""")

# ── Finding 4 ────────────────────────────────────────────────────────────────
rep("In the regime-conditional Sharpe table (14 base strategies × 8 regimes), regime 5\n(low macro level, falling, with positive convexity — a late-cycle or early-recession\nenvironment) produces MSR(sample) conditional Sharpe=1.679 vs MSR(LW) conditional\nSharpe=1.482. Sample wins by +0.197 within this regime. Regime 5 accounts for 779\nof the 4 512 daily observations (~17%). In all other regimes MSR(LW) matches or\nbeats MSR(sample). The switching rule exploits this: SWITCH(v2a) routes R5→MSR(sample)\nspecifically.",
    "In the regime-conditional Sharpe table (12 base strategies × 8 regimes, 29-asset\n2003–2026 sample), Regime 5 (low macro level, falling, with positive convexity — a\nlate-cycle or early-recession environment) produces MSR(sample) conditional\nSharpe=1.392 vs MSR(LW) conditional Sharpe=1.097. Sample wins by +0.295 within this\nregime. Regime 5 accounts for 924 of the 5,868 daily observations (15.8%). In all other\nregimes MSR(LW) matches or beats MSR(sample). The switching rule exploits this:\nSWITCH(v2a) routes R5→MSR(sample) specifically.")

# ── Finding 5 ────────────────────────────────────────────────────────────────
rep("The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all\nothers→MDP(LW), achieving Sharpe=1.179. Regime-conditional analysis on 12\nsingle-strategy baselines showed:\n\n- R0 (1 176 days, 26%): MSR(LW) conditional Sharpe=1.186, best non-SWITCH strategy\n- R5 (779 days, 17%): MSR(sample) conditional Sharpe=1.679, best non-SWITCH strategy\n\nSubstituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)\nyields SWITCH(v2a) Sharpe=1.340 (+0.161 vs v1). V2a achieves this with only two\ntargeted swaps and no change to the default rule — a tractable candidate refinement.\nThe empirical gain does not clear statistical significance at conventional levels\n(z=0.91, p=0.37; see §5.1), so we retain v2a as a candidate improvement rather than\na documented one.",
    "The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all\nothers→MDP(LW), achieving Sharpe=1.080 on the 29-asset 2003–2026 sample.\nRegime-conditional analysis on 12 single-strategy baselines over the training period\n(2003–2022) identified the v2a routing:\n\n- R0 (1,603 days, 27%): MSR(LW) was selected as the R0 target from training-period analysis; on the full 2003–2026 sample MDP(LW) leads R0 at conditional Sharpe=1.326 (MSR(LW) at 0.869), but this post-hoc observation does not retroactively change the training-derived rule\n- R5 (924 days, 15.8%): MSR(sample) conditional Sharpe=1.392 (full 2003–2026 sample), best non-SWITCH strategy in R5\n\nSubstituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)\nyields SWITCH(v2a) Sharpe=1.514 (+0.434 vs v1). The empirical gain clears statistical\nsignificance at the 5% level (Memmel z=2.05, p=0.040; see §5.3 Finding R3),\nrepresenting the strongest regime-conditional evidence in the study.")

# ── Finding 6 ────────────────────────────────────────────────────────────────
rep("VMP lifts Sharpe for every one of the original 24 strategy families without exception\n(24/24 improvements). The lift ranges from +0.145 (FF3-Mom) to +0.521 (MSR(sample)).\nThe magnitude is inversely correlated with how well the base strategy already manages\nvolatility clustering: MSR(sample) has the largest lift because its concentration-driven\nvol spikes are the most amenable to scaling back. HRP variants have the smallest lifts\n(+0.165, +0.162) because HRP's cluster-based weighting already produces smoother\nrealized vol. Median lift across all 24 strategies: ≈+0.270 Sharpe points.",
    "VMP lifts Sharpe for every one of the original 24 strategy families without exception\n(24/24 improvements). The lift ranges from +0.119 (FF3-Mom) to +0.400 (MSR(sample)).\nThe magnitude is inversely correlated with how well the base strategy already manages\nvolatility clustering: MSR(sample) has the largest lift because its concentration-driven\nvol spikes are the most amenable to scaling back. HRP variants have the smallest lifts\namong traditional strategies (+0.130, +0.139) because HRP's cluster-based weighting\nalready produces smoother realized vol. Median lift across all 24 strategies: +0.194\nSharpe points.")

# ── Finding 7 ────────────────────────────────────────────────────────────────
rep("VMP(MSR(sample)) Sharpe=1.405 surpasses raw MSR(LW) Sharpe=1.262 (+0.143). The vol\nmanagement overlay applied to a concentrated, over-fit portfolio reduces exposure\nprecisely during the high-vol episodes that the overfit concentration creates, producing\nbetter realized risk-adjusted returns than shrinkage alone. Practically: a cheaper\nestimator (no LW computation) with VMP on top outperforms the more expensive estimator\nwithout VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.533 > GMV(LW)\nSharpe=0.896, and VMP(MDP(sample)) Sharpe=1.460 > MDP(LW) Sharpe=1.182.",
    "VMP(MSR(sample)) Sharpe=1.295 surpasses raw MSR(LW) Sharpe=1.059 (+0.236). The vol\nmanagement overlay applied to a concentrated, over-fit portfolio reduces exposure\nprecisely during the high-vol episodes that the overfit concentration creates, producing\nbetter realized risk-adjusted returns than shrinkage alone. Practically: a cheaper\nestimator (no LW computation) with VMP on top outperforms the more expensive estimator\nwithout VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.345 > GMV(LW)\nSharpe=0.954, and VMP(MDP(sample)) Sharpe=1.368 > MDP(LW) Sharpe=1.167.")

# ── Finding 8 ────────────────────────────────────────────────────────────────
rep("TSMOM(12m) Sharpe=0.626 is the lowest base-strategy Sharpe in the table.",
    "TSMOM(12m) Sharpe=0.801 is among the weaker base-strategy Sharpes in the table.")
rep("This asymmetry is partially mitigated at shorter\nlookback: TSMOM(6m) Sharpe=0.904. VMP(TSMOM(12m)) Sharpe=0.976 (+0.350) achieves\nEW-comparable performance by scaling down exposure during the high-vol drawdown\nperiods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is\nnear the median of all 62 strategies and adds little over VMP(EW) Sharpe=1.253.",
    "This asymmetry is partially mitigated at shorter\nlookback: TSMOM(6m) Sharpe=0.971. VMP(TSMOM(12m)) Sharpe=1.059 (+0.258) achieves\nEW-comparable performance by scaling down exposure during the high-vol drawdown\nperiods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is\nnear the median of all 62 strategies and adds little over VMP(EW) Sharpe=1.133.")

# ── Finding 9 ────────────────────────────────────────────────────────────────
rep("BL-Mom(LW) annualized return=20.01% is the highest base-strategy return, driven by\nmomentum-tilted Black-Litterman views rotating into high-momentum assets during\ntrending periods. The cost is severe: max drawdown=−50.85%, the worst in the table.\nVMP(BL-Mom(LW)) return=24.97% (+4.96 pp) with max drawdown compressed to −36.01%\n(+14.84 pp improvement). The Calmar ratio improves from 0.394 to 0.693. No other\nstrategy pair in the table reaches the 20%+ return threshold. The high drawdown\nremains a practical barrier: the strategy lost more than half its value peak-to-trough\neven after VMP, unsuitable for most risk budgets without hard drawdown stops.",
    "BL-Mom(LW) annualized return=12.57% is among the higher base-strategy returns,\ndriven by momentum-tilted Black-Litterman views rotating into high-momentum assets\nduring trending periods. The cost is drawdown risk: max drawdown=−21.34%.\nVMP(BL-Mom(LW)) return=14.65% (+2.08 pp); the VMP overlay does not substantially\ncompress the drawdown (max drawdown=−21.84%) because the worst periods align with\nmomentum reversals rather than pure volatility spikes. The Calmar ratio improves from\n0.589 to 0.671. BL-Mom(LW) is no longer the return leader in the 29-asset 2003–2026\nsample; VMP(EW) leads at 15.31% (reflecting the strong 2003–2007 equity expansion\ncaptured by the extended sample). The risk profile is substantially more benign than\nthe prior 30-asset study (former maxdd=−50.85%) because BTC-USD exclusion removes the\nmost extreme drawdown contributor.")

# ── Finding 10 ───────────────────────────────────────────────────────────────
rep("Both report ret=12.76%,\nvol=14.77%, Sharpe=0.887, maxdd=−37.86%.",
    "Both report ret=12.48%,\nvol=13.92%, Sharpe=0.915, maxdd=−37.86%.")

# ── Finding 11 ───────────────────────────────────────────────────────────────
rep("FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)\nachieves Sharpe=0.936 with vol=3.39% and ret=3.17%. The risk-adjusted performance is\ncompetitive with EW (Sharpe=0.976) but the absolute return is too low for most\ninstitutional mandates. VMP lifts Sharpe to 1.146 (ret=3.77%) but the vol\nstabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol\nearns 3.17 cents per dollar per year. The anomaly is confirmed within this universe\nbut requires 3–4× leverage to match EW on absolute return while preserving the\nSharpe advantage.",
    "FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)\nachieves Sharpe=1.021 with vol=4.25% and ret=4.34%. The risk-adjusted performance is\ncompetitive with EW (Sharpe=0.924) but the absolute return is too low for most\ninstitutional mandates. VMP lifts Sharpe to 1.165 (ret=4.59%) but the vol\nstabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol\nearns 4.34 cents per dollar per year. The anomaly is confirmed within this universe\nbut requires 3–4× leverage to match EW on absolute return while preserving the\nSharpe advantage.")

# ── Finding 12 ───────────────────────────────────────────────────────────────
rep("The improvements from regime-conditional switching (SWITCH(v2a) Sharpe=1.340 vs\nSWITCH(LW) Sharpe=1.179, Δ=+0.161) and from VMP on top of the original rule\n(VMP(SWITCH(LW)) Sharpe=1.438 vs SWITCH(LW) Sharpe=1.179, Δ=+0.259) are comparable\nin magnitude. Both approaches target the same underlying risk — volatility clustering\nand regime-dependent return distribution — through different mechanisms. Stacking them\n(applying VMP to v2a) yields Sharpe=1.588 and Calmar=0.906, the best combined\nperformance in the study, but the marginal gain from the second layer is subadditive:\nVMP alone on the v1 rule gives +0.259, regime switching alone gives +0.161, combined\ngives +0.409, not +0.420. The two refinements share roughly 10% of their variance\nexplained.",
    "The improvement from regime-conditional switching (SWITCH(v2a) Sharpe=1.514 vs\nSWITCH(LW) Sharpe=1.080, Δ=+0.434) dominates the improvement from VMP on top of the\noriginal rule (VMP(SWITCH(LW)) Sharpe=1.265 vs SWITCH(LW) Sharpe=1.080, Δ=+0.184) in\nthe 29-asset 2003–2026 sample. Both approaches target the same underlying risk —\nvolatility clustering and regime-dependent return distribution — through different\nmechanisms. Stacking them (applying VMP to v2a) yields Sharpe=1.660 and Calmar=0.941,\nthe best combined performance in the study, but the marginal gain from the second layer\nis subadditive: VMP alone on the v1 rule gives +0.184, regime switching alone gives\n+0.434, combined gives +0.580, not +0.618. The two refinements share roughly 6% of\ntheir incremental Sharpe.")

# ── Finding 13 ───────────────────────────────────────────────────────────────
rep("The three strongest base strategies net of costs are GMV(sample), MSR(LW), MDP(LW), all low-turnover\nstrategies where the optimizer changes weights only modestly between rebalances. The three\nweakest net-of-cost base strategies are TSMOM(12m), BL-Rev(LW), FF3-Mom, where frequent weight rotation\nor large momentum-driven tilts generate daily turnover high enough to erode a meaningful\nshare of gross Sharpe. The median gross-to-net Sharpe degradation across all 24 base\nstrategies is 0.098 Sharpe points; the maximum degradation is 0.277\n(FF3-Mom). Finding 6 (VMP improves all 24/24 original base strategies) survives qualitatively on\na net basis: every VMP variant's net Sharpe exceeds the corresponding base strategy's net\nSharpe for the original 24 families, since the VMP overlay adds Sharpe by scaling down\nduring high-vol periods and the base-strategy turnover cost is the same for both. The\nFF3-Mom-LS exception (VMP worsens an already near-zero-Sharpe long-short strategy) does\nnot affect the original 24-family result. Finding 9 (BL-Mom return leadership)\ndoes not survive the cost screen: BL-Mom(LW) gross Sharpe=1.049 falls to net\nSharpe=0.985 at 4.91% average daily turnover, dropping out of the\ntop-10 net ranking. Regime-conditional switching strategies (SWITCH variants) sit at a cost\nsweet spot — their turnover (1.98% avg for SWITCH(LW)) is moderate because\nthe regime signal is monthly and most regime-to-strategy assignments persist for many days\n— and they retain their strong net-Sharpe rankings. VMP(SWITCH(LW)) net Sharpe\n1.381 is among the best strategies on a fully net-of-cost basis. Under\nasset-class-stratified costs (§3.3.4), high-equity-turnover strategies gain more\nthan low-turnover ones relative to the flat baseline, as equity ETF rates (3 bps)\nfall far below the flat 10-bps assumption; the qualitative Finding 13 ranking —\nregime-conditional and low-turnover strategies as implementability leaders — holds\nunder both cost regimes.",
    "The three strongest base strategies net of costs are MDP(LW), EW, and HRP(LW) — all\nlow-turnover strategies where the optimizer changes weights only modestly between\nrebalances. The three weakest net-of-cost base strategies are BL-Rev(LW), FF3-Mom, and\nTSMOM(12m), where frequent weight rotation or large momentum-driven tilts generate daily\nturnover high enough to erode a meaningful share of gross Sharpe. The median gross-to-net\nSharpe degradation across the 24 original base strategies is 0.099 Sharpe points; the\nmaximum degradation is 0.291 (FF3-Mom). Finding 6 (VMP improves all 24/24 original base\nstrategies) survives qualitatively on a net basis: every VMP variant's net Sharpe exceeds\nthe corresponding base strategy's net Sharpe for the original 24 families, since the VMP\noverlay adds Sharpe by scaling down during high-vol periods and the base-strategy turnover\ncost is the same for both. The FF3-Mom-LS exception (VMP worsens an already near-zero-Sharpe\nlong-short strategy) does not affect the original 24-family result. BL-Mom(LW) gross\nSharpe=1.042 falls to net Sharpe=0.934 at 5.11% average daily turnover, dropping out of\nthe top-10 net ranking. Regime-conditional switching strategies (SWITCH variants) sit at a\ncost sweet spot — their turnover (2.04% avg for SWITCH(LW)) is moderate because the regime\nsignal is monthly and most regime-to-strategy assignments persist for many days — and they\nretain their strong net-Sharpe rankings. VMP(SWITCH(LW)) net Sharpe 1.201 is among the\nbest strategies on a fully net-of-cost basis. Under asset-class-stratified costs (§3.3.4),\nhigh-equity-turnover strategies gain more than low-turnover ones relative to the flat\nbaseline; the qualitative Finding 13 ranking — regime-conditional and low-turnover\nstrategies as implementability leaders — holds under both cost regimes.")

# ── Finding 14 ───────────────────────────────────────────────────────────────
rep("Activating the short leg in a heterogeneous 29-asset universe does not rescue\nmomentum strategies — it worsens their performance. TSMOM-LS(12m) achieves\nSharpe 0.414, materially below TSMOM(12m) long-only at 0.626; FF3-Mom-LS\nproduces Sharpe 0.088 gross and −0.273 net of 10 bps, making it the weakest\nstrategy in the study on a cost-adjusted basis. The mechanism is compositional:\nin a universe spanning equities, fixed income, and commodities, assets with\nnegative 12-month momentum frequently include bonds and commodities in the midst\nof their respective drawdown periods — asset classes that subsequently\nmean-revert and impose losses on the short leg. This directly contradicts the\nconclusions of @moskowitz2012time, whose TSMOM results were derived from a\nfutures universe dominated by equity index and currency contracts where the short\nleg captures genuine momentum losers rather than structurally mean-reverting\nasset classes. The exception is BL-Mom-LS(LW) (Sharpe 0.991 vs. BL-Mom(LW)\nlong-only at 1.049), which uses Bayesian view-tilting to selectively short\nunderweighted assets and avoids the crude composition problem of rank-based\nshorting; the modest Sharpe gap reflects a deliberate risk-profile trade-off\nrather than a strategy failure.",
    "Activating the short leg in a heterogeneous 29-asset universe does not rescue\nmomentum strategies — in most cases it worsens their performance. TSMOM-LS(12m)\nachieves Sharpe 0.645, below TSMOM(12m) long-only at 0.801; FF3-Mom-LS produces\nSharpe 0.103 gross and −0.306 net of 10 bps, making it the weakest strategy in the\nstudy on a cost-adjusted basis. The mechanism is compositional: in a universe spanning\nequities, fixed income, and commodities, assets with negative 12-month momentum\nfrequently include bonds and commodities in the midst of their respective drawdown\nperiods — asset classes that subsequently mean-revert and impose losses on the short\nleg. This directly contradicts the conclusions of @moskowitz2012time, whose TSMOM results\nwere derived from a futures universe dominated by equity index and currency contracts\nwhere the short leg captures genuine momentum losers rather than structurally\nmean-reverting asset classes. The exception is BL-Mom-LS(LW) (Sharpe 0.904 vs.\nBL-Mom(LW) long-only at 1.042), which uses Bayesian view-tilting to selectively short\nunderweighted assets and avoids the crude composition problem of rank-based shorting;\nthe modest Sharpe gap reflects a deliberate risk-profile trade-off — BL-Mom-LS vol\ncollapses from 12.07% to 4.65% and max drawdown from −21.34% to −11.87%, making the\nL/S form a qualitatively different instrument for risk-budgeted mandates.")

# ── Finding 15 — rewrite garbled passage ─────────────────────────────────────
rep("The prior 30-ticker study included BTC-USD with a forward-fill survivorship bias:\n636 trading days before the asset's inception (2010-07-13) carried a forward-filled\n2010 price, representing 13.8% of the period. BTC-USD is excluded entirely from\nthe current 29-asset rebuild, extending the sample to 2003-01-02. Prior analysis\n(8-ticker sensitivity with BTC-USD excluded entirely) showed a median Sharpe delta of\n+0.229 attributable to BTC inclusion — BTC-USD is excluded entirely for Survivorship hygiene\nrather than to minimise return; the loss is accepted in exchange for a longer,\neconomically richer Survivorship-clean sample covering the dot-com recovery and\npre-GFC expansion.\nThe headline findings from the 30-asset study (VMP universal lift, MSR Michaud overfit,\nHRP sample-beat-shrinkage) all survive in the 29-asset comparison, with the full-sample\nSharpe numbers updated to the 2003–2026 window in the revised Appendix A table.",
    "BTC-USD is excluded entirely from the current 29-asset rebuild to eliminate the\nforward-fill survivorship bias documented in the prior 30-ticker study, where the 636\ntrading days before BTC's inception (2010-07-13) carried a forward-filled 2010 price,\nrepresenting 13.8% of that period. The exclusion is a deliberate sacrifice: prior\n8-strategy sensitivity analysis on the no-BTC sub-universe showed a median Sharpe delta\nof +0.229 attributable to BTC inclusion, indicating BTC was a material contributor to\nportfolio returns rather than noise. The loss is accepted in exchange for cleaner\nsurvivorship hygiene and the 5-year sample extension to 2003, which captures the\ndot-com recovery and the pre-GFC expansion. The headline findings from the 30-asset\nstudy (VMP universal lift, MSR Michaud overfit, regime-conditional structure) all\nsurvive in the 29-asset comparison; Finding 3 (HRP shrinkage exception) is reframed\nas near-invariance based on the new sample's empirical cross-sample sign reversal.")

# ── §5.1 Finding R1 ──────────────────────────────────────────────────────────
rep("Memmel test: $z=2.78$, $p=0.005$. The largest single-estimator\nsubstitution effect in the study survives statistical scrutiny; shrinkage dominance\nover sample covariance for MSR is a reliable empirical result at this sample size.",
    "Memmel test on the 29-asset 2003–2026 sample: $z=1.13$, $p=0.259$. The directional\nfinding — LW shrinkage improves MSR — is consistent across both samples tested but\ndoes not reach significance on the extended 23.3-year sample (former: $z=2.78$,\n$p=0.005$ on 30-asset 2008–2026). The longer pre-GFC period (2003–2007) dilutes the\nshrinkage benefit. The directional conclusion remains consistent.")

# Replace old "MSR Michaud overfit, MSR(LW)−MSR(sample)=+0.378" in Finding R1 header
rep("**Finding 2** (MSR Michaud overfit, MSR(LW)−MSR(sample)=+0.378 Sharpe) is highly\nsignificant.",
    "**Finding 2** (MSR Michaud overfit, MSR(LW)−MSR(sample)=+0.164 Sharpe on the 29-asset\n2003–2026 sample) is tested as follows.")

# ── §5.2 Finding R2: block-bootstrap ────────────────────────────────────────
rep("Block-bootstrap 95% confidence\nintervals for the genuine top-10 strategies (excluding the VMP(GMV(sample)) artifact;\nsee Section 3.2) confirm that all VMP variants' intervals lie above Sharpe 0.60, with\nthe leading three non-artifact strategies (VMP(MDP(sample)), VMP(SWITCH(sample)),\nVMP(SWITCH(LW))) spanning roughly [0.73, 2.06], [0.96, 2.01], and [0.85, 2.00]\nrespectively. VMP(GMV(sample)) bootstrap CIs [0.65, 2.25] are excluded from\ncomparative inference because the underlying base strategy is a degenerate cash corner.",
    "Block-bootstrap 95% confidence\nintervals for the genuine top-10 strategies (excluding the VMP(GMV(sample)) artifact;\nsee Section 3.2) confirm that all VMP variants' intervals lie above Sharpe 0.60, with\nthe leading three non-artifact strategies (VMP(MDP(LW)), VMP(MDP(sample)),\nVMP(MSR(sample))) spanning roughly [0.73, 2.06], [0.79, 1.97], and [0.70, 1.90]\nrespectively. VMP(GMV(sample)) bootstrap CIs are excluded from\ncomparative inference because the underlying base strategy is a degenerate cash corner.")

# ── §5.3 Finding R3 — reframe (SWITCH now significant) ───────────────────────
rep("**Finding 5** (SWITCH(v2a) improvement over SWITCH(LW), $\\Delta=+0.161$) does not\nclear statistical significance at conventional levels ($z=0.91$, $p=0.37$).\nThe methodological contribution — the regime-conditional Sharpe analysis identifying\nMSR(LW)→R0 and MSR(sample)→R5 as the dominant non-SWITCH strategies within their\nrespective regimes — remains a valid empirical observation. The quantitative Sharpe\ngain itself, however, falls within sampling noise. We retain v2a as a candidate\nrefinement rather than a documented improvement.",
    "**Finding 5** (SWITCH(v2a) improvement over SWITCH(LW), $\\Delta=+0.434$ on the\n29-asset 2003–2026 sample) clears statistical significance at the 5% level\n($z=2.05$, $p=0.040$). The regime-conditional Sharpe analysis identifying\nMSR(sample)→R5 as the dominant strategy in Regime 5 is now a statistically supported\nfinding. We elevate v2a from candidate refinement to documented improvement on this\nsample.")

# ── Insert Finding R4 after R3 ────────────────────────────────────────────────
rep("# Out-of-Sample Validation {#sec:oos}",
    """## Finding R4 — HRP Memmel test: near-invariance confirmed

**Finding 3** (HRP near-invariance to shrinkage) is directly tested via Memmel (2003)
paired contrast. On the 29-asset 2003–2026 sample (T=5,868 daily observations):
HRP(sample) Sharpe=1.045, HRP(LW) Sharpe=1.093, Δ=−0.047 (LW marginally ahead).
Memmel z=−0.67, p=0.506. The sign is opposite to the prior 30-asset 2008–2026 study
where HRP(sample) led by +0.037 (sample ahead). This cross-sample sign reversal,
combined with the non-significant test on both samples, supports the near-invariance
conclusion: neither covariance estimator dominates for HRP in a statistically meaningful
way across long multi-asset samples.

# Out-of-Sample Validation {#sec:oos}""")

# ── OOS: update SWITCH comparison ────────────────────────────────────────────
rep("1.514 vs SWITCH(LW) v1 at 1.110 (Δ=+0.404).",
    "1.514 vs SWITCH(LW) v1 at 1.080 (Δ=+0.434).")

# ── OOS: training-only rule ───────────────────────────────────────────────────
rep("The training-only-derived v2a rule (R0→MSR(LW), R5→MSR(sample), others→MDP(LW)) is\nidentical in mapping to the v2a rule derived from the prior full-sample analysis —\nthe regime-to-strategy structure is stable across derivation windows. SWITCH(v2a)\nachieves full-sample Sharpe 1.514 vs SWITCH(LW) v1 at 1.110 (Δ=+0.404).",
    "The training-only-derived v2a rule (R0→MSR(LW), R5→MSR(sample), others→MDP(LW)) is\nidentical in mapping to the v2a rule derived from the prior full-sample analysis —\nthe regime-to-strategy structure is stable across derivation windows. SWITCH(v2a)\nachieves full-sample Sharpe 1.514 vs SWITCH(LW) v1 at 1.080 (Δ=+0.434).")

# ── OOS: HRP bullet ────────────────────────────────────────────────────────────
rep("**HRP shrinkage exception** — HRP(sample) continues to outperform HRP(LW) on the test\nperiod, preserving Finding 3. The cluster-boundary degradation from shrinkage is a\nstructural property that does not depend on the training window.",
    "**HRP near-invariance** — HRP(sample) and HRP(LW) produce similar performance on both\nthe full sample and the test period. The near-invariance finding (Finding 3, Finding R4)\nis consistent across sub-periods: neither estimator dominates reliably.")

# ── Discussion §7.1 VMP Volatility Management: median lift ───────────────────
rep("The lift is not uniform, however. Strategies\nthat already embody volatility management — HRP through cluster-based inverse-variance\nweighting, RP through equal risk contribution — show the smallest gains (+0.162,\n+0.165).",
    "The lift is not uniform, however. Strategies\nthat already embody volatility management — HRP through cluster-based inverse-variance\nweighting, RP through equal risk contribution — show the smallest gains (+0.130–+0.139,\n+0.181–+0.183).")
rep("MSR(sample), TSMOM(12m) — show the largest gains (+0.521, +0.350).",
    "MSR(sample), TSMOM(12m) — show the largest gains (+0.400, +0.258).")
rep("confirms the theoretical intuition of @moreira2017volatility: VMP adds the most value\nwhere the base strategy's realized volatility is most forecastable, and thus most\nreducible.",
    "confirms the theoretical intuition of @moreira2017volatility: VMP adds the most value\nwhere the base strategy's realized volatility is most forecastable, and thus most\nreducible. The median lift of +0.194 Sharpe points (across 24 original strategies)\nis economically significant and consistent across the six strategy families.")

# ── Discussion §7.2 Shrinkage vs. Structure ──────────────────────────────────
rep("The exception is hierarchical structure. HRP(sample) Sharpe=0.902 beats HRP(LW)\nSharpe=0.865 because shrinkage smooths the pairwise correlations that HRP's dendrogram\nrelies on to define cluster boundaries (Finding 3). This is a structural incompatibility:\nLW shrinkage pulls correlations toward a common mean, blurring the block structure that\nencodes economic asset groupings. The mechanism is absent in all other families because\nthey operate directly on $\\Sigma$ rather than its cluster topology. The practical\nimplication is that HRP should be paired with sample (or alternatively lightly\nregularized) covariance, while all other families benefit from full shrinkage.",
    "The exception is hierarchical structure: HRP shows near-invariance to shrinkage choice\n(Finding 3, Finding R4). HRP(sample) Sharpe=1.045, HRP(LW) Sharpe=1.093 — a\nnon-significant difference (Memmel p=0.506) with opposite sign to the prior 30-asset\nsample. The structural intuition — that LW shrinkage blurs the block correlations HRP's\ndendrogram relies on for cluster boundaries — remains plausible but is not empirically\nreliable at this sample size. The practical implication is that HRP's performance is\napproximately invariant to shrinkage choice; practitioners may use either estimator.\nMean-variance families (MSR, GMV, MDP) consistently benefit from LW shrinkage.")

# ── §7.2 interaction with regime ─────────────────────────────────────────────
rep("VMP and SWITCH(v2a) both respond to volatility regimes — VMP through a\ndaily multiplicative scalar, regime switching through a monthly strategy replacement.\nThe two mechanisms share roughly 10% of their variance explained, confirming they are\npartial substitutes. Stacking both yields the best combined Sharpe in the study (1.588\nfor VMP(SWITCH(v2a))), but the gain is subadditive: the marginal value of the second\nlayer diminishes when the first already adapts to vol regimes. Practitioners face a\ncomplexity-cost tradeoff: the VMP overlay alone over a simple base strategy (e.g.,\nVMP(MDP(LW)) Sharpe=1.437) achieves near-top performance without the regime\nclassification infrastructure.",
    "VMP and SWITCH(v2a) both respond to volatility regimes — VMP through a\ndaily multiplicative scalar, regime switching through a monthly strategy replacement.\nThe two mechanisms share roughly 6% of their incremental Sharpe, confirming they are\npartial substitutes. Stacking both yields the best combined Sharpe in the study (1.660\nfor VMP(SWITCH(v2a))), but the gain is subadditive: the marginal value of the second\nlayer diminishes when the first already adapts to vol regimes. Practitioners face a\ncomplexity-cost tradeoff: the VMP overlay alone over a simple base strategy (e.g.,\nVMP(MDP(LW)) Sharpe=1.372) achieves near-top performance without the regime\nclassification infrastructure.")

# ── §7.4 18-year → 23.3-year ────────────────────────────────────────────────
rep("It is a ranking of average performance over a specific 18-year\nwindow that happened to include a particular sequence of macro regimes.",
    "It is a ranking of average performance over a specific 23.3-year\nwindow that happened to include a particular sequence of macro regimes.")

# ── §7.4 MSR(LW) sub-period numbers ─────────────────────────────────────────
rep("MSR(LW), the best-performing non-degenerate base strategy in the full-sample table\n(Sharpe=1.262), ranges from 0.34 in 2008–2012 to 2.48 in 2013–2017 and back to 0.58\nin 2018–2022. This within-strategy range of 2.14 Sharpe points dwarfs the full-sample\ncross-strategy spread of approximately 0.98 points (from BL-Rev(LW) at 0.547 to\nMSR(LW) at 1.262). VMP(MSR(LW)) similarly swings from 0.48 to 2.46 across the same\nwindows.",
    "MSR(LW), one of the better-performing non-degenerate base strategies in the full-sample\ntable (Sharpe=1.059), ranges from 0.53 in 2008–2012 to 1.51 in 2013–2017 and 0.41 in\n2018–2022 (from the Appendix B sub-period table). This within-strategy range of 1.10\nSharpe points is large relative to the full-sample cross-strategy spread of approximately\n0.50 points (from BL-Rev(LW) at 0.663 to MDP(LW) at 1.167). VMP(MSR(LW)) similarly\nswings from 0.70 to 1.46 across the same windows.")

# ── §8 Conclusion: median VMP lift ───────────────────────────────────────────
rep("the VMP overlay is a universal\nSharpe-improver with a median lift of +0.270 that works",
    "the VMP overlay is a universal\nSharpe-improver with a median lift of +0.194 that works")

# ── §8 Conclusion: HRP shrinkage statement ───────────────────────────────────
rep("(2) Ledoit-Wolf shrinkage is\nuniversally beneficial except for HRP, where it degrades cluster boundary information;",
    "(2) Ledoit-Wolf shrinkage is consistently beneficial for mean-variance families; HRP\nshows near-invariance to shrinkage choice with no statistically detectable difference\nacross both samples (Finding 3, Finding R4);")

# ── §8 Long-short extensions: vol/drawdown comparison ────────────────────────
rep("BL-Mom-LS(LW) achieves Sharpe 0.991 with vol\n5.56% and max drawdown −20.30%, a dramatically improved risk profile vs. BL-Mom(LW)\n(vol 19.12%, drawdown −50.85%), at the cost of lower absolute return (5.50% vs.\n20.01%).",
    "BL-Mom-LS(LW) achieves Sharpe 0.904 with vol\n4.65% and max drawdown −11.87%, a dramatically improved risk profile vs. BL-Mom(LW)\n(vol 12.07%, drawdown −21.34%), at the cost of lower absolute return (4.18% vs.\n12.57%).")

# ── §8 Transaction cost discussion: BL-Mom-LS ────────────────────────────────
rep("and its Sharpe\n(0.991) is nearly identical to BL-Mom(LW) long-only (1.049), yet vol collapses\nfrom 19.12% to 5.56% and max drawdown from −50.85% to −20.30%, making the L/S\nform a qualitatively different instrument for risk-budgeted mandates.",
    "and its Sharpe\n(0.904) is near BL-Mom(LW) long-only (1.042), yet vol collapses\nfrom 12.07% to 4.65% and max drawdown from −21.34% to −11.87%, making the L/S\nform a qualitatively different instrument for risk-budgeted mandates.")

# ── FF3-Mom-LS net Sharpe ────────────────────────────────────────────────────
rep("FF3-Mom-LS produces Sharpe 0.088 gross and −0.273 net of 10 bps, confirming",
    "FF3-Mom-LS produces Sharpe 0.103 gross and −0.306 net of 10 bps, confirming")

# ── Write back ────────────────────────────────────────────────────────────────
MD.write_text(text)
print(f"\nUpdated results.md — {len(text):,} characters")
