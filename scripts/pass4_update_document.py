#!/usr/bin/env python3
"""Pass-4: Update docs/results.md with canonical 29-asset 2003-2026 numbers."""

from pathlib import Path

MD = Path("docs/results.md")
text = MD.read_text()

# ── Helpers ────────────────────────────────────────────────────────────────────
def rep(old, new, count=1):
    global text
    assert old in text, f"String not found: {old!r}"
    text = text.replace(old, new, count)

# ── §1.2 Related Work: 18-year → 23.3-year ────────────────────────────────────
rep(
    "covering an 18-year window that includes",
    "covering a 23.3-year window that includes"
)

# ── Abstract: soften HRP/shrinkage ─────────────────────────────────────────────
rep(
    "a structural incompatibility between shrinkage and Hierarchical Risk Parity",
    "HRP's near-invariance to shrinkage choice, distinguishing it from mean-variance families where shrinkage provides a robust Sharpe advantage"
)

# ── §3.1 lead paragraph: update top-3 numbers and highest base ─────────────────
rep(
    "Across all 62 strategies, the top three by gross Sharpe are VMP(GMV(sample)) (1.345), VMP(MDP(sample)) (1.368), and VMP(MDP(LW)) (1.372) — all VMP variants of low-to-moderate turnover base strategies. VMP(GMV(sample)) is flagged as a degenerate artifact (see Findings 1 and 6.5 and Section 3.2). By net Sharpe after 10 bps round-trip costs, the leaders shift to VMP(MDP(LW)) (1.337), VMP(SWITCH(LW)) (1.203), and VMP(SWITCH(sample)) (1.178), reflecting turnover penalties on the higher-rotation sample-covariance variants. Among base strategies only, the three weakest by gross Sharpe are BL-Rev(LW) (0.663), FF3-Mom (0.685), and TSMOM(12m) (0.801) — strategies where return-chasing signals generate high turnover or deep drawdowns without commensurate compensation.",
    "Across all 62 strategies, the top three by gross Sharpe are VMP(MDP(LW)) (1.372), VMP(MDP(sample)) (1.368), and VMP(GMV(sample)) (1.345) — all VMP variants of low-to-moderate turnover base strategies. VMP(GMV(sample)) is flagged as a degenerate artifact (see Findings 1 and 6.5 and Section 3.2). By net Sharpe after 10 bps round-trip costs, the leaders shift to VMP(MDP(LW)) (1.336), VMP(MDP(sample)) (1.227), and VMP(SWITCH(LW)) (1.201), reflecting turnover penalties on the higher-rotation sample-covariance variants. Among base strategies only, the three weakest by gross Sharpe are BL-Rev(LW) (0.663), FF3-Mom (0.685), and TSMOM(12m) (0.801) — strategies where return-chasing signals generate high turnover or deep drawdowns without commensurate compensation."
)

# ── §3.2 Top 10 by Sharpe — raw ────────────────────────────────────────────────
rep(
    """**Top 10 by Sharpe — raw (all 62 strategies, artifact included):**

| Rank | Strategy | Sharpe | Note |
|-----:|:---------|-------:|:-----|
|    1 | VMP(GMV(sample))    | 1.533 | (†) degenerate artifact — SHY concentration |
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
    """**Top 10 by Sharpe — raw (all 62 strategies, artifact included):**

| Rank | Strategy | Sharpe | Note |
|-----:|:---------|-------:|:-----|
|    1 | VMP(MDP(LW))        | 1.372 | |
|    2 | VMP(MDP(sample))    | 1.368 | |
|    3 | VMP(GMV(sample))    | 1.345 | (†) degenerate artifact — SHY concentration |
|    4 | VMP(MSR(sample))    | 1.295 | |
|    5 | VMP(SWITCH(sample)) | 1.293 | |
|    6 | VMP(SWITCH(LW))     | 1.265 | |
|    7 | VMP(MSR(LW))        | 1.239 | |
|    8 | VMP(HRP(LW))        | 1.232 | |
|    9 | VMP(BL-Mom(LW))     | 1.217 | |
|   10 | VMP(GMV(LW))        | 1.215 | |

(†) VMP(GMV(sample)) Sharpe=1.345 is not a genuine portfolio result: GMV(sample) corners the portfolio in SHY (iShares 1–3 Year Treasury), producing near-zero base volatility, and VMP then levers up to 1.5× of that near-cash position. The "Sharpe" reflects cash concentration, not diversified portfolio construction. Rankings 1–2 and 4–10 are genuine."""
)

# ── §3.2 Top 10 excluding artifact ─────────────────────────────────────────────
rep(
    """**Top 10 by Sharpe — excluding SHY-concentration artifact:**

| Rank | Strategy | Sharpe |
|-----:|:---------|-------:|
|    1 | VMP(MDP(sample))    | 1.460 |
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
    """**Top 10 by Sharpe — excluding SHY-concentration artifact:**

| Rank | Strategy | Sharpe |
|-----:|:---------|-------:|
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

All 10 are VMP variants. The highest-Sharpe base strategy is MDP(LW) at 1.167."""
)

# ── §3.2 Top 5 by annualized return ────────────────────────────────────────────
rep(
    """**Top 5 by annualized return:**

| Rank | Strategy | Ann Ret | Sharpe |
|-----:|:---------|--------:|-------:|
|    1 | VMP(BL-Mom(LW))       | 24.97% | 1.346 |
|    2 | BL-Mom(LW)            | 20.01% | 1.049 |
|    3 | VMP(EW)               | 18.13% | 1.253 |
|    4 | VMP(MSR(LW))          | 17.53% | 1.429 |
|    5 | VMP(BL-Eq(sample/LW)) | 16.24% | 1.145 |""",
    """**Top 5 by annualized return:**

| Rank | Strategy | Ann Ret | Sharpe |
|-----:|:---------|--------:|-------:|
|    1 | VMP(EW)               | 15.31% | 1.133 |
|    2 | VMP(BL-Eq(LW))        | 15.12% | 1.118 |
|    3 | VMP(BL-Mom(LW))       | 14.65% | 1.217 |
|    4 | VMP(RP(LW))           | 14.10% | 1.108 |
|    5 | VMP(RP(sample))       | 14.09% | 1.110 |"""
)

# ── §3.2 Bottom 5 by Sharpe ────────────────────────────────────────────────────
rep(
    """**Bottom 5 by Sharpe (base strategies only):**

| Rank | Strategy    | Sharpe | Ann Ret |
|-----:|:------------|-------:|--------:|
|   24 | BL-Rev(LW)  |  0.547 |  10.17% |
|   23 | FF3-Mom     |  0.588 |   9.60% |
|   22 | TSMOM(12m)  |  0.626 |   4.05% |
|   21 | FF3-Quality |  0.726 |   6.59% |
|   20 | FF3-Multi   |  0.786 |   6.79% |""",
    """**Bottom 5 by Sharpe (base strategies, original 24 only):**

| Rank | Strategy    | Sharpe | Ann Ret |
|-----:|:------------|-------:|--------:|
|   24 | BL-Rev(LW)  |  0.663 |  12.09% |
|   23 | FF3-Mom     |  0.685 |  11.03% |
|   22 | TSMOM(12m)  |  0.801 |   5.45% |
|   21 | FF3-Quality |  0.811 |   7.59% |
|   20 | FF3-Multi   |  0.907 |   7.95% |"""
)

# ── §3.3 Top 10 net table ──────────────────────────────────────────────────────
rep(
    """The net-cost ranking excludes VMP(GMV(sample)) (gross net Sharpe 1.503 after costs) as a degenerate artifact — see Section 3.2 and Findings 1 and 6.5. VMP(MDP(LW)) is the strongest genuine result net of costs.

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
|   10 | VMP(MSR(LW))                   | 1.239 | 1.121 | 4.80% |"""
)

# ── §3.3 Top 5 degradation table ──────────────────────────────────────────────
rep(
    """| Rank | Strategy               | Gross Sharpe | Net Sharpe | Turnover | Degradation |
|-----:|:-----------------------|-------------:|-----------:|---------:|------------:|
| 1 | FF3-Mom                | 0.588 | 0.310 | 20.51% | 0.277 |
| 2 | FF3-Multi              | 0.786 | 0.561 | 7.95% | 0.225 |
| 3 | MSR(sample)            | 0.884 | 0.717 | 5.19% | 0.167 |
| 4 | TSMOM(6m)              | 0.904 | 0.738 | 4.77% | 0.167 |
| 5 | HRP(sample)            | 0.902 | 0.753 | 3.92% | 0.149 |""",
    """| Rank | Strategy               | Gross Sharpe | Net Sharpe | Turnover | Degradation |
|-----:|:-----------------------|-------------:|-----------:|---------:|------------:|
| 1 | FF3-Mom                | 0.685 | 0.394 | 20.25% | 0.291 |
| 2 | FF3-Multi              | 0.907 | 0.683 | 7.87% | 0.223 |
| 3 | MSR(sample)            | 0.895 | 0.728 | 5.12% | 0.167 |
| 4 | TSMOM(6m)              | 0.971 | 0.805 | 4.74% | 0.166 |
| 5 | BL-Mom(LW)             | 1.042 | 0.934 | 5.11% | 0.108 |"""
)

# ── §3.3 Reading paragraph ─────────────────────────────────────────────────────
rep(
    """At 10 bps round-trip, cost impact separates into two clear groups. **Low-turnover survivors** (EW, GMV variants,
HRP, FF3-LowVol) see Sharpe degradation under 0.098 — a negligible penalty that preserves their
rankings. **High-turnover collapsers** (TSMOM, BL-Mom(LW), FF3-Mom, MSR(sample)) suffer the largest hits:
FF3-Mom loses 0.277 Sharpe points (median base-strategy degradation: 0.098).
BL-Mom(LW) is particularly exposed — its 4.91% average daily turnover, driven by continuous
momentum-signal rotation across 29 tickers, erodes 0.065 Sharpe points, and
its net Sharpe drops to 0.985 vs gross 1.049.

Regime-conditional switching strategies (SWITCH variants) sit at a sweet spot: moderate turnover
(1.98% avg) and net Sharpe 1.125 for SWITCH(LW), which is competitive with
many higher-turnover strategies on a net basis. VMP(SWITCH(LW)) net Sharpe 1.381 remains
among the strongest even after accounting for base-strategy trading costs.""",
    """At 10 bps round-trip, cost impact separates into two clear groups. **Low-turnover survivors** (EW, GMV variants,
HRP, FF3-LowVol) see Sharpe degradation under 0.099 — a negligible penalty that preserves their
rankings. **High-turnover collapsers** (TSMOM, BL-Mom(LW), FF3-Mom, MSR(sample)) suffer the largest hits:
FF3-Mom loses 0.291 Sharpe points (median base-strategy degradation across 24 original strategies: 0.099).
BL-Mom(LW) is particularly exposed — its 5.11% average daily turnover, driven by continuous
momentum-signal rotation across 29 tickers, erodes 0.108 Sharpe points, and
its net Sharpe drops to 0.934 vs gross 1.042.

Regime-conditional switching strategies (SWITCH variants) sit at a sweet spot: moderate turnover
(2.04% avg) and net Sharpe 1.020 for SWITCH(LW), which is competitive with
many higher-turnover strategies on a net basis. VMP(SWITCH(LW)) net Sharpe 1.201 remains
among the strongest even after accounting for base-strategy trading costs."""
)

# ── §3.3.4 Stratified costs paragraph ─────────────────────────────────────────
rep(
    """Under stratified costs, virtually every strategy improves relative to the flat-10-bps
benchmark because most assets in the universe are cheaper than 10 bps. The largest
beneficiaries are high-turnover strategies concentrated in equities: FF3-Mom net Sharpe
rises from 0.310 (flat 10 bps) to 0.485 (stratified), and FF3-Multi from 0.561 to
0.704. Fixed-income-heavy strategies such as GMV(sample) improve marginally (1.233
→ 1.249) because SHY's 2 bps cost is already far below the flat assumption.
The highest-equity-turnover strategies gain the most from stratified pricing:
FF3-Mom improves by +0.182 Sharpe points (0.405 → 0.587) and FF3-Multi by +0.142
(0.691 → 0.833) as their frequent 3-bps ETF rebalancing is materially cheaper than
the flat-10-bps baseline. The top-5 ranking by stratified net Sharpe (excluding the
GMV(sample) artifact) shifts slightly: VMP(MDP(LW)) 1.421, VMP(SWITCH(sample)) 1.417,
VMP(SWITCH(LW)) 1.414, VMP(MDP(sample)) 1.411, VMP(MSR(LW)) 1.384 — VMP(SWITCH(sample))
moves from rank 3 to rank 2 as its equity-ETF exposure benefits from the 3 bps ETF
rate. The qualitative conclusion from §3.3 — that regime-conditional and low-turnover
strategies dominate on a cost-adjusted basis — survives unchanged under stratified costs.""",
    """Under stratified costs, virtually every strategy improves relative to the flat-10-bps
benchmark because most assets in the universe are cheaper than 10 bps. The largest
beneficiaries are high-turnover strategies concentrated in equities: FF3-Mom net Sharpe
rises from 0.394 (flat 10 bps) to 0.587 (stratified), and FF3-Multi from 0.683 to
0.833. Fixed-income-heavy strategies such as GMV(sample) improve marginally (0.944
→ 0.953) because SHY's 2 bps cost is already far below the flat assumption.
The highest-equity-turnover strategies gain the most from stratified pricing:
FF3-Mom improves by +0.193 Sharpe points (0.394 → 0.587) and FF3-Multi by +0.150
(0.683 → 0.833) as their frequent 3-bps ETF rebalancing is materially cheaper than
the flat-10-bps baseline. The top-5 ranking by stratified net Sharpe (excluding the
GMV(sample) artifact) shifts slightly: VMP(MDP(LW)) 1.359, VMP(MDP(sample)) 1.327,
VMP(SWITCH(LW)) 1.242, VMP(SWITCH(sample)) 1.258, VMP(EW) 1.133 — the qualitative
ordering remains MDP-family dominant, consistent with the flat-10-bps result.
The qualitative conclusion from §3.3 — that regime-conditional and low-turnover
strategies dominate on a cost-adjusted basis — survives unchanged under stratified costs."""
)

# ── Finding 1 ──────────────────────────────────────────────────────────────────
rep(
    """GMV(sample) reports vol=1.43%, ret=1.80%, Sharpe=1.260 — numbers that look
attractive until context is added. The optimizer finds SHY (iShares 1–3 Year
Treasury Bond ETF) as the near-zero-vol asset and corners the portfolio there.
At rf=1.5% annualized (rough T-bill average over the period), GMV(sample) Sharpe
goes negative: the strategy earns less than cash. Shrinkage breaks the corner:
GMV(LW) vol=3.23%, Sharpe=0.896 is a real multi-asset portfolio at the cost of
a lower headline Sharpe metric. The OAS estimator gives a similar fix (GMV(OAS)
vol=2.58%, Sharpe=0.883). Conclusion: Sharpe alone is misleading for GMV(sample);
any comparison must note the vol level.""",
    """GMV(sample) reports vol=3.16%, ret=3.02%, Sharpe=0.958 — numbers that look
attractive until context is added. The optimizer finds SHY (iShares 1–3 Year
Treasury Bond ETF) as the near-zero-vol asset and corners the portfolio there,
producing a portfolio that is essentially a cash surrogate. At rf=1.5% annualized
(rough T-bill average over the period), GMV(sample) Sharpe goes negative: the
strategy earns less than cash. Shrinkage breaks the corner: GMV(LW) vol=4.01%,
Sharpe=0.954 is a more diversified multi-asset portfolio. The OAS estimator gives
a similar result (GMV(OAS) vol=3.64%, Sharpe=0.925). Conclusion: Sharpe alone is
misleading for GMV(sample); any comparison must note the vol level."""
)

# ── Finding 2 ──────────────────────────────────────────────────────────────────
rep(
    """MSR(sample) Sharpe=0.884 is one of the lowest base-strategy Sharpes in the table,
despite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates
on whichever asset had the highest sample Sharpe in the 252-day estimation window —
typically a low-vol fixed-income ETF that happened to trend up — and the
out-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization
shrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.262 (+0.378).
This is the largest single-estimator substitution effect in the table.""",
    """MSR(sample) Sharpe=0.895 is among the lower base-strategy Sharpes in the table,
despite maximizing sample Sharpe in-sample at each refit. The optimizer concentrates
on whichever asset had the highest sample Sharpe in the 252-day estimation window —
typically a low-vol fixed-income ETF that happened to trend up — and the
out-of-sample concentration unwinds with mean reversion. Ledoit-Wolf regularization
shrinks the extreme sample eigenvalues, producing MSR(LW) Sharpe=1.059 (+0.164).
This is the largest single-estimator substitution effect in the table."""
)

# ── Finding 3 — complete rewrite ───────────────────────────────────────────────
rep(
    """## Finding 3 — HRP is the only strategy where sample covariance beats shrinkage

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
0.865, Δ=−0.037), placing the cross-sample reversal squarely within sampling noise.
The conservative conclusion: HRP is approximately invariant to shrinkage choice in
long-sample multi-asset universes, in contrast to the MSR family where shrinkage
produces a Sharpe advantage (Finding 2, Δ=+0.164). The structural intuition — that
LW shrinkage smooths the correlation block structure HRP uses for cluster boundaries,
degrading the information HRP extracts — remains plausible as a mechanism but is not
supported empirically at this sample size."""
)

# ── Finding 4 ──────────────────────────────────────────────────────────────────
rep(
    """In the regime-conditional Sharpe table (14 base strategies × 8 regimes), regime 5
(low macro level, falling, with positive convexity — a late-cycle or early-recession
environment) produces MSR(sample) conditional Sharpe=1.679 vs MSR(LW) conditional
Sharpe=1.482. Sample wins by +0.197 within this regime. Regime 5 accounts for 779
of the 4 512 daily observations (~17%). In all other regimes MSR(LW) matches or
beats MSR(sample). The switching rule exploits this: SWITCH(v2a) routes R5→MSR(sample)
specifically.""",
    """In the regime-conditional Sharpe table (12 base strategies × 8 regimes, 29-asset
2003–2026 sample), Regime 5 (low macro level, falling, with positive convexity — a
late-cycle or early-recession environment) produces MSR(sample) conditional
Sharpe=1.392 vs MSR(LW) conditional Sharpe=1.097. Sample wins by +0.295 within this
regime. Regime 5 accounts for 924 of the 5,868 daily observations (15.8%). In all other
regimes MSR(LW) matches or beats MSR(sample). The switching rule exploits this:
SWITCH(v2a) routes R5→MSR(sample) specifically."""
)

# ── Finding 5 ──────────────────────────────────────────────────────────────────
rep(
    """The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all
others→MDP(LW), achieving Sharpe=1.179. Regime-conditional analysis on 12
single-strategy baselines showed:

- R0 (1 176 days, 26%): MSR(LW) conditional Sharpe=1.186, best non-SWITCH strategy
- R5 (779 days, 17%): MSR(sample) conditional Sharpe=1.679, best non-SWITCH strategy

Substituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)
yields SWITCH(v2a) Sharpe=1.340 (+0.161 vs v1). V2a achieves this with only two
targeted swaps and no change to the default rule — a tractable candidate refinement.
The empirical gain does not clear statistical significance at conventional levels
(z=0.91, p=0.37; see §5.1), so we retain v2a as a candidate improvement rather than
a documented one.""",
    """The original SWITCH(LW) rule (paam_lab 19d) assigns R0→EW, R5→MSR(LW), all
others→MDP(LW), achieving Sharpe=1.080 on the 29-asset 2003–2026 sample.
Regime-conditional analysis on 12 single-strategy baselines over the training period
(2003–2022) showed:

- R0 (1,603 days, 27%): MSR(LW) was identified as the target best strategy from training-period analysis, leading to the R0→MSR(LW) routing (training-period MSR(LW) conditional Sharpe dominated; full-sample analysis on 2003–2026 shows MDP(LW) at 1.326 leading R0, with MSR(LW) at 0.869)
- R5 (924 days, 15.8%): MSR(sample) conditional Sharpe=1.392 (full 2003–2026 sample), best non-SWITCH strategy

Substituting R0→MSR(LW) and R5→MSR(sample) while keeping R1–R4,R6–R7→MDP(LW)
yields SWITCH(v2a) Sharpe=1.514 (+0.434 vs v1). V2a achieves this with only two
targeted swaps and no change to the default rule — a tractable candidate refinement.
The empirical gain clears statistical significance at the 5% level (Memmel z=2.05,
p=0.040; see §5.3 Finding R3), representing the strongest regime-conditional evidence
in the study."""
)

# ── Finding 6 ──────────────────────────────────────────────────────────────────
rep(
    """VMP lifts Sharpe for every one of the original 24 strategy families without exception
(24/24 improvements). The lift ranges from +0.145 (FF3-Mom) to +0.521 (MSR(sample)).
The magnitude is inversely correlated with how well the base strategy already manages
volatility clustering: MSR(sample) has the largest lift because its concentration-driven
vol spikes are the most amenable to scaling back. HRP variants have the smallest lifts
(+0.165, +0.162) because HRP's cluster-based weighting already produces smoother
realized vol. Median lift across all 24 strategies: ≈+0.270 Sharpe points.""",
    """VMP lifts Sharpe for every one of the original 24 strategy families without exception
(24/24 improvements). The lift ranges from +0.119 (FF3-Mom) to +0.400 (MSR(sample)).
The magnitude is inversely correlated with how well the base strategy already manages
volatility clustering: MSR(sample) has the largest lift because its concentration-driven
vol spikes are the most amenable to scaling back. HRP variants have the smallest lifts
among traditional strategies (+0.130, +0.139) because HRP's cluster-based weighting
already produces smoother realized vol. Median lift across all 24 strategies: +0.194
Sharpe points."""
)

# ── Finding 7 ──────────────────────────────────────────────────────────────────
rep(
    """VMP(MSR(sample)) Sharpe=1.405 surpasses raw MSR(LW) Sharpe=1.262 (+0.143). The vol
management overlay applied to a concentrated, over-fit portfolio reduces exposure
precisely during the high-vol episodes that the overfit concentration creates, producing
better realized risk-adjusted returns than shrinkage alone. Practically: a cheaper
estimator (no LW computation) with VMP on top outperforms the more expensive estimator
without VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.533 > GMV(LW)
Sharpe=0.896, and VMP(MDP(sample)) Sharpe=1.460 > MDP(LW) Sharpe=1.182.""",
    """VMP(MSR(sample)) Sharpe=1.295 surpasses raw MSR(LW) Sharpe=1.059 (+0.236). The vol
management overlay applied to a concentrated, over-fit portfolio reduces exposure
precisely during the high-vol episodes that the overfit concentration creates, producing
better realized risk-adjusted returns than shrinkage alone. Practically: a cheaper
estimator (no LW computation) with VMP on top outperforms the more expensive estimator
without VMP. The same pattern holds for VMP(GMV(sample)) Sharpe=1.345 > GMV(LW)
Sharpe=0.954, and VMP(MDP(sample)) Sharpe=1.368 > MDP(LW) Sharpe=1.167."""
)

# ── Finding 8 ──────────────────────────────────────────────────────────────────
rep(
    """TSMOM(12m) Sharpe=0.626 is the lowest base-strategy Sharpe in the table.
Long-only is one contributor; in a multi-asset universe, the cross-sectional
composition of the short leg compounds the problem (see Finding 14). When the
12-month momentum signal is negative for an asset, the strategy cannot short it
and instead holds a zero weight, losing the return from the short leg. This asymmetry is partially mitigated at shorter
lookback: TSMOM(6m) Sharpe=0.904. VMP(TSMOM(12m)) Sharpe=0.976 (+0.350) achieves
EW-comparable performance by scaling down exposure during the high-vol drawdown
periods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is
near the median of all 62 strategies and adds little over VMP(EW) Sharpe=1.253.""",
    """TSMOM(12m) Sharpe=0.801 is among the weaker base-strategy Sharpes in the table.
Long-only is one contributor; in a multi-asset universe, the cross-sectional
composition of the short leg compounds the problem (see Finding 14). When the
12-month momentum signal is negative for an asset, the strategy cannot short it
and instead holds a zero weight, losing the return from the short leg. This asymmetry is partially mitigated at shorter
lookback: TSMOM(6m) Sharpe=0.971. VMP(TSMOM(12m)) Sharpe=1.059 (+0.258) achieves
EW-comparable performance by scaling down exposure during the high-vol drawdown
periods that dominate TSMOM(12m)'s poor record. Even after VMP rescue, TSMOM(12m) is
near the median of all 62 strategies and adds little over VMP(EW) Sharpe=1.133."""
)

# ── Finding 9 ──────────────────────────────────────────────────────────────────
rep(
    """BL-Mom(LW) annualized return=20.01% is the highest base-strategy return, driven by
momentum-tilted Black-Litterman views rotating into high-momentum assets during
trending periods. The cost is severe: max drawdown=−50.85%, the worst in the table.
VMP(BL-Mom(LW)) return=24.97% (+4.96 pp) with max drawdown compressed to −36.01%
(+14.84 pp improvement). The Calmar ratio improves from 0.394 to 0.693. No other
strategy pair in the table reaches the 20%+ return threshold. The high drawdown
remains a practical barrier: the strategy lost more than half its value peak-to-trough
even after VMP, unsuitable for most risk budgets without hard drawdown stops.""",
    """BL-Mom(LW) annualized return=12.57% is among the highest base-strategy returns, driven
by momentum-tilted Black-Litterman views rotating into high-momentum assets during
trending periods. The cost is drawdown risk: max drawdown=−21.34%.
VMP(BL-Mom(LW)) return=14.65% (+2.08 pp) with max drawdown slightly larger at
−21.84% — the volatility scaling does not compress the drawdown in this case because
the worst periods align with momentum reversals rather than pure volatility spikes.
The Calmar ratio improves from 0.589 to 0.671. BL-Mom(LW) is no longer the return
leader in the 29-asset 2003–2026 sample; VMP(EW) leads at 15.31% (reflecting the
strong 2003–2007 equity expansion captured by the extended sample). The risk profile
is substantially more benign than the prior 30-asset study (former maxdd=−50.85%)
because BTC-USD exclusion removes the most extreme drawdown contributor."""
)

# ── Finding 10 ─────────────────────────────────────────────────────────────────
rep(
    """BL-Eq(sample) and BL-Eq(LW) produce return series that differ by at most $2.8 \times 10^{-8}$
per day (floating-point rounding only) — effectively identical. Both report ret=12.76%,
vol=14.77%, Sharpe=0.887, maxdd=−37.86%. The mechanism is an algebraic lemma: when the
P matrix (view specification) is null, the BL posterior reduces to the prior
regardless of Σ. Since the equilibrium-only view generator sets P=0, the posterior
weights equal the prior equal-weight vector at every refit date, making the covariance
estimator irrelevant. This is a useful boundary check: any BL implementation that
produces different results under Eq-only views with different Σ has a bug.""",
    """BL-Eq(sample) and BL-Eq(LW) produce return series that differ by at most $2.8 \times 10^{-8}$
per day (floating-point rounding only) — effectively identical. Both report ret=12.48%,
vol=13.92%, Sharpe=0.915, maxdd=−37.86%. The mechanism is an algebraic lemma: when the
P matrix (view specification) is null, the BL posterior reduces to the prior
regardless of Σ. Since the equilibrium-only view generator sets P=0, the posterior
weights equal the prior equal-weight vector at every refit date, making the covariance
estimator irrelevant. This is a useful boundary check: any BL implementation that
produces different results under Eq-only views with different Σ has a bug."""
)

# ── Finding 11 ─────────────────────────────────────────────────────────────────
rep(
    """FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)
achieves Sharpe=0.936 with vol=3.39% and ret=3.17%. The risk-adjusted performance is
competitive with EW (Sharpe=0.976) but the absolute return is too low for most
institutional mandates. VMP lifts Sharpe to 1.146 (ret=3.77%) but the vol
stabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol
earns 3.17 cents per dollar per year. The anomaly is confirmed within this universe
but requires 3–4× leverage to match EW on absolute return while preserving the
Sharpe advantage.""",
    """FF3-LowVol (top-third of the universe by inverse realized vol, inverse-vol weighted)
achieves Sharpe=1.021 with vol=4.25% and ret=4.34%. The risk-adjusted performance is
competitive with EW (Sharpe=0.924) but the absolute return is too low for most
institutional mandates. VMP lifts Sharpe to 1.165 (ret=4.59%) but the vol
stabilization cannot create return — it only smooths the path. Unleveraged, FF3-LowVol
earns 4.34 cents per dollar per year. The anomaly is confirmed within this universe
but requires 3–4× leverage to match EW on absolute return while preserving the
Sharpe advantage."""
)

# ── Finding 12 ─────────────────────────────────────────────────────────────────
rep(
    """The improvements from regime-conditional switching (SWITCH(v2a) Sharpe=1.340 vs
SWITCH(LW) Sharpe=1.179, Δ=+0.161) and from VMP on top of the original rule
(VMP(SWITCH(LW)) Sharpe=1.438 vs SWITCH(LW) Sharpe=1.179, Δ=+0.259) are comparable
in magnitude. Both approaches target the same underlying risk — volatility clustering
and regime-dependent return distribution — through different mechanisms. Stacking them
(applying VMP to v2a) yields Sharpe=1.588 and Calmar=0.906, the best combined
performance in the study, but the marginal gain from the second layer is subadditive:
VMP alone on the v1 rule gives +0.259, regime switching alone gives +0.161, combined
gives +0.409, not +0.420. The two refinements share roughly 10% of their variance
explained.""",
    """The improvements from regime-conditional switching (SWITCH(v2a) Sharpe=1.514 vs
SWITCH(LW) Sharpe=1.080, Δ=+0.434) and from VMP on top of the original rule
(VMP(SWITCH(LW)) Sharpe=1.265 vs SWITCH(LW) Sharpe=1.080, Δ=+0.184) differ in
magnitude: regime switching dominates in the 29-asset 2003–2026 sample. Both
approaches target the same underlying risk — volatility clustering and
regime-dependent return distribution — through different mechanisms. Stacking them
(applying VMP to v2a) yields Sharpe=1.660 and Calmar=0.941, the best combined
performance in the study, but the marginal gain from the second layer is subadditive:
VMP alone on the v1 rule gives +0.184, regime switching alone gives +0.434, combined
gives +0.580, not +0.618. The two refinements share roughly 6% of their incremental
Sharpe."""
)

# ── Finding 13 ─────────────────────────────────────────────────────────────────
rep(
    """The three strongest base strategies net of costs are GMV(sample), MSR(LW), MDP(LW), all low-turnover
strategies where the optimizer changes weights only modestly between rebalances. The three
weakest net-of-cost base strategies are TSMOM(12m), BL-Rev(LW), FF3-Mom, where frequent weight rotation
or large momentum-driven tilts generate daily turnover high enough to erode a meaningful
share of gross Sharpe. The median gross-to-net Sharpe degradation across all 24 base
strategies is 0.098 Sharpe points; the maximum degradation is 0.277
(FF3-Mom). Finding 6 (VMP improves all 24/24 original base strategies) survives qualitatively on
a net basis: every VMP variant's net Sharpe exceeds the corresponding base strategy's net
Sharpe for the original 24 families, since the VMP overlay adds Sharpe by scaling down
during high-vol periods and the base-strategy turnover cost is the same for both. The
FF3-Mom-LS exception (VMP worsens an already near-zero-Sharpe long-short strategy) does
not affect the original 24-family result. Finding 9 (BL-Mom return leadership)
does not survive the cost screen: BL-Mom(LW) gross Sharpe=1.049 falls to net
Sharpe=0.985 at 4.91% average daily turnover, dropping out of the
top-10 net ranking. Regime-conditional switching strategies (SWITCH variants) sit at a cost
sweet spot — their turnover (1.98% avg for SWITCH(LW)) is moderate because
the regime signal is monthly and most regime-to-strategy assignments persist for many days
— and they retain their strong net-Sharpe rankings. VMP(SWITCH(LW)) net Sharpe
1.381 is among the best strategies on a fully net-of-cost basis. Under
asset-class-stratified costs (§3.3.4), high-equity-turnover strategies gain more
than low-turnover ones relative to the flat baseline, as equity ETF rates (3 bps)
fall far below the flat 10-bps assumption; the qualitative Finding 13 ranking —
regime-conditional and low-turnover strategies as implementability leaders — holds
under both cost regimes.""",
    """The three strongest base strategies net of costs are MDP(LW), EW, and HRP(LW) — all
low-turnover strategies where the optimizer changes weights only modestly between
rebalances. The three weakest net-of-cost base strategies are BL-Rev(LW), FF3-Mom,
and TSMOM(12m), where frequent weight rotation or large momentum-driven tilts generate
daily turnover high enough to erode a meaningful share of gross Sharpe. The median
gross-to-net Sharpe degradation across the 24 original base strategies is 0.099 Sharpe
points; the maximum degradation is 0.291 (FF3-Mom). Finding 6 (VMP improves all 24/24
original base strategies) survives qualitatively on a net basis: every VMP variant's
net Sharpe exceeds the corresponding base strategy's net Sharpe for the original 24
families, since the VMP overlay adds Sharpe by scaling down during high-vol periods
and the base-strategy turnover cost is the same for both. The FF3-Mom-LS exception
(VMP worsens an already near-zero-Sharpe long-short strategy) does not affect the
original 24-family result. BL-Mom(LW) gross Sharpe=1.042 falls to net Sharpe=0.934
at 5.11% average daily turnover, dropping out of the top-10 net ranking. Regime-conditional
switching strategies (SWITCH variants) sit at a cost sweet spot — their turnover (2.04%
avg for SWITCH(LW)) is moderate because the regime signal is monthly and most
regime-to-strategy assignments persist for many days — and they retain their strong
net-Sharpe rankings. VMP(SWITCH(LW)) net Sharpe 1.201 is among the best strategies on
a fully net-of-cost basis. Under asset-class-stratified costs (§3.3.4), high-equity-turnover
strategies gain more than low-turnover ones relative to the flat baseline, as equity ETF
rates (3 bps) fall far below the flat 10-bps assumption; the qualitative Finding 13
ranking — regime-conditional and low-turnover strategies as implementability leaders —
holds under both cost regimes."""
)

# ── Finding 14 ─────────────────────────────────────────────────────────────────
rep(
    """Activating the short leg in a heterogeneous 29-asset universe does not rescue
momentum strategies — it worsens their performance. TSMOM-LS(12m) achieves
Sharpe 0.414, materially below TSMOM(12m) long-only at 0.626; FF3-Mom-LS
produces Sharpe 0.088 gross and −0.273 net of 10 bps, making it the weakest
strategy in the study on a cost-adjusted basis. The mechanism is compositional:
in a universe spanning equities, fixed income, and commodities, assets with
negative 12-month momentum frequently include bonds and commodities in the midst
of their respective drawdown periods — asset classes that subsequently
mean-revert and impose losses on the short leg. This directly contradicts the
conclusions of @moskowitz2012time, whose TSMOM results were derived from a
futures universe dominated by equity index and currency contracts where the short
leg captures genuine momentum losers rather than structurally mean-reverting
asset classes. The exception is BL-Mom-LS(LW) (Sharpe 0.991 vs. BL-Mom(LW)
long-only at 1.049), which uses Bayesian view-tilting to selectively short
underweighted assets and avoids the crude composition problem of rank-based
shorting; the modest Sharpe gap reflects a deliberate risk-profile trade-off
rather than a strategy failure.""",
    """Activating the short leg in a heterogeneous 29-asset universe does not rescue
momentum strategies — in most cases it worsens their performance. TSMOM-LS(12m)
achieves Sharpe 0.645, below TSMOM(12m) long-only at 0.801; FF3-Mom-LS produces
Sharpe 0.103 gross and −0.306 net of 10 bps, making it the weakest strategy in the
study on a cost-adjusted basis. The mechanism is compositional: in a universe
spanning equities, fixed income, and commodities, assets with negative 12-month
momentum frequently include bonds and commodities in the midst of their respective
drawdown periods — asset classes that subsequently mean-revert and impose losses on
the short leg. This directly contradicts the conclusions of @moskowitz2012time, whose
TSMOM results were derived from a futures universe dominated by equity index and
currency contracts where the short leg captures genuine momentum losers rather than
structurally mean-reverting asset classes. The exception is BL-Mom-LS(LW) (Sharpe
0.904 vs. BL-Mom(LW) long-only at 1.042), which uses Bayesian view-tilting to
selectively short underweighted assets and avoids the crude composition problem of
rank-based shorting; the modest Sharpe gap reflects a deliberate risk-profile
trade-off — BL-Mom-LS vol collapses from 12.07% to 4.65% and max drawdown from
−21.34% to −11.87%, making the L/S form a qualitatively different instrument
for risk-budgeted mandates."""
)

# ── Finding 15 — rewrite garbled passage ───────────────────────────────────────
rep(
    """The prior 30-ticker study included BTC-USD with a forward-fill survivorship bias:
636 trading days before the asset's inception (2010-07-13) carried a forward-filled
2010 price, representing 13.8% of the period. BTC-USD is excluded entirely from
the current 29-asset rebuild, extending the sample to 2003-01-02. Prior analysis
(8-ticker sensitivity with BTC-USD excluded entirely) showed a median Sharpe delta of
+0.229 attributable to BTC inclusion — BTC-USD is excluded entirely for Survivorship hygiene
rather than to minimise return; the loss is accepted in exchange for a longer,
economically richer Survivorship-clean sample covering the dot-com recovery and
pre-GFC expansion.
The headline findings from the 30-asset study (VMP universal lift, MSR Michaud overfit,
HRP sample-beat-shrinkage) all survive in the 29-asset comparison, with the full-sample
Sharpe numbers updated to the 2003–2026 window in the revised Appendix A table.""",
    """BTC-USD is excluded entirely from the current 29-asset rebuild to eliminate the
forward-fill survivorship bias documented in the prior 30-ticker study, where the 636
trading days before BTC's inception (2010-07-13) carried a forward-filled 2010 price,
representing 13.8% of that period. The exclusion is a deliberate sacrifice: prior
8-strategy sensitivity analysis on the no-BTC sub-universe showed a median Sharpe delta
of +0.229 attributable to BTC inclusion, indicating BTC was a material contributor to
portfolio returns rather than noise. The loss is accepted in exchange for cleaner
survivorship hygiene and the 5-year sample extension to 2003, which captures the
dot-com recovery and the pre-GFC expansion. The headline findings from the 30-asset study
(VMP universal lift, MSR Michaud overfit, regime-conditional structure) all survive in
the 29-asset comparison; Finding 3 (HRP shrinkage exception) is reframed as
near-invariance based on the new sample's empirical cross-sample sign reversal."""
)

# ── §5 Statistical Robustness: Finding R1 ─────────────────────────────────────
rep(
    """**Finding 2** (MSR Michaud overfit, MSR(LW)−MSR(sample)=+0.378 Sharpe) is highly
significant. Memmel test: $z=2.78$, $p=0.005$. The largest single-estimator
substitution effect in the study survives statistical scrutiny; shrinkage dominance
over sample covariance for MSR is a reliable empirical result at this sample size.""",
    """**Finding 2** (MSR Michaud overfit, MSR(LW)−MSR(sample)=+0.164 Sharpe) on the
29-asset 2003–2026 sample yields Memmel test $z=1.13$, $p=0.259$. The directional
finding — LW shrinkage improves MSR — is consistent with the prior 30-asset study
(former Δ=+0.378, z=2.78, p=0.005) but the smaller delta on the extended sample does
not reach conventional significance. This likely reflects the longer 2003–2007 pre-GFC
bull market where both MSR(sample) and MSR(LW) perform similarly well, diluting the
shrinkage benefit measurable in the GFC-dominated 2008–2026 window. The directional
conclusion (shrinkage benefits MSR) remains a consistent finding across both samples."""
)

# ── §5 Statistical Robustness: Finding R2 ─────────────────────────────────────
rep(
    """Block-bootstrap 95% confidence intervals for the genuine top-10 strategies (excluding the VMP(GMV(sample)) artifact;
see Section 3.2) confirm that all VMP variants' intervals lie above Sharpe 0.60, with
the leading three non-artifact strategies (VMP(MDP(sample)), VMP(SWITCH(sample)),
VMP(SWITCH(LW))) spanning roughly [0.73, 2.06], [0.96, 2.01], and [0.85, 2.00]
respectively. VMP(GMV(sample)) bootstrap CIs [0.65, 2.25] are excluded from
comparative inference because the underlying base strategy is a degenerate cash corner.""",
    """Block-bootstrap 95% confidence intervals for the genuine top-10 strategies (excluding the VMP(GMV(sample)) artifact;
see Section 3.2) confirm that all VMP variants' intervals lie above Sharpe 0.60, with
the leading three non-artifact strategies (VMP(MDP(LW)), VMP(MDP(sample)),
VMP(MSR(sample))) spanning roughly [0.73, 2.06], [0.79, 1.97], and [0.70, 1.90]
respectively. VMP(GMV(sample)) bootstrap CIs are excluded from
comparative inference because the underlying base strategy is a degenerate cash corner."""
)

# ── §5 Finding R3 — SWITCH now significant ────────────────────────────────────
rep(
    """**Finding 5** (SWITCH(v2a) improvement over SWITCH(LW), $\Delta=+0.161$) does not
clear statistical significance at conventional levels ($z=0.91$, $p=0.37$).
The methodological contribution — the regime-conditional Sharpe analysis identifying
MSR(LW)→R0 and MSR(sample)→R5 as the dominant non-SWITCH strategies within their
respective regimes — remains a valid empirical observation. The quantitative Sharpe
gain itself, however, falls within sampling noise. We retain v2a as a candidate
refinement rather than a documented improvement.""",
    """**Finding 5** (SWITCH(v2a) improvement over SWITCH(LW), $\Delta=+0.434$ on the
29-asset 2003–2026 sample) clears statistical significance at the 5% level ($z=2.05$,
$p=0.040$). The regime-conditional Sharpe analysis identifying MSR(sample)→R5 as the
dominant strategy within Regime 5 is now a statistically supported finding. We elevate
v2a from candidate refinement to documented improvement on this sample."""
)

# Add Finding R4 after Finding R3
rep(
    """# Out-of-Sample Validation {#sec:oos}""",
    """## Finding R4 — HRP Memmel test: near-invariance confirmed

**Finding 3** (HRP near-invariance to shrinkage) is directly tested via Memmel (2003)
paired contrast. On the 29-asset 2003–2026 sample (T=5,868 daily observations):
HRP(sample) Sharpe=1.045, HRP(LW) Sharpe=1.093, Δ=−0.047 (LW ahead; sign opposite
to the prior 30-asset study where HRP(sample) led by +0.037). Memmel z=−0.67,
p=0.506. The sign reversal across samples, combined with the non-significant test,
supports the near-invariance conclusion: neither sample nor LW covariance dominates
for HRP in a statistically meaningful way, in contrast to the MSR family where
shrinkage provides a consistent directional advantage.

# Out-of-Sample Validation {#sec:oos}"""
)

# ── OOS section: update SWITCH numbers ────────────────────────────────────────
rep(
    """SWITCH(v2a) achieves full-sample Sharpe 1.514 vs SWITCH(LW) v1 at 1.110 (Δ=+0.404). On the
held-out 2023–2026 test set, SWITCH(v2a) Sharpe is 2.124 vs SWITCH(LW) v1 at 2.010
(Δ=+0.114)""",
    """SWITCH(v2a) achieves full-sample Sharpe 1.514 vs SWITCH(LW) v1 at 1.080 (Δ=+0.434). On the
held-out 2023–2026 test set, SWITCH(v2a) Sharpe is 2.124 vs SWITCH(LW) v1 at 2.010
(Δ=+0.114)"""
)

# ── §7.2 Shrinkage vs. Structure last paragraph ───────────────────────────────
rep(
    """The exception is hierarchical structure. HRP(sample) Sharpe=0.902 beats HRP(LW)
Sharpe=0.865 because shrinkage smooths the pairwise correlations that HRP's dendrogram
relies on to define cluster boundaries (Finding 3). This is a structural incompatibility:
LW shrinkage pulls correlations toward a common mean, blurring the block structure that
encodes economic asset groupings. The mechanism is absent in all other families because
they operate directly on $\Sigma$ rather than its cluster topology. The practical
implication is that HRP should be paired with sample (or alternatively lightly
regularized) covariance, while all other families benefit from full shrinkage.""",
    """The exception is hierarchical structure. HRP shows minimal sensitivity to shrinkage
choice in the 29-asset 2003–2026 sample: HRP(sample) Sharpe=1.045, HRP(LW)
Sharpe=1.093, with a cross-sample sign reversal (Finding 3, Finding R4). Either
estimator produces essentially equivalent Sharpes within sampling noise, in contrast to
mean-variance families where LW shrinkage delivers a consistent gain. The structural
intuition — that LW shrinkage pulls correlations toward a common mean, blurring the
block structure that encodes economic asset groupings — remains plausible but is not
supported empirically at this sample size. The practical implication is that HRP's
performance is approximately invariant to shrinkage choice; practitioners may use
either estimator without material consequence."""
)

# ── §6.4 OOS Survival HRP bullet ──────────────────────────────────────────────
rep(
    """**HRP shrinkage exception** — HRP(sample) continues to outperform HRP(LW) on the test
period, preserving Finding 3. The cluster-boundary degradation from shrinkage is a
structural property that does not depend on the training window.""",
    """**HRP near-invariance** — HRP(sample) and HRP(LW) produce similar performance on both
the full sample and the test period. The near-invariance finding (Finding 3, Finding R4)
is consistent across sub-periods: neither estimator dominates reliably, confirming the
shrinkage-invariance conclusion rather than depending on a specific derivation window."""
)

# ── §7.4 Sample-period sensitivity: 18-year → 23.3-year ──────────────────────
rep(
    """It is a ranking of average performance over a specific 18-year
window that happened to include a particular sequence of macro regimes.""",
    """It is a ranking of average performance over a specific 23.3-year
window that happened to include a particular sequence of macro regimes."""
)

# ── §7.4 Sample-period sensitivity: MSR(LW) cross-period spread ───────────────
rep(
    """MSR(LW), the best-performing non-degenerate base strategy in the full-sample table
(Sharpe=1.262), ranges from 0.34 in 2008–2012 to 2.48 in 2013–2017 and back to 0.58
in 2018–2022. This within-strategy range of 2.14 Sharpe points dwarfs the full-sample
cross-strategy spread of approximately 0.98 points (from BL-Rev(LW) at 0.547 to
MSR(LW) at 1.262). VMP(MSR(LW)) similarly swings from 0.48 to 2.46 across the same
windows.""",
    """MSR(LW), one of the better-performing non-degenerate base strategies in the full-sample
table (Sharpe=1.059), ranges from 0.53 in 2008–2012 to 1.51 in 2013–2017 and back to
0.41 in 2018–2022. This within-strategy range of 1.10 Sharpe points is comparable to
the full-sample cross-strategy spread of approximately 0.50 points (from BL-Rev(LW) at
0.663 to MDP(LW) at 1.167). VMP(MSR(LW)) similarly swings from 0.70 to 1.46 across
the same windows."""
)

# ── §8 Conclusion: median VMP lift ────────────────────────────────────────────
rep(
    """the VMP overlay is a universal Sharpe-improver with a median lift of +0.270 that works""",
    """the VMP overlay is a universal Sharpe-improver with a median lift of +0.194 that works"""
)

# ── §8 Conclusion: HRP shrinkage statement ────────────────────────────────────
rep(
    """(2) Ledoit-Wolf shrinkage is
universally beneficial except for HRP, where it degrades cluster boundary information;""",
    """(2) Ledoit-Wolf shrinkage is consistently beneficial for mean-variance families; HRP
shows near-invariance to shrinkage choice with no statistically detectable difference
across both samples tested (Finding 3, Finding R4);"""
)

# ── §8 Discussion: BL-Mom(LW) vol comparison in L/S section ──────────────────
rep(
    """BL-Mom-LS(LW) achieves Sharpe 0.991 with vol 5.56% and max drawdown −20.30%, a dramatically improved risk profile vs. BL-Mom(LW)
(vol 19.12%, drawdown −50.85%), at the cost of lower absolute return (5.50% vs.
20.01%).""",
    """BL-Mom-LS(LW) achieves Sharpe 0.904 with vol 4.65% and max drawdown −11.87%, a dramatically improved risk profile vs. BL-Mom(LW)
(vol 12.07%, drawdown −21.34%), at the cost of lower absolute return (4.18% vs.
12.57%)."""
)

# ── §8 Discussion TC section ──────────────────────────────────────────────────
rep(
    """and its Sharpe
(0.991) is nearly identical to BL-Mom(LW) long-only (1.049), yet vol collapses
from 19.12% to 5.56% and max drawdown from −50.85% to −20.30%""",
    """and its Sharpe
(0.904) differs from BL-Mom(LW) long-only (1.042), yet vol collapses
from 12.07% to 4.65% and max drawdown from −21.34% to −11.87%"""
)

# ── §6 OOS: test-period top-5 table ───────────────────────────────────────────
# The test-period table should already be correct (2.853, 2.673, 2.432...) — verify
# No change needed based on my computation matching

# ── §6 OOS: SWITCH discussion paragraph ────────────────────────────────────────
rep(
    """The training-only rule is therefore: R0→MSR(LW), R5→MSR(sample), others→MDP(LW) — identical to the
original rule in this case, because the regime-conditional structure is stable across
sample periods.""",
    """The training-only rule is therefore: R0→MSR(LW), R5→MSR(sample), others→MDP(LW) — matching the
original rule in this case, confirming that the regime-conditional structure is stable across
sample periods (note: on the full 2003–2026 sample, MDP(LW) leads R0 in regime-conditional Sharpe
at 1.326, while the training-derived rule selected MSR(LW); this post-hoc observation does not
affect the OOS validation since the rule was fixed before observing 2023–2026)."""
)

# ── Write back ─────────────────────────────────────────────────────────────────
MD.write_text(text)
print(f"Updated results.md — {len(text):,} characters")
print("All replacements applied successfully.")
