# Appendix F — Transaction Costs

## F.1 Cost Ladder

The main paper (§6.1) reports two cost regimes: a flat 10 bps round-trip benchmark (§3.3) and an asset-class-stratified schedule (§3.3.4). Turnover is measured as one-way daily turnover $\tau_t = \frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|$ aggregated to an annualised rate. Costs are applied per rebalance event; the column `Turnover` in the master table records the mean annualised one-way rate.

**Flat benchmark (10 bps).** A uniform 10 bps round-trip cost is applied per unit of one-way turnover: $\mathrm{CostDrag} = \tau \times 0.0010$. This is conservative — most assets in the 29-asset universe trade inside 10 bps on institutional order sizes.

**Stratified schedule.** Following Hilpisch (2026) Ch. 11, asset-class-specific round-trip costs are assigned as follows:

| Asset Class | Assets | Per-Trade Cost |
|---|---|---|
| Investment-grade fixed-income ETFs | SHY, IEF, TLT, AGG, HYG | 2 bps |
| Broad US equity + sector ETFs | SPY, IWM, XLK, XLF, XLE, XLV, XLP, XLU | 3 bps |
| US single-name equities | AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, WMT | 5 bps |
| International equity ETFs | EFA, EEM, FXI | 5 bps |
| Commodities and FX | GLD, SLV, DBC, USO, EURUSD | 5 bps |
| **Flat benchmark** | All 29 assets | **10 bps** |

The stratified schedule spans a 5-fold range (2–10 bps) versus the flat benchmark's single rate. The effective portfolio-level cost rate for any strategy depends on its weight allocation: fixed-income-heavy strategies (GMV variants) benefit most from the stratified schedule; equity-momentum strategies (FF3-Mom) benefit from 3-bps ETF pricing rather than the flat-10-bps assumption. The `NetStrat` column in `data/published/master_table_62strategies.csv` applies the stratified schedule to each strategy's turnover.

## F.2 Turnover Distribution

The 62 configurations span a roughly 3,000-fold range of annualised turnover: from EW (0.62% per year, driven solely by rebalance drift) to VMP(FF3-Mom) (2,025% per year, combining factor rotation with daily VMP exposure adjustment).

**Low-turnover survivors** (annualised turnover ≤ 10%): EW (0.62%), VMP(EW) (0.62%), GMV(OAS) (47.7%), GMV(LW) (54.5%), FF3-LowVol (38.7%), MDP(LW) (78.5%), and VMP(MDP(LW)) (78.5%). These strategies lose fewer than 0.015 Sharpe points to costs under either schedule, preserving their gross-Sharpe rankings almost intact.

**High-turnover collapsers**: FF3-Mom (2,025%), VMP(FF3-Mom) (2,025%), FF3-Mom-LS (1,456%), VMP(FF3-Mom-LS) (1,456%), BL-Rev(LW) (947%), and VMP(MSR(LW)) / VMP(MSR(sample)) (~480–510%). FF3-Mom's 20.25% annualised one-way turnover at flat 10 bps produces a 0.291 Sharpe degradation (gross 0.685 → net 0.394), making it uncompetitive on a cost-adjusted basis. Under the stratified schedule its net Sharpe recovers to 0.592 (degradation 0.093) because most of its rebalancing occurs in 3-bps ETFs.

Regime-switching strategies sit at a cost sweet spot: SWITCH(LW) turns over 2.04% per year (regime transitions are monthly, not daily), incurring only 0.060 Sharpe points of drag at flat 10 bps (gross 1.080 → net 1.020) and 0.020 points under the stratified schedule (net 1.060).

## F.3 Full 62-Configuration Net Sharpe Ranking (Stratified Costs)

Table F.1 ranks all 62 configurations by net Sharpe under the stratified asset-class cost schedule (`NetStrat`). Gross Sharpe and annualised one-way turnover are included for reference. Cost drag is the difference between gross and stratified-net Sharpe.

**Table F.1. Full 62-strategy net-of-stratified-cost ranking.**

| Rank | Configuration | Gross Sharpe | Turnover | Cost Drag | Net Sharpe |
|---|---|---|---|---|---|
| 1 | VMP(MDP(LW)) | 1.372 | 78.5% | 0.012 | 1.360 |
| 2 | VMP(GMV(sample))† | 1.345 | 18.2% | 0.006 | 1.339 |
| 3 | VMP(MDP(sample)) | 1.368 | 261.8% | 0.036 | 1.332 |
| 4 | VMP(SWITCH(sample)) | 1.293 | 327.9% | 0.032 | 1.260 |
| 5 | VMP(SWITCH(LW)) | 1.265 | 203.6% | 0.021 | 1.243 |
| 6 | VMP(MSR(sample)) | 1.295 | 494.2% | 0.064 | 1.230 |
| 7 | VMP(GMV(LW)) | 1.215 | 54.5% | 0.011 | 1.204 |
| 8 | VMP(MSR(LW)) | 1.239 | 479.8% | 0.041 | 1.198 |
| 9 | VMP(GMV(OAS)) | 1.207 | 47.7% | 0.010 | 1.196 |
| 10 | VMP(HRP(LW)) | 1.232 | 337.1% | 0.038 | 1.194 |
| 11 | VMP(BL-Mom(LW)) | 1.217 | 510.8% | 0.039 | 1.178 |
| 12 | VMP(FF3-LowVol) | 1.165 | 38.7% | 0.008 | 1.157 |
| 13 | MDP(LW) | 1.166 | 78.5% | 0.012 | 1.154 |
| 14 | VMP(MSR\_C(LW)) | 1.187 | 538.1% | 0.045 | 1.141 |
| 15 | VMP(EW) | 1.133 | 0.6% | 0.000 | 1.133 |
| 16 | VMP(HRP(sample)) | 1.176 | 396.8% | 0.045 | 1.131 |
| 17 | VMP(BL-Eq(LW)) | 1.118 | 0.9% | 0.000 | 1.118 |
| 18 | VMP(BL-Eq(sample)) | 1.118 | 0.9% | 0.000 | 1.118 |
| 19 | VMP(RP(sample)) | 1.110 | 8.6% | 0.001 | 1.109 |
| 20 | VMP(RP(LW)) | 1.108 | 10.9% | 0.001 | 1.107 |
| 21 | VMP(TSMOM(6m)) | 1.133 | 474.4% | 0.055 | 1.078 |
| 22 | MDP(sample) | 1.101 | 261.8% | 0.033 | 1.068 |
| 23 | SWITCH(LW) | 1.080 | 203.6% | 0.020 | 1.060 |
| 24 | HRP(LW) | 1.093 | 337.1% | 0.038 | 1.055 |
| 25 | VMP(TSMOM(12m)) | 1.058 | 272.8% | 0.033 | 1.025 |
| 26 | MSR(LW) | 1.059 | 479.8% | 0.040 | 1.019 |
| 27 | FF3-LowVol | 1.021 | 38.7% | 0.008 | 1.013 |
| 28 | BL-Mom(LW) | 1.042 | 510.8% | 0.038 | 1.003 |
| 29 | HRP(sample) | 1.045 | 396.8% | 0.044 | 1.001 |
| 30 | VMP(MSR\_C(sample)) | 1.055 | 517.9% | 0.054 | 1.001 |
| 31 | SWITCH(sample) | 1.029 | 327.9% | 0.028 | 1.001 |
| 32 | VMP(FF3-Quality) | 1.027 | 375.4% | 0.035 | 0.992 |
| 33 | MSR\_C(LW) | 1.024 | 538.1% | 0.045 | 0.979 |
| 34 | VMP(FF3-Multi) | 1.037 | 787.0% | 0.074 | 0.963 |
| 35 | GMV(sample) | 0.958 | 18.2% | 0.004 | 0.953 |
| 36 | VMP(BL-Mom-LS(LW)) | 1.042 | 456.3% | 0.093 | 0.950 |
| 37 | GMV(LW) | 0.954 | 54.5% | 0.010 | 0.944 |
| 38 | RP(sample) | 0.929 | 8.6% | 0.001 | 0.929 |
| 39 | RP(LW) | 0.926 | 10.9% | 0.001 | 0.925 |
| 40 | EW | 0.924 | 0.6% | 0.000 | 0.924 |
| 41 | TSMOM(6m) | 0.971 | 474.4% | 0.051 | 0.920 |
| 42 | GMV(OAS) | 0.925 | 47.7% | 0.009 | 0.916 |
| 43 | BL-Eq(sample) | 0.915 | 0.9% | 0.000 | 0.915 |
| 44 | BL-Eq(LW) | 0.915 | 0.9% | 0.000 | 0.915 |
| 45 | VMP(MVO\_C(LW)) | 0.877 | 47.6% | 0.007 | 0.870 |
| 46 | MSR(sample) | 0.895 | 494.2% | 0.049 | 0.846 |
| 47 | VMP(MVO\_C(sample)) | 0.850 | 51.5% | 0.007 | 0.842 |
| 48 | FF3-Multi | 0.907 | 787.0% | 0.071 | 0.836 |
| 49 | MSR\_C(sample) | 0.864 | 517.9% | 0.049 | 0.815 |
| 50 | BL-Mom-LS(LW) | 0.904 | 456.3% | 0.091 | 0.813 |
| 51 | VMP(TSMOM-LS(12m)) | 0.833 | 227.4% | 0.033 | 0.800 |
| 52 | FF3-Quality | 0.811 | 375.4% | 0.030 | 0.781 |
| 53 | TSMOM(12m) | 0.801 | 272.8% | 0.032 | 0.769 |
| 54 | VMP(BL-Rev(LW)) | 0.816 | 947.1% | 0.047 | 0.769 |
| 55 | MVO\_C(sample) | 0.757 | 51.5% | 0.007 | 0.750 |
| 56 | MVO\_C(LW) | 0.750 | 47.6% | 0.006 | 0.744 |
| 57 | VMP(FF3-Mom) | 0.804 | 2025.4% | 0.100 | 0.704 |
| 58 | BL-Rev(LW) | 0.663 | 947.1% | 0.041 | 0.622 |
| 59 | TSMOM-LS(12m) | 0.645 | 227.4% | 0.030 | 0.615 |
| 60 | FF3-Mom | 0.685 | 2025.4% | 0.093 | 0.592 |
| 61 | FF3-Mom-LS | 0.103 | 1456.4% | 0.150 | −0.046 |
| 62 | VMP(FF3-Mom-LS) | −0.037 | 1456.4% | 0.150 | −0.187 |

†VMP(GMV(sample)) flagged as a degenerate artifact; see §3.2 and Finding 6.5.

All figures derived directly from `data/published/master_table_62strategies.csv`, column `NetStrat`.

## F.4 Implementability Framing

The implementability filter from §6.1 subjects strategies to three sequential gates: gross Sharpe, cost-adjusted Sharpe, and risk-budget approval. The cost-adjusted ranking in Table F.1 shows that the gross-Sharpe leaders also survive the cost filter when their turnover is low.

**Cost-adjusted leaders** (flat 10 bps, column `Net 10bps`): VMP(MDP(LW)) at 1.336 and VMP(SWITCH(LW)) at 1.201 are the top two genuine configurations after excluding the GMV(sample) artifact. MDP-family strategies dominate because their low-to-moderate turnover (78 bps per year for MDP(LW)) generates negligible cost drag (0.012 Sharpe points).

**SWITCH(LW) net cost loss.** SWITCH(LW) base strategy loses 0.060 Sharpe points at flat 10 bps (gross 1.080 → net 10bps 1.020). Under stratified costs the loss falls to 0.020 points (net 1.060), because SWITCH(LW)'s 2.04% annual turnover is concentrated in the ETF sub-universe priced at 2–3 bps. VMP(SWITCH(LW)) retains net Sharpe 1.201 (flat 10 bps) / 1.243 (stratified) — both placing it in the top five of its respective ranking.

**FF3-Mom collapse (Finding 13).** FF3-Mom's gross Sharpe of 0.685 falls to 0.394 net of flat 10 bps (loss = 0.291 Sharpe points, the largest degradation in the 62-strategy table) and recovers partially to 0.592 under the 5-bps stratified ETF schedule (loss = 0.093 Sharpe points). Even under the more favourable stratified schedule, FF3-Mom ranks 60th of 62 configurations. Its annual turnover of 2,025% disqualifies it as an institutional allocation regardless of cost assumption. VMP(FF3-Mom) (rank 57, net 0.704 stratified) fares marginally better but remains in the bottom decile.

The qualitative conclusion is robust to cost schedule: low-to-moderate turnover strategies with meaningful signal (MDP, SWITCH, VMP variants thereof) dominate the implementability-filtered ranking, while factor strategies that derive return from high-frequency rotation (FF3-Mom, FF3-Multi, BL-Mom(LW)) pay prohibitive cost penalties at any realistic fee level.
