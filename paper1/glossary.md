# Glossary of Acronyms

## Strategy acronyms

| Acronym | Full name | Definition |
|---|---|---|
| BL | Black-Litterman | Bayesian portfolio construction framework blending an equilibrium prior with investor views. |
| EW | Equal Weight | 1/N allocation; each asset receives equal portfolio weight at each rebalance. |
| ERC | Equal Risk Contribution | Synonym for Risk Parity (RP); each asset contributes equally to total portfolio variance. |
| FF3 | Fama-French 3-Factor | Factor-tilt strategy family based on cross-sectional ranking by momentum, low-volatility, quality, or multi-factor composite signals; named after Fama & French (1993). |
| GMV | Global Minimum Variance | Portfolio that minimises total variance subject to full investment and long-only constraints. |
| HRP | Hierarchical Risk Parity | Cluster-and-bisect allocation using Ward-linkage dendrogram; no matrix inversion required (de Prado 2016). |
| L/S | Long-Short | Portfolio construction with both long and short legs; gross exposure = 1, net $\approx$ 0 for market-neutral variants. |
| MDP | Most Diversified Portfolio | Maximises the Diversification Ratio $(\boldsymbol{\sigma}^\top\mathbf{w}) / \sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}}$ (Choueifaty & Coignard 2008). |
| MSR | Maximum Sharpe Ratio | Tangency portfolio; maximises $\hat{\mu}^\top\mathbf{w} / \sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}}$ over long-only weights. |
| MV / MVO | Mean-Variance Optimization | Classical Markowitz (1952) portfolio selection; balances expected return against variance. |
| RP | Risk Parity | Equal Risk Contribution; see ERC. |
| SWITCH | Regime-Conditional Switching | Routes to one of three base strategies (MSR(LW), MSR(sample), MDP(LW)) based on the prevailing macro regime; routing rule derived entirely from training data. |
| TSMOM | Time-Series Momentum | Weights assets by the sign of their trailing return scaled by target volatility (Moskowitz, Ooi & Pedersen 2012). |
| VMP | Volatility-Managed Portfolio | Daily inverse-vol overlay that scales each strategy's exposure by $\bar{\sigma}/\hat{\sigma}_t$, clipped to [0.25×, 1.50×] (Moreira & Muir 2017). |

---

## Estimator acronyms

| Acronym | Full name | Definition |
|---|---|---|
| LW | Ledoit-Wolf shrinkage | Analytically optimal linear shrinkage of the sample covariance matrix toward a structured target; reduces eigenvalue noise and improves conditioning (Ledoit & Wolf 2004). |
| OAS | Oracle Approximating Shrinkage | Non-linear shrinkage estimator minimising the Frobenius-norm distance to the oracle estimator (Chen et al. 2010). |

---

## Regime labels (R0–R7)

The macro classifier assigns one of eight regime labels at each month-end. Labels reflect the
dominant macro character derived from GDP growth, CPI inflation, unemployment, VIX, S&P 500
trailing return, and three yield-curve features (level, slope, curvature).

| Code | Label | Macro character |
|---|---|---|
| R0 | Expansion | Broad-based growth; equities trending; credit spreads narrow. |
| R1 | Recovery | Post-contraction rebound; unemployment falling from peak. |
| R2 | Neutral | Mid-cycle; no strong directional macro signal. |
| R3 | Slow Growth | Below-trend expansion; weak but positive GDP. |
| R4 | Stress | Elevated VIX and/or credit spread widening without full recession. |
| R5 | Low Growth & Contracting | Late-cycle deceleration or early contraction; yield-curve flattening or inversion. |
| R6 | Crisis | Sharp drawdown in equities; spike in VIX; acute risk-off. |
| R7 | Contraction | Sustained below-zero GDP growth; recession. |

Regime coverage over the 2003–2026 sample (5,868 NYSE trading days): R0 27.3%; R1 25.2%;
R2 3.9%; R3 1.4%; R4 3.6%; R5 15.7%; R6 2.5%; R7 20.2%. See Appendix D for the full
methodology and transition matrices.

---

## Other acronyms

| Acronym | Full name |
|---|---|
| CI | Confidence Interval |
| FRED | Federal Reserve Economic Data (St. Louis Fed public economic data series) |
| GFC | Global Financial Crisis (2008–2009) |
| IS | In-Sample (2003–2022 training period in this study) |
| NYSE | New York Stock Exchange (trading-day calendar used throughout) |
| OOS | Out-of-Sample (2023–2026 test period in this study) |
| TC | Transaction Costs |
