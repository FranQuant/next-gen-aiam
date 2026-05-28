# Appendix C — Master Strategy Table (62 Configurations)

Source: `data/published/master_table_62strategies.csv`. All statistics are computed over the full sample 2003-01-02 to 2026-04-30 (5,869 NYSE trading days). Annualised return and volatility use $\sqrt{252}$ scaling. Sharpe ratio is computed at $r_f = 0$ (annualised return / annualised vol). Max drawdown is the peak-to-trough decline on the cumulative return series. Calmar ratio is annualised return / |max drawdown|.

Within each family, base configurations are listed first, VMP-overlay variants immediately below. For OOS statistics (test period 2023-01-01 to 2026-04-30) see Table 1 and the discussion in §6.2 of the main paper. Row count confirmed: **62**.

---

## Classical MV

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| EW | 12.60% | 13.89% | 0.924 | −37.86% | 0.333 |
| VMP(EW) | 15.31% | 13.37% | 1.133 | −27.32% | 0.560 |
| GMV(sample) | 3.02% | 3.16% | 0.958 | −9.62% | 0.314 |
| VMP(GMV(sample)) | 3.13% | 2.31% | 1.345 | −8.53% | 0.367 |
| GMV(ledoit_wolf) | 3.81% | 4.01% | 0.954 | −11.09% | 0.344 |
| VMP(GMV(ledoit_wolf)) | 4.47% | 3.65% | 1.215 | −12.66% | 0.353 |
| GMV(oas) | 3.36% | 3.64% | 0.925 | −10.10% | 0.332 |
| VMP(GMV(oas)) | 3.86% | 3.18% | 1.207 | −11.60% | 0.333 |
| MSR(sample) | 6.71% | 7.58% | 0.895 | −19.99% | 0.336 |
| VMP(MSR(sample)) | 7.59% | 5.78% | 1.295 | −10.10% | 0.751 |
| MSR(ledoit_wolf) | 11.21% | 10.56% | 1.059 | −19.85% | 0.565 |
| VMP(MSR(ledoit_wolf)) | 13.07% | 10.34% | 1.239 | −21.37% | 0.612 |

---

## Diversification

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| MDP(sample) | 5.67% | 5.13% | 1.101 | −17.75% | 0.320 |
| VMP(MDP(sample)) | 6.52% | 4.70% | 1.368 | −12.71% | 0.513 |
| MDP(ledoit_wolf) | 6.55% | 5.57% | 1.166 | −14.80% | 0.443 |
| VMP(MDP(ledoit_wolf)) | 7.70% | 5.52% | 1.372 | −13.03% | 0.591 |
| RP(sample) | 11.65% | 12.74% | 0.929 | −33.10% | 0.352 |
| VMP(RP(sample)) | 14.09% | 12.60% | 1.110 | −26.23% | 0.537 |
| RP(ledoit_wolf) | 11.62% | 12.77% | 0.926 | −32.90% | 0.353 |
| VMP(RP(ledoit_wolf)) | 14.10% | 12.62% | 1.108 | −26.15% | 0.539 |
| HRP(sample) | 6.80% | 6.50% | 1.045 | −16.57% | 0.410 |
| VMP(HRP(sample)) | 7.54% | 6.36% | 1.176 | −15.10% | 0.500 |
| HRP(ledoit_wolf) | 7.14% | 6.51% | 1.093 | −15.65% | 0.457 |
| VMP(HRP(ledoit_wolf)) | 7.99% | 6.40% | 1.232 | −13.41% | 0.596 |

---

## Regime Switch

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| SWITCH(sample) | 8.31% | 8.07% | 1.029 | −19.04% | 0.436 |
| VMP(SWITCH(sample)) | 9.27% | 7.05% | 1.293 | −13.11% | 0.707 |
| SWITCH(ledoit_wolf) | 9.55% | 8.81% | 1.080 | −20.16% | 0.474 |
| VMP(SWITCH(ledoit_wolf)) | 10.60% | 8.24% | 1.265 | −17.39% | 0.610 |

---

## TS Momentum

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| TSMOM(12m) | 5.45% | 6.93% | 0.801 | −21.68% | 0.252 |
| VMP(TSMOM(12m)) | 6.98% | 6.58% | 1.058 | −13.80% | 0.506 |
| TSMOM(6m) | 7.02% | 7.25% | 0.971 | −24.18% | 0.290 |
| VMP(TSMOM(6m)) | 7.70% | 6.75% | 1.133 | −12.47% | 0.618 |

---

## Black-Litterman

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| BL-Eq(sample) | 12.48% | 13.92% | 0.915 | −37.86% | 0.330 |
| VMP(BL-Eq(sample)) | 15.12% | 13.41% | 1.118 | −27.39% | 0.552 |
| BL-Eq(LW) | 12.48% | 13.92% | 0.915 | −37.86% | 0.330 |
| VMP(BL-Eq(LW)) | 15.12% | 13.41% | 1.118 | −27.39% | 0.552 |
| BL-Mom(LW) | 12.57% | 12.07% | 1.042 | −21.34% | 0.589 |
| VMP(BL-Mom(LW)) | 14.65% | 11.81% | 1.217 | −21.84% | 0.671 |
| BL-Rev(LW) | 12.09% | 20.34% | 0.663 | −51.42% | 0.235 |
| VMP(BL-Rev(LW)) | 13.95% | 18.02% | 0.816 | −45.86% | 0.304 |

---

## Factor

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| FF3-Mom | 11.03% | 17.52% | 0.685 | −41.03% | 0.269 |
| VMP(FF3-Mom) | 12.37% | 16.13% | 0.804 | −29.61% | 0.418 |
| FF3-LowVol | 4.34% | 4.25% | 1.021 | −10.68% | 0.406 |
| VMP(FF3-LowVol) | 4.59% | 3.92% | 1.165 | −11.15% | 0.412 |
| FF3-Quality | 7.59% | 9.59% | 0.811 | −23.15% | 0.328 |
| VMP(FF3-Quality) | 8.59% | 8.37% | 1.027 | −17.42% | 0.493 |
| FF3-Multi | 7.95% | 8.87% | 0.907 | −19.85% | 0.400 |
| VMP(FF3-Multi) | 8.80% | 8.48% | 1.037 | −15.29% | 0.575 |

---

## Constrained MV

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| MSR\_C(ledoit\_wolf) | 10.52% | 10.29% | 1.024 | −19.48% | 0.540 |
| VMP(MSR\_C(ledoit\_wolf)) | 12.27% | 10.19% | 1.187 | −18.38% | 0.668 |
| MSR\_C(sample) | 6.82% | 8.01% | 0.864 | −18.68% | 0.365 |
| VMP(MSR\_C(sample)) | 7.70% | 7.28% | 1.055 | −13.19% | 0.584 |
| MVO\_C(ledoit\_wolf) | 2.98% | 4.02% | 0.750 | −13.51% | 0.221 |
| VMP(MVO\_C(ledoit\_wolf)) | 3.22% | 3.69% | 0.877 | −14.25% | 0.226 |
| MVO\_C(sample) | 2.95% | 3.94% | 0.757 | −14.05% | 0.210 |
| VMP(MVO\_C(sample)) | 2.99% | 3.54% | 0.850 | −14.70% | 0.203 |

---

## Long-Short

| Configuration | Ann Return | Ann Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| TSMOM-LS(12m) | 3.67% | 5.86% | 0.645 | −16.42% | 0.224 |
| VMP(TSMOM-LS(12m)) | 4.48% | 5.44% | 0.833 | −14.29% | 0.313 |
| BL-Mom-LS(LW) | 4.18% | 4.65% | 0.904 | −11.87% | 0.352 |
| VMP(BL-Mom-LS(LW)) | 4.78% | 4.58% | 1.042 | −11.95% | 0.400 |
| FF3-Mom-LS | 0.53% | 8.99% | 0.103 | −26.97% | 0.020 |
| VMP(FF3-Mom-LS) | −0.73% | 8.99% | −0.037 | −35.32% | −0.021 |

---

**Row count: 62.** All figures match `data/published/master_table_62strategies.csv` to the precision shown. The BL-Eq(sample) and BL-Eq(LW) rows are numerically identical (differences below rounding threshold) because the equilibrium prior $\boldsymbol{\pi}$ and view matrix are identical in both variants; the only distinction is the shrinkage applied to $\boldsymbol{\Sigma}$ in the MSR solve, which converges to the same solution on this universe.
