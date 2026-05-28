# Appendix D — Regime Classification Pipeline

## D.1 Macro Indicators

The regime engine classifies economic conditions monthly using eight macro indicators. Six series are sourced from the Federal Reserve Economic Data (FRED); two market-based series (VIX, SPX) are sourced from EODHD because FRED does not carry real-time equity volatility data at daily resolution.

| Indicator | Variable | Source / ID | Construction | Lookback |
|---|---|---|---|---|
| Real GDP growth | GDP\_QoQ | FRED: GDPC1 | QoQ percentage change of real GDP | 6 months |
| CPI inflation | CPI\_MoM | FRED: CPIAUCSL | MoM percentage change of headline CPI | 12 months |
| Unemployment rate | UNEM | FRED: UNRATE | Level (percent) | 12 months |
| 10-year Treasury yield | YC\_10Y | FRED: GS10 | Monthly constant-maturity yield | 12 months |
| 2-year Treasury yield | YC\_2Y | FRED: GS2 | Monthly constant-maturity yield | 12 months |
| Yield-curve slope | YC\_STEP | Computed: GS10 − GS2 | 10y–2y term spread | 12 months |
| Equity volatility | VIX | EODHD: VIX.INDX | Monthly mean of daily close | 6 months |
| Equity return | SPX | EODHD: GSPC.INDX | MoM return of S&P 500 Index | 6 months |

All FRED series are indexed by publication date, not the reference period. Data are fetched with a 2000-01-01 start (three-year warmup before the 2003 study window) and resampled to month-end frequency. Macro data used in strategy execution are therefore never observed before their FRED release date, eliminating look-ahead bias.

The lookback column specifies the lag-window parameter $h$ used in the change and convexity features (§D.2). GDP, VIX, and SPX receive a 6-month lookback because they move faster than the slow-moving structural series; all yield and unemployment indicators receive a 12-month lookback.

## D.2 Feature Engineering

For each indicator $x$, three scalar features are computed at each month-end decision date $t$:

**Level (smoothed):**
$$\mathrm{lvl}_t = \frac{1}{3} \sum_{k=0}^{2} x_{t-k}.$$
A 3-month rolling mean suppresses data-revision noise in GDP and CPI while preserving trend.

**Change (first difference over lookback):**
$$\mathrm{chg}_t = \mathrm{lvl}_t - \mathrm{lvl}_{t-h},$$
where $h$ is the indicator-specific lookback in months. Positive $\mathrm{chg}$ means the smoothed level has risen over the past $h$ months.

**Convexity (second difference / acceleration):**
$$\mathrm{conv}_t = \frac{\mathrm{lvl}_t + \mathrm{lvl}_{t-h}}{2} - \mathrm{lvl}_{t-h/2}.$$
Positive convexity indicates the rate of change is accelerating; negative indicates deceleration. An $\varepsilon$-deadband of $10^{-3}$ suppresses noise near zero: observations with $|\mathrm{chg}| \leq \varepsilon$ or $|\mathrm{conv}| \leq \varepsilon$ are assigned the previous month's regime (state persistence).

Eight indicators $\times$ three features $= 24$ feature series are constructed each month. Implementation: `src/aiam/data/regimes/regime_engine.py`, function `compute_features`.

## D.3 Regime Classification Rule

The three binary flags derived from each indicator's features determine a regime index 0–7 via the complete binary decision tree in Table D.1. The level flag is high if $\mathrm{lvl}_t \geq \bar{\mathrm{lvl}}_{t,60}$, the 5-year (60-month) rolling mean of the smoothed level.

**Table D.1. Regime coding.**

| Regime | Level | Change | Convexity | Label |
|---|---|---|---|---|
| R0 | High | ↑ | Accelerating | Expansion |
| R1 | High | ↑ | Decelerating | Recovery |
| R2 | High | ↓ | Decelerating | Neutral |
| R3 | Low | ↑ | Decelerating | Slow Growth |
| R4 | High | ↓ | Accelerating | Stress |
| R5 | Low | ↓ | Accelerating | Low & Contracting |
| R6 | Low | ↑ | Accelerating | Crisis |
| R7 | Low | ↓ | Decelerating | Contraction |

Each of the eight indicators produces its own regime classification via this rule. The **dominant regime** at month $t$ is the mode across the eight per-indicator classifications:
$$r_t^* = \mathrm{mode}\bigl( r_t^{\mathrm{GDP}},\; r_t^{\mathrm{CPI}},\; r_t^{\mathrm{UNEM}},\; r_t^{\mathrm{YC10}},\; r_t^{\mathrm{YC2}},\; r_t^{\mathrm{YCSTEP}},\; r_t^{\mathrm{VIX}},\; r_t^{\mathrm{SPX}} \bigr).$$

The mode is forward-looking-free: `pd.DataFrame.mode(axis=1)` operates row-wise on the concurrent per-indicator outputs. The dominant regime is forward-filled to daily frequency within `SwitchingStrategy`; no future information enters at the daily level.

Implementation: `src/aiam/data/regimes/regime_engine.py`, function `build_regime_signals`. Precomputed regime parquet: `data/published/regime_signals.parquet`.

**Citation.** López de Prado (2016).

## D.4 Full-Sample Regime Distribution (2003–2026)

The regime parquet is indexed monthly. The 5,868 study trading days are assigned via forward-fill from the month-end dominant regime.

| Regime | Label | Trading Days | Share |
|---|---|---|---|
| R0 | Expansion | 1,603 | 27.3% |
| R1 | Recovery | 1,481 | 25.2% |
| R2 | Neutral | 230 | 3.9% |
| R3 | Slow Growth | 85 | 1.4% |
| R4 | Stress | 210 | 3.6% |
| R5 | Low & Contracting | 924 | 15.7% |
| R6 | Crisis | 148 | 2.5% |
| R7 | Contraction | 1,187 | 20.2% |
| **Total** | | **5,868** | **100%** |

Expansion (R0) and Recovery (R1) together account for 52.5% of sample days. The two contraction regimes (R5 + R7) account for 35.9%. Rare regimes (R2–R4, R6) together total less than 12% of the sample, which limits the statistical precision of conditional Sharpe estimates within those regimes.

## D.5 Regime Transition Matrix

Table D.2 shows month-to-month transition probabilities (row = origin, column = destination), estimated from 280 monthly observations in the 2003–2026 study window. Sparse regimes (R3, R6) have fewer than 10 observed transitions each; their rows should be interpreted with caution.

**Table D.2. Regime transition probabilities (%) — monthly frequency.**

| From \ To | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|---|
| R0 | 57.6 | 17.6 | 1.2 | 0.0 | 2.4 | 8.2 | 0.0 | 12.9 |
| R1 | 11.7 | 59.7 | 3.9 | 1.3 | 2.6 | 9.1 | 1.3 | 10.4 |
| R2 | 18.2 | 9.1 | 36.4 | 9.1 | 9.1 | 0.0 | 9.1 | 9.1 |
| R3 | 25.0 | 0.0 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 |
| R4 | 50.0 | 8.3 | 8.3 | 0.0 | 16.7 | 16.7 | 0.0 | 0.0 |
| R5 | 14.3 | 4.1 | 2.0 | 0.0 | 6.1 | 61.2 | 4.1 | 8.2 |
| R6 | 42.9 | 0.0 | 0.0 | 0.0 | 14.3 | 0.0 | 42.9 | 0.0 |
| R7 | 13.8 | 18.5 | 0.0 | 3.1 | 1.5 | 4.6 | 0.0 | 58.5 |

Diagonal dominance is strong for the high-frequency regimes: R0 stays R0 57.6% of months, R1 stays R1 59.7%, R5 stays R5 61.2%, R7 stays R7 58.5%. The two main escape routes from Contraction (R7) are Recovery (R1, 18.5%) and Expansion (R0, 13.8%), consistent with the classic recession-to-recovery cyclical pattern.

## D.6 Cross-Reference

The SWITCH(v2a) routing rule — R0→MSR(LW), R5→MSR(sample), default→MDP(LW) — was derived from training-period conditional Sharpe analysis and applied OOS without re-estimation. See §5 of the main paper and Appendix B §B.10 for the full routing specification.
