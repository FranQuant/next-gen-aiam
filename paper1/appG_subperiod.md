# Appendix G — Sub-Period Sensitivity Tables

This appendix supplies the numerical data underlying the sub-period discussion in §6.3. Two
tables are provided: calendar-year returns for all 24 strategies from the §6.3 heatmap (Table G.1a and
G.1b, split across time), and the five-sub-period Sharpe table for eight representative
strategies that is referenced in the §6.3 narrative (Table G.2).

---

## Table G.1a — Calendar-Year Returns, 2003–2014 (%)

24 strategies × 12 years. Returns are compounded annual returns (simple, not log), expressed as
percentages rounded to one decimal place. Strategies in **2003** show uniform values within
covariance-based families because insufficient trailing history is available at inception;
allocations default to equal-weight before sufficient estimation data accumulates.
Source: `data/published/strategy_returns_base.parquet` and
`data/cache/portfolio_returns/switch_v2a_oos_29assets.parquet`.

```{=latex}
\begin{landscape}
```

| Strategy | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EW | +25.3 | +21.0 | +16.3 | +18.8 | +20.8 | −25.0 | +31.7 | +11.7 | +4.3 | +11.2 | +13.7 | +5.4 |
| GMV(samp) | +24.0 | +1.9 | +1.4 | +4.2 | +8.1 | +2.4 | +1.5 | +2.4 | +1.6 | +0.3 | +0.5 | +0.6 |
| GMV(LW) | +24.0 | +5.0 | +0.9 | +5.6 | +9.0 | +1.2 | +1.4 | +3.3 | +4.9 | +3.0 | +1.3 | +0.1 |
| MSR(samp) | +24.0 | +10.6 | +12.4 | +3.8 | +9.0 | +1.9 | −0.6 | +3.5 | +3.1 | +5.5 | +3.8 | +6.2 |
| MSR(LW) | +24.0 | +12.8 | +25.0 | +3.3 | +11.7 | +1.1 | −5.5 | +6.6 | +8.1 | +7.1 | +11.9 | +14.8 |
| MDP(samp) | +24.0 | +9.1 | +3.1 | +10.2 | +10.5 | +3.2 | +2.6 | +3.1 | +8.3 | +2.4 | +4.5 | +3.3 |
| MDP(LW) | +24.0 | +9.3 | +4.3 | +8.9 | +13.1 | +2.0 | +4.2 | +3.7 | +9.1 | +3.3 | +4.4 | +3.2 |
| RP(samp) | +24.0 | +14.9 | +13.0 | +21.5 | +21.3 | −21.2 | +21.8 | +11.5 | +4.2 | +11.3 | +13.7 | +5.4 |
| RP(LW) | +24.0 | +14.8 | +13.0 | +21.5 | +21.0 | −20.9 | +20.2 | +11.5 | +4.1 | +11.3 | +13.7 | +5.4 |
| HRP(samp) | +24.0 | +6.7 | +6.3 | +9.8 | +9.8 | −3.5 | +6.6 | +5.3 | +4.6 | +7.6 | +7.8 | +3.3 |
| HRP(LW) | +24.0 | +5.9 | +3.3 | +10.1 | +10.5 | −2.3 | +8.0 | +6.0 | +5.9 | +8.1 | +7.1 | +2.8 |
| SWITCH(samp) | +25.3 | +13.2 | +6.2 | +5.8 | +15.4 | +2.6 | +7.2 | +3.1 | +6.1 | +5.8 | +7.3 | +1.2 |
| SWITCH(LW) | +25.3 | +14.0 | +8.4 | +11.5 | +16.5 | −3.3 | +7.6 | +3.7 | +7.2 | +7.5 | +13.9 | +1.4 |
| TSMOM(12m) | +24.0 | +9.2 | +4.6 | +9.5 | +12.9 | −3.2 | +3.9 | +5.4 | +2.3 | +3.4 | +8.4 | +4.3 |
| TSMOM(6m) | +20.2 | +6.8 | +4.8 | +9.1 | +13.5 | −2.9 | +8.8 | +4.3 | +5.4 | +2.2 | +13.8 | +4.8 |
| BL-Eq(samp) | +24.0 | +18.7 | +14.0 | +21.5 | +21.2 | −25.0 | +31.7 | +11.7 | +4.3 | +11.2 | +13.7 | +5.4 |
| BL-Eq(LW) | +24.0 | +18.7 | +14.0 | +21.5 | +21.2 | −25.0 | +31.7 | +11.7 | +4.3 | +11.2 | +13.7 | +5.4 |
| BL-Mom(LW) | +24.0 | +13.8 | +32.7 | 0.0 | +12.2 | +2.1 | −6.7 | +9.2 | +8.1 | +7.2 | +12.1 | +15.6 |
| BL-Rev(LW) | +24.0 | +22.0 | −6.3 | +23.8 | +22.7 | −36.5 | +33.0 | +17.3 | +10.4 | +24.6 | +0.9 | −8.6 |
| FF3-Mom | +24.0 | +23.3 | +8.6 | +18.0 | +18.9 | −18.3 | +24.4 | +6.8 | +12.9 | +7.4 | +16.9 | +2.7 |
| FF3-LowVol | +24.0 | +4.1 | +1.7 | +5.6 | +9.0 | +4.8 | +2.0 | +4.7 | +4.9 | +3.1 | +3.7 | +1.5 |
| FF3-Qlty | +24.0 | +13.3 | +14.5 | +11.5 | +13.6 | −4.3 | −1.0 | +7.5 | +6.2 | +4.0 | +6.8 | +8.0 |
| FF3-Multi | +24.1 | +13.4 | +8.4 | +11.7 | +14.0 | −4.8 | +8.3 | +6.6 | +8.5 | +4.9 | +9.1 | +4.1 |
| SWITCH(v2a) | +24.0 | +15.6 | +16.1 | +9.6 | +16.8 | +11.8 | −4.2 | +9.6 | +5.0 | +5.2 | +5.3 | +16.0 |

```{=latex}
\end{landscape}
```

---

## Table G.1b — Calendar-Year Returns, 2015–2026* (%)

Same 24 strategies. 2026* = through April 30, 2026 (partial year; annualised figures should
not be inferred from this column).

```{=latex}
\begin{landscape}
```

| Strategy | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EW | −2.2 | +16.7 | +20.3 | −5.4 | +25.9 | +13.9 | +21.6 | −6.5 | +16.9 | +19.9 | +24.4 | +11.0 |
| GMV(samp) | +0.4 | +1.4 | +1.1 | +1.1 | +4.6 | +2.3 | −0.5 | −4.1 | +5.0 | +5.9 | +5.3 | +1.4 |
| GMV(LW) | −1.8 | +3.1 | +7.4 | −0.3 | +7.9 | +2.9 | −1.2 | −7.3 | +4.3 | +5.2 | +8.8 | +3.4 |
| MSR(samp) | +3.8 | +9.1 | +9.0 | −4.1 | +5.2 | +3.4 | +5.2 | +4.2 | +15.4 | +14.5 | +7.8 | +2.9 |
| MSR(LW) | +4.5 | +13.5 | +20.7 | −11.9 | +12.6 | +13.5 | +6.9 | +4.6 | +17.9 | +33.3 | +24.3 | +10.3 |
| MDP(samp) | −2.9 | +5.1 | +12.7 | +0.5 | +8.3 | +2.7 | 0.0 | −9.7 | +2.8 | +16.9 | +11.3 | +4.8 |
| MDP(LW) | −3.8 | +6.7 | +13.4 | −1.0 | +12.9 | +3.9 | +0.9 | −9.0 | +4.1 | +17.1 | +15.0 | +8.1 |
| RP(samp) | −2.2 | +16.7 | +20.3 | −5.4 | +25.3 | +10.3 | +18.9 | −6.0 | +12.4 | +19.9 | +22.7 | +11.1 |
| RP(LW) | −2.2 | +16.7 | +20.3 | −5.4 | +25.5 | +9.9 | +18.8 | −6.1 | +12.5 | +19.9 | +24.4 | +11.0 |
| HRP(samp) | −1.0 | +8.3 | +10.8 | −3.1 | +12.6 | +7.4 | +5.8 | −5.9 | +8.4 | +10.1 | +15.5 | +5.3 |
| HRP(LW) | −2.3 | +6.6 | +12.3 | −0.3 | +13.2 | +10.1 | +7.6 | −6.5 | +7.3 | +11.7 | +16.3 | +5.1 |
| SWITCH(samp) | +3.0 | +6.5 | +12.8 | +5.5 | +2.1 | +12.2 | +10.1 | −9.9 | +24.7 | +17.4 | +12.8 | +2.8 |
| SWITCH(LW) | +1.2 | +11.7 | +14.5 | +4.5 | +6.7 | +14.0 | +11.4 | −10.1 | +23.3 | +16.7 | +15.2 | +6.3 |
| TSMOM(12m) | +0.5 | +3.1 | +11.7 | −6.9 | +12.8 | +2.3 | +7.8 | −9.6 | −2.0 | +11.4 | +12.5 | +4.2 |
| TSMOM(6m) | −2.7 | +7.8 | +15.1 | −1.8 | +10.3 | +2.3 | +7.2 | +7.5 | +6.9 | +9.5 | +9.5 | +3.9 |
| BL-Eq(samp) | −2.2 | +16.7 | +20.3 | −5.4 | +25.9 | +13.9 | +21.6 | −6.5 | +16.9 | +19.9 | +24.4 | +11.0 |
| BL-Eq(LW) | −2.2 | +16.7 | +20.3 | −5.4 | +25.9 | +13.9 | +21.6 | −6.5 | +16.9 | +19.9 | +24.4 | +11.0 |
| BL-Mom(LW) | +4.0 | +23.0 | +23.0 | −11.1 | +12.4 | +12.7 | +5.6 | +7.1 | +19.7 | +44.9 | +26.6 | +8.9 |
| BL-Rev(LW) | −15.5 | +46.2 | +13.1 | +16.7 | +15.1 | +3.0 | +33.5 | −6.5 | +14.5 | +30.0 | +25.8 | +16.4 |
| FF3-Mom | −4.0 | +17.8 | +20.5 | −11.0 | +20.0 | −7.1 | +18.5 | −2.4 | +14.1 | +22.3 | +31.9 | +7.8 |
| FF3-LowVol | −0.7 | +3.3 | +4.7 | −0.2 | +8.0 | +5.4 | +0.8 | −6.3 | +5.4 | +5.0 | +7.5 | +1.9 |
| FF3-Qlty | +0.6 | −1.9 | +20.7 | −4.1 | +12.4 | +8.6 | +3.1 | −16.4 | +15.3 | +26.0 | +15.2 | +3.5 |
| FF3-Multi | −1.2 | +6.3 | +15.1 | −4.9 | +13.5 | +3.1 | +7.4 | −8.1 | +11.7 | +17.5 | +18.0 | +4.4 |
| SWITCH(v2a) | +1.8 | +13.6 | +15.6 | −1.0 | +17.2 | +18.2 | +1.4 | −6.0 | +21.0 | +17.2 | +13.8 | +10.2 |

```{=latex}
\end{landscape}
```

---

## Table G.2 — Five-Sub-Period Sharpe Ratios

Eight representative strategies × five non-overlapping sub-periods spanning 2003–2026. Sharpe
ratios are annualised ($r_f = 0$). This table is the quantitative basis for the narrative in
§6.3. The SWITCH(v2a) row uses the training-only routing (derived from 2003–2022 only); all
other rows come from `data/published/strategy_returns_base.parquet` and
`strategy_returns_vmp.parquet`. Values are rounded to two decimal places.

| Strategy | 2003–2007 | 2008–2012 | 2013–2017 | 2018–2022 | 2023–2026 |
|---|---:|---:|---:|---:|---:|
| EW | 1.72 | 0.35 | 1.17 | 0.63 | 2.03 |
| MSR(LW) | 1.42 | 0.53 | 1.51 | 0.41 | 1.80 |
| MSR(sample) | 1.53 | 0.89 | 1.44 | 0.30 | 1.19 |
| MDP(LW) | 1.61 | 0.82 | 1.27 | 0.26 | 2.30 |
| SWITCH(LW) | 1.59 | 0.58 | 1.24 | 0.53 | 1.73 |
| SWITCH(v2a) | 1.24 | 1.23 | 1.36 | 0.48 | 2.01 |
| VMP(MSR(LW)) | 1.57 | 0.70 | 1.46 | 0.50 | 2.18 |
| VMP(MDP(LW)) | 1.87 | 1.02 | 1.26 | 0.56 | 2.43 |

---

## G.3 Reading the tables

Two patterns dominate.

**Within-strategy variation dwarfs cross-strategy variation.** MSR(LW), one of the better
base strategies on a full-sample basis (Sharpe 1.059), swings from 0.41 in 2018–2022 to 1.51
in 2013–2017 — a within-strategy range of 1.10 Sharpe points. The full-sample cross-strategy
spread from the weakest non-degenerate base strategy to the strongest spans 0.50 Sharpe points.
The within-strategy range is more than twice that spread. MDP(LW) reaches its full-sample best
(Sharpe 1.167) by averaging a 2.30 in 2023–2026 against a 0.26 in 2018–2022. The headline
ranking reflects this specific 23-year macro sequence, not a stable ordering of strategy quality.

**SWITCH(v2a) is the partial exception.** Its 2008–2012 sub-period Sharpe of 1.23 is the
highest in Table G.2 for that window — crisis protection from routing to MSR(sample) in R5
(Low & Contracting). Its five sub-period values (1.24, 1.23, 1.36, 0.48, 2.01) are more
stable than any mean-variance strategy in the table: the worst sub-period is 0.48 (2018–2022)
versus 0.26 for MDP(LW) and 0.30 for MSR(sample). Yet even SWITCH(v2a) drops sharply in
2018–2022, demonstrating that regime conditioning reduces sub-period variance without eliminating
it. The three mechanisms identified in §§3–5 — VMP, shrinkage, regime conditioning — persist
as directional benefits across sub-periods; the *ranking* of specific configurations does not.

In the calendar-year tables (G.1a–b), BL-Rev(LW) and BL-Mom(LW) exhibit the largest
year-to-year swings: BL-Rev(LW) reaches +46.2% in 2016 and −36.5% in 2008; BL-Mom(LW)
hits +44.9% in 2024 and −11.1% in 2018. SWITCH(LW) and HRP variants show the smallest
year-to-year variation in the base-strategy group, consistent with their structural design
goals (diversification and regime-conditional routing, respectively).
