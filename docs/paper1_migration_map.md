# Paper 1 — Paragraph Migration Map

Source: `docs/results.md` (1,354 lines, current structure)
Target: new six-section + appendix structure defined below
Date: 2026-05-27

## Target section codes

| Code | Full title |
|------|-----------|
| §1   | Introduction |
| §2   | Data, Methods, and Naive Horse Race |
| §3   | VMP is a Universal Sharpe-Improver |
| §4   | Estimator Choice Matters |
| §5   | Regimes Add Value |
| §6   | Costs, Survival, and Sub-Period Robustness |
| §7   | Conclusion |
| App A | Universe description |
| App B | Strategy zoo (all families, specs, math) |
| App C | Master performance table |
| App D | Regime pipeline |
| App E | Statistical robustness |
| App F | Transaction-cost detail |
| App G | Sub-period tables |
| App H | Reproducibility |
| CUT  | Drop entirely |

## Action codes

| Code | Meaning |
|------|---------|
| KEEP_AS_IS | Copy verbatim with at most minor wording edits |
| COMPRESS | Keep the key claim; cut supporting detail by ≥50% |
| RE_THREAD | Move to different section; light rewording for flow |
| MOVE_TO_APPENDIX | Exact content moves; no loss of information |
| CUT | Remove; content either redundant or out of scope |
| REWRITE | Blank-page rewrite; source is only inspiration |
| SPLIT | Single source block feeds two destinations |

---

## Migration table

| # | Source (line range / heading) | Topic — first 8 words | Destination | Action |
|--:|-------------------------------|------------------------|-------------|--------|
| 1 | 8–18 Abstract | This study evaluates 62 portfolio allocation strategies | §1 | REWRITE |
| 2 | 25–36 §1 > Motivation ¶1 | Portfolio construction has been a central problem | §1 | KEEP_AS_IS |
| 3 | 37–46 §1 > Motivation ¶2 | The benchmark proposed by @demiguel2009optimal raised | §1 | KEEP_AS_IS |
| 4 | 48–51 §1 > Motivation ¶3 | This study expands the comparison to 62 strategies | §1 | COMPRESS |
| 5 | 54–62 §1 > Related Work ¶1 | The foundational challenge of portfolio out-of-sample | §1 | KEEP_AS_IS |
| 6 | 64–77 §1 > Related Work ¶2 | Diversification-based strategies that side-step expected | §1 | KEEP_AS_IS |
| 7 | 79–83 §1 > Related Work ¶3 | Volatility-managed portfolios scale exposure inversely | §1 | KEEP_AS_IS |
| 8 | 85–106 §1 > Contribution (numbered list) | This study makes three contributions | §1 | REWRITE (update to new section structure) |
| 9 | 113–127 §2 > Universe ¶1 | The evaluation universe comprises 29 tickers spanning | App A | MOVE_TO_APPENDIX |
| 10 | 129–138 §2 > Universe ¶2 | The 2003 start date extends the sample | App A (detail) + §2 (Sharpe formula) | SPLIT |
| 11 | 140–157 §2 > Walk-Forward Harness (bullet list) | All 62 strategies evaluated through common walk-forward | §2 | KEEP_AS_IS |
| 12 | 159–173 §2 > Backtesting Hygiene (numbered list) | Six practices govern the harness implementation | §2 | KEEP_AS_IS |
| 13 | 179–196 §2 > Strategy Families > Classical MV | Global Minimum Variance minimizes portfolio variance | App B | MOVE_TO_APPENDIX |
| 14 | 200–222 §2 > Strategy Families > Diversification-Based | Maximum Diversification Portfolio maximizes diversification ratio | App B | MOVE_TO_APPENDIX |
| 15 | 225–235 §2 > Strategy Families > Black-Litterman | Black-Litterman combines equilibrium prior with investor | App B | MOVE_TO_APPENDIX |
| 16 | 237–246 §2 > Strategy Families > TS Momentum | Time-series momentum constructs positions proportional | App B | MOVE_TO_APPENDIX |
| 17 | 250–255 §2 > Strategy Families > Factor Portfolios | Factor portfolios follow rank-then-weight approach: assets | App B | MOVE_TO_APPENDIX |
| 18 | 259–272 §2 > VMP Overlay (formula + figure caption) | VMP overlay scales each strategy's daily exposure | §3 | RE_THREAD (mechanism belongs with the universal-lift section) |
| 19 | 276–296 §2 > Regime Classification | Regime engine classifies macro state from eight FRED | App D (full) + §5 (2-sentence summary) | SPLIT |
| 20 | 302–307 §3 > 62-Strategy Comparison intro ¶ | Across all 62 strategies, the top three | §2 | RE_THREAD (horse-race summary) |
| 21 | 308–412 Table 1 (62-strategy longtable) | Full 62-strategy comparison longtable LaTeX block | App C | MOVE_TO_APPENDIX |
| 22 | 414 Figure: cumulative_wealth.png | Cumulative wealth curves on log-y axis 2003–2026 | §2 | KEEP_AS_IS (primary visual for horse-race narrative) |
| 23 | 416 Figure: underwater_drawdown.png | Underwater drawdown paths for top 5 non-degenerate | §2 | KEEP_AS_IS |
| 24 | 418 Figure: strategy_correlation_matrix.png | Pairwise return correlation matrix for all 62 | App E | MOVE_TO_APPENDIX |
| 25 | 420 Figure: asset_class_allocation_timeline.png | Asset-class allocation over time for four representative | §5 | RE_THREAD (shows regime-driven reallocation) |
| 26 | 426–441 §3 > Rankings ¶ + Top-10 raw table | Sharpe ratio vs. maximum drawdown figure + table | §2 | COMPRESS (headline only; full table to App C) |
| 27 | 443–458 Rankings — Top-10 excl. artifact table | Top 10 by Sharpe excluding SHY-concentration artifact | §2 | COMPRESS (top 3 with brief note) |
| 28 | 460–478 Rankings — Top-5 return / Bottom-5 Sharpe | Top 5 by annualized return, bottom 5 by Sharpe | App C | MOVE_TO_APPENDIX |
| 29 | 482–486 §3.3 > VMP cost footnote blockquote | VMP exposure scaling assumed costless in this sensitivity | App F | MOVE_TO_APPENDIX (footnote in App F) |
| 30 | 488–490 §3.3 > TC uniform-10-bps intro sentence | All figures below apply uniform 10 bps round-trip | App F | MOVE_TO_APPENDIX |
| 31 | 491–506 §3.3 > Top-10 net 10bps table + note | Top 10 by Sharpe net of 10 bps (artifact excluded) | §6 | RE_THREAD |
| 32 | 508–516 §3.3 > Top-5 Sharpe degradation table | Top 5 strategies by Sharpe degradation base only | §6 | RE_THREAD |
| 33 | 518–532 §3.3 > Reading ¶ | At 10 bps round-trip, cost impact separates into | §6 | RE_THREAD |
| 34 | 534–558 §3.3.4 > Asset-class-stratified costs ¶¶ | Uniform 10 bps flat-rate assumption is conservative | App F | MOVE_TO_APPENDIX |
| 35 | 560 Figure: stratified_vs_flat_costs.png | Scatter of net Sharpe under flat 10 bps | App F | MOVE_TO_APPENDIX |
| 36 | 566–576 Finding 1 — GMV(sample) cash corner | GMV(sample) reports vol=3.16%, ret=3.02%, Sharpe=0.958 | §4 | COMPRESS (fold into estimator-choice narrative; drop rf re-computation) |
| 37 | 578–587 Finding 2 — MSR Michaud overfit | MSR(sample) Sharpe=0.895 is among lower base-strategy | §4 | KEEP_AS_IS |
| 38 | 589–600 Finding 3 — HRP shrinkage invariance | In 29-asset 2003–2026 sample, HRP(sample) Sharpe=1.045 | §4 | KEEP_AS_IS |
| 39 | 602–610 Finding 4 — Regime 5 exception | In regime-conditional Sharpe table, Regime 5 produces | §5 | KEEP_AS_IS |
| 40 | 612–627 Finding 5 — SWITCH(v2a) construction | Original SWITCH(LW) rule assigns R0→EW, R5→MSR(LW) | §5 | KEEP_AS_IS |
| 41 | 629–646 Finding 6 — VMP 24/24 universal lift | VMP lifts Sharpe for every one of original 24 | §3 | KEEP_AS_IS |
| 42 | 648–651 Finding 6.5 — VMP(GMV) rank-1 artifact | VMP(GMV(sample)) Sharpe=1.533 is highest but artifact | App C (footnote) | CUT from main text; retain as note in App C |
| 43 | 653–661 Finding 7 — VMP makes shrinkage redundant | VMP(MSR(sample)) Sharpe=1.295 surpasses MSR(LW) Sharpe | §3 | KEEP_AS_IS |
| 44 | 663–673 Finding 8 — TSMOM weakest; VMP rescues | TSMOM(12m) Sharpe=0.801 is among weaker base-strategy | §2 | COMPRESS (one sentence in horse-race discussion) |
| 45 | 675–686 Finding 9 — BL-Mom return leader | BL-Mom(LW) annualized return=12.57% is among higher | §2 | COMPRESS (absorbed into horse-race ranking paragraph) |
| 46 | 688–697 Finding 10 — BL circularity lemma | BL-Eq(sample) and BL-Eq(LW) produce return series | §4 | KEEP_AS_IS |
| 47 | 699–710 Finding 11 — Low-vol anomaly unleveraged | FF3-LowVol achieves Sharpe=1.021 with vol=4.25% | §2 | COMPRESS (one sentence; leveraging discussion cut) |
| 48 | 712–722 Finding 12 — VMP + regime partial substitutes | VMP(SWITCH(v2a)) Sharpe improvement from regime switching | §3 (VMP layer) + §5 (regime layer) | RE_THREAD (split claim across both sections) |
| 49 | 724–747 Finding 13 — TC survival | At 10 bps round-trip, Sharpe landscape reorganizes | §6 | KEEP_AS_IS |
| 50 | 749–766 Finding 14 — Long-short underperforms | Activating short leg in heterogeneous 29-asset universe | §6 | COMPRESS (2 sentences on LS as risk-profile tool) |
| 51 | 768–782 Finding 15 — BTC excluded | BTC-USD excluded entirely to eliminate forward-fill | App A | MOVE_TO_APPENDIX |
| 52 | 789–792 §5 Stat Robustness intro ¶ | To assess robustness, block-bootstrap confidence intervals | App E | MOVE_TO_APPENDIX |
| 53 | 794–801 Finding R1 — MSR(LW) vs MSR(sample) | Finding 2 tested via Memmel; z=1.13, p=0.259 | App E | MOVE_TO_APPENDIX |
| 54 | 803–818 Finding R2 — VMP sign-test | Finding 6 most powerfully defended by sign-test | App E | MOVE_TO_APPENDIX |
| 55 | 819 Figure: bootstrap_sharpe_cis.png | Block-bootstrap 95% CIs for top 10 strategies | App E | MOVE_TO_APPENDIX |
| 56 | 821–828 Finding R3 — SWITCH(v2a) significant at 5% | Finding 5 SWITCH(v2a) improvement z=2.05, p=0.040 | App E | MOVE_TO_APPENDIX (in-text summary stays in §5) |
| 57 | 830–839 Finding R4 — HRP Memmel test | Finding 3 HRP near-invariance tested via Memmel | App E | MOVE_TO_APPENDIX (in-text summary stays in §4) |
| 58 | 847–856 §6 OOS > OOS Methodology ¶ | Train/test split imposes strict temporal boundary | §2 | RE_THREAD (belongs with harness description) |
| 59 | 858–872 §6 OOS > SWITCH(v2a) Re-Derivation ¶ | v2a rule constructed from training-period regime-conditional | §5 | RE_THREAD |
| 60 | 874–890 §6 OOS > Test-Period Leaderboard table | Top 5 by annualized Sharpe 2023–2026 held-out | §2 + App C | COMPRESS + RE_THREAD (brief table note in §2; full table to App C) |
| 61 | 892–914 §6 OOS > OOS Survival of Key Findings | VMP universal lift / MSR Michaud OOS replication | §3/§4/§5/§6 | RE_THREAD (dissolve into each thematic section) |
| 62 | 916–929 §6 OOS > Findings That Require Caveat OOS | Regime switching improvement lower power on 3.3 yr | §5 + §6 | RE_THREAD |
| 63 | 935–958 §7 Discussion > Volatility Management Meta-Overlay | Most striking empirical result is universality of VMP | §3 | RE_THREAD (primary §3 closing argument) |
| 64 | 960–977 §7 Discussion > Shrinkage vs. Structure | Covariance estimation matters enormously for MV strategies | §4 | RE_THREAD (primary §4 synthesis) |
| 65 | 979–1002 §7 Discussion > TC as Implementability Filter | Cost-sensitivity analysis functions as implementability filter | §6 | RE_THREAD |
| 66 | 1004–1055 §7 Discussion > Sample-Period Sensitivity | Sub-period analysis reveals within-strategy variation | §6 (narrative) + App G (Table 2) | SPLIT |
| 67 | 1061–1074 §8 RL > Motivation | Preceding paradigms share common structure: estimate | §2 | COMPRESS (1 ¶ contextualising RL in horse race) |
| 68 | 1078–1092 §8 RL > Problem Formulation | Cast allocation as offline batch Markov decision process | CUT | CUT (detail outside paper-1 classical scope) |
| 69 | 1095–1107 §8 RL > Algorithms | Two on-policy policy-gradient methods are evaluated | CUT | CUT |
| 70 | 1109–1127 §8 RL > Results (table + equity curve figs) | All three RL configurations land within 0.0012 Sharpe | §2 | COMPRESS (RL row kept in horse-race table; equity-curve figs CUT) |
| 71 | 1129–1150 §8 RL > Static Collapse + 1/N optimum | Convergence reflects common endpoint; OOS turnover negligible | CUT | CUT (detail; summary absorbed into §2 RL paragraph) |
| 72 | 1152–1167 §8 RL > Relation to Prior Paradigms | Conclusion corroborates direct-weight DL track | §7 | COMPRESS (2 sentences in Conclusion) |
| 73 | 1169–1188 §8 RL > Limitations and Conclusion | Result is conditional on 17-feature state representation | §7 | COMPRESS (1 sentence on RL scope limit) |
| 74 | 1194–1210 §9 Conclusion ¶1 | This study evaluated 62 portfolio allocation strategies | §7 | REWRITE (for new section structure) |
| 75 | 1211 §9 Conclusion — temporal-caveat sentence | Most forceful caveat is temporal: within-strategy sub-period | §7 | KEEP_AS_IS |
| 76 | 1213–1224 §9 Conclusion — Limitations ¶ | All results are for specific 29-ticker universe | §7 | KEEP_AS_IS |
| 77 | 1226–1242 §9 Conclusion — Long-short extensions ¶ | Three long-short variants added to quantify constraint gap | §6 | RE_THREAD (belongs with L/S discussion in §6) |
| 78 | 1244–1256 §9 Conclusion — Future work ¶ | Harness architecture accommodates several natural extensions | §7 | KEEP_AS_IS |
| 79 | 1259–1262 References section | Standard bibliography block | (unchanged) | KEEP_AS_IS |
| 80 | 1264–1353 List of Acronyms | BL, EW, FF3, GMV, HRP, MDP, MSR … | (unchanged) | KEEP_AS_IS |

---

## Paragraph count per destination

| Destination | Row numbers | Count |
|-------------|-------------|-------|
| §1 Intro | 1–8 | 8 |
| §2 Data/methods/horse race | 10(part), 11, 12, 20, 22, 23, 26, 27, 44, 45, 47, 58, 60(part), 67, 70 | 15 |
| §3 VMP universal | 18, 41, 43, 48(part), 63 | 5 |
| §4 Estimator choice | 36, 37, 38, 46, 48(part not in §3), 64 | 6 |
| §5 Regimes add value | 19(part), 25, 39, 40, 48(part), 56(summary), 59, 61(part), 62 | 8 |
| §6 Costs/survival | 31, 32, 33, 49, 50, 61(part), 62(part), 65, 66(part), 72, 77 | 10 |
| §7 Conclusion | 72, 73, 74, 75, 76, 78 | 6 |
| App A | 9, 10(part), 51 | 3 |
| App B | 13, 14, 15, 16, 17 | 5 |
| App C | 21, 28, 42, 60(part) | 4 |
| App D | 19(part) | 1 |
| App E | 52, 53, 54, 55, 56, 57 | 6 |
| App F | 29, 30, 34, 35 | 4 |
| App G | 66(part) — Table 2 | 1 |
| App H | *(none — new content required)* | 0 |
| CUT | 42(main text), 68, 69, 71 | 4 |

**Total source blocks accounted for:** 80
**Blocks with split destinations (count in two sections):** 10 (rows 10, 19, 48, 60, 61, 62, 66, 70, 72, 77)
**Net new content required:** App H (reproducibility guide, ~1 page)

---

## Open questions

1. **RL section scope** (rows 67–73). The six-section structure has no dedicated RL section. The current content — 7 sub-sections, ~130 lines — does not map cleanly anywhere. Proposed treatment: one compressed paragraph in §2 contextualizing RL in the horse race, RL performance rows kept in App C, and 2 sentences on the static-collapse finding in §7. Confirm whether this level of compression is acceptable or whether a new App I (RL methods) should be added.

2. **Findings 8, 9, 11 marginal content** (rows 44, 45, 47). These three findings (TSMOM weakness, BL-Mom return leadership, low-vol anomaly unleveraged) address individual strategies rather than cross-cutting themes. They don't map to any named section cleanly; all are marked COMPRESS with absorption into §2 horse-race prose. Confirm: are any of these section-worthy, or should all three be compressed to a single paragraph in §2?

3. **Finding 6.5 duplication** (row 42). The VMP(GMV(sample)) artifact is already covered by Finding 1 (row 36) and in the Rankings section (row 26). Finding 6.5 as a standalone block is pure duplication. Proposed action: CUT from §3/§4 main text, retain only as a one-line footnote in App C. Confirm.

4. **OOS section dissolution** (rows 58–62). The entire current §6 OOS Validation section (86 lines) is proposed for dissolution into the four thematic sections. This means no standalone OOS section in the new paper. If reviewers expect a dedicated OOS section, reconsider; otherwise keep as distributed OOS evidence woven into each claim.

5. **App H reproducibility** is referenced in the new structure but has no source content in `results.md`. It needs to be written from scratch. Candidate content: pipeline commands (from CLAUDE.md), data/published artifact hashes, expected output values. Confirm scope.

6. **Figure inventory for new structure**. The current paper has 11 figures. The proposed new structure will likely retain 6–7 in the main body and move 4–5 to appendices:
   - Keep in main: cumulative_wealth (§2), underwater_drawdown (§2), regime_conditional_heatmap (§5), bootstrap_sharpe_cis (App E or §3), rolling_sharpe_small_multiples (§6), calendar_returns_heatmap (App G or §6).
   - Move/drop: strategy_correlation_matrix → App E; stratified_vs_flat_costs → App F; vmp_exposure_mechanism → §3 or App F; equity_curves_rl/ppo → CUT if RL is compressed.
   Confirm figure placement before rewriting caption cross-references.

7. **SWITCH(v2a) train-only vs. full-sample Sharpe** (rows 40, 59). The current paper cites Sharpe=1.514 (full sample) and 1.608 for VMP(SWITCH(v2a)). The CLAUDE.md canonical value for VMP(SWITCH(v2a)) is **1.608 (train-only, lookahead-free)**. Check whether §5 should report only the train-only 1.608 or both. Currently results.md says 1.608 in the stacking discussion (line 717) — consistent. Flag for proofreading pass.
