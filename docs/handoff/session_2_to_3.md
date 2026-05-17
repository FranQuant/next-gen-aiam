# Session 2 → Session 3 Handoff

## State at handoff (commit edf88c9)

Sessions 1 + 1.5B + 2 complete:
- Paper PDF (38 pages) at `docs/results.pdf`, 62-strategy comparative harness, commit `60b6830`
- Notebooks: `01_paper_reproduction.ipynb` + `02_practitioner_analytics.ipynb` + `03_ml_strategies.ipynb`
- Published dataset at `data/published/`, 5 artifacts + README
- Test suite: 124 tests, all passing
- ML scaffolding: `src/aiam/ml/`, `src/aiam/features/asset_class.py`, `src/aiam/strategy/ml_strategies.py`
- ML notebook 03 fully populated: 17 sections, 9+ figures, 28-strategy extended comparison

## Session 2 final headline numbers (test period 2023–2026)

Top 6 from the 28-strategy extended comparison sorted by Sharpe:

| Strategy | Family | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| MSR(Ensemble_μ̂) | ML (ensemble) | 0.166 | 0.060 | **2.579** | -0.059 |
| VMP(MDP(LW)) | Classical | 0.149 | 0.058 | 2.422 | -0.048 |
| MSR(RF_μ̂) | ML (single-fit) | 0.209 | 0.081 | 2.394 | -0.068 |
| SignalTilt(XGB) | ML (single-fit) | 0.707 | 0.245 | 2.304 | -0.226 |
| VMP(SignalTilt(XGB)) | ML + VMP | 0.720 | 0.250 | 2.292 | -0.205 |
| MSR(Lasso_μ̂) | ML (single-fit) | 0.218 | 0.089 | 2.272 | -0.116 |

## Five empirical findings (Session 2 contributions)

1. **MSR(ML μ̂) is the value extractor.** Approach A (MSR with ML-predicted μ̂) outperforms Approach B (SignalTilt wrapping) for Lasso and RF. Best single-fit: MSR(RF_μ̂) 2.394 vs SignalTilt(RF) 2.252; MSR(Lasso_μ̂) 2.272 vs SignalTilt(Lasso) 2.140. Exception: XGB where SignalTilt(XGB) 2.304 edges MSR(XGB_μ̂) 2.180 — the optimizer amplifies XGB's noisier predictions. MSR wrapping works best when predictions are well-calibrated; check feature-importance interpretation before assuming MSR always wins.

2. **Ensemble is the headline result.** MSR(Ensemble_μ̂) Sharpe 2.579 tops the full 28-strategy comparison, beating the next-best classical strategy VMP(MDP(LW)) at 2.422. Simple equal-weighted average of Lasso + RF + XGB μ̂ fed into MSR; no retraining required. Ensemble reduces idiosyncratic model noise and stabilizes optimizer inputs — the value comes from diversifying prediction error, not from a superior individual model.

3. **VMP overlay does not help ML strategies.** VMP-wrapped ML variants consistently rank below un-wrapped equivalents: VMP(SignalTilt(XGB)) 2.292 < SignalTilt(XGB) 2.304; VMP(MSR(RF_μ̂)) 2.177 < MSR(RF_μ̂) 2.394. VMP benefits classical strategies whose returns are noisier; ML strategies already apply implicit signal normalization, so the additional vol-scaling layer adds friction without informational content.

4. **HP sensitivity: sharp val/OOS disagreement.** XGB default (depth=6) shows catastrophic validation overfitting — `best_iteration=0`, val_IC = 0.013 — yet ranks near the top OOS at Sharpe 2.304. The 2019–2022 validation window (COVID + rate-hike cycle) is poorly representative of 2023–2026 (post-shock environment). Val_IC ranking (Lasso 0.112 > RF 0.038 > XGB 0.013 at defaults) inverts OOS Sharpe ranking for SignalTilt (XGB 2.304 > RF 2.252 > Lasso 2.140). HP validation on this regime-shifted universe must be treated skeptically.

5. **Walk-forward refit underperforms single-fit.** All 9 walk-forward strategies (annual refit, trailing 10-year window, default and val-optimal HPs) rank below their single-fit counterparts. Best WF Sharpe: SignalTilt(WF-lasso-default) 2.033 vs single-fit SignalTilt(Lasso) 2.140. Single-fit models trained through 2020 encode the COVID + early rate-shock regime in feature weights; the 2023–2026 test period remains a post-shock environment where that memory is an asset. Default HPs beat val-optimal for Lasso in WF (validation bias confirmed) but not for RF (+0.134) or XGB (+0.083).

## Methodological lessons for Session 3 (DL)

1. **High bar to clear.** Best ML Sharpe is 2.579 (MSR(Ensemble_μ̂)). DL strategies need to add interpretable value beyond marginal Sharpe improvements. Natural DL angle: capture temporal patterns the tree-based models miss (sequence dependencies, regime transitions, cross-asset attention).

2. **Single-fit methodology stays for the first DL pass.** Walk-forward underperforms single-fit on this universe (Finding #5). Don't repeat walk-forward in Session 3a; document the choice and revisit only if Session 3b results motivate it.

3. **MSR(ML μ̂) wrapping is the value extractor.** Approach A (MSR with ML predictions) outperforms Approach B (SignalTilt wrapping) for most single models, and MSR(Ensemble) is the headline result. Session 3 should test BOTH wrappers for DL outputs — same comparison structure as Session 2.

4. **VMP overlay doesn't help ML strategies.** Skip VMP-wrapping for DL in Session 3 unless a specific reason emerges. Avoids wasted comparison rows.

5. **Ensemble matters.** Pure single-model strategies underperform their ensembles in the MSR wrapper. Plan to ensemble DL models (MLP + LSTM + Transformer predictions averaged into MSR) as the headline DL strategy.

## Open methodological questions

- How does DL handle the regime sensitivity revealed by §16 HP sensitivity and §17 walk-forward? Need different validation strategy.
- Should DL training use the same 17-feature panel or also use raw sequence input? LSTM/Transformer should use sequence input; MLP can use the existing panel.
- Permutation importance doesn't work for sequence models. What's the right interpretability tool? (Gradient-based attribution, attention weights for Transformer.)

## Specific DL architectures aligned with this universe

| Model | Input | Output | Where it fits |
|---|---|---|---|
| MLP | 17-feature cross-section per (Date, Asset) | 21-day forward return prediction | Direct successor to RF/XGB; uses same feature panel |
| LSTM | Per-asset return sequence (lookback 252d) | 21-day forward return prediction | Captures temporal patterns |
| Transformer (small) | Cross-asset return matrix (lookback 252d × 29 assets) | 21-day forward return per asset | Cross-sectional attention; can learn asset interactions |

Recommended scaffolding pattern: parallel to Session 2a. Create `src/aiam/dl/` with `workflow.py` (PyTorch training helpers) and `src/aiam/strategy/dl_strategies.py` with `MLPSignalStrategy`, `LSTMSignalStrategy`, `TransformerSignalStrategy`. Mirror the `PointInTimeStrategy` ABC and the SignalTilt/MSR wrapping pattern from `src/aiam/strategy/ml_strategies.py`.

## How to start Session 3

1. Read this handoff doc + CLAUDE.md
2. Verify environment: `python -c "import torch; print(torch.__version__)"` — install if needed (PyTorch + torchvision via `pip install torch torchvision`)
3. Follow the same Session 2 structure: 3a (scaffolding) → 3b (experiment) → 3c-* (polish/extensions if needed)
4. First prompt should match Session 2a's pattern: scaffolding pass building `src/aiam/dl/`, tests, no notebook execution
5. Use the Claude Code prompt preamble from CLAUDE.md: ship without checking in, single atomic commit, self-verification block, final report with actual numbers
