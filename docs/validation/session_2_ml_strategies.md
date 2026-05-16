# Session 2 — ML Strategies Validation

## Methodology

Three ML strategies (Lasso, Random Forest, XGBoost) are trained once on the full training window (~2003–2019, after 1-year warmup) and evaluated out-of-sample on the test period (2023-01-01 to 2026-04-30, ~3.3 years). The target is the 21-day forward cumulative return. Features: 10 numeric (momentum, volatility, RSI, ATR ratio, Bollinger pct, overnight gap, volume z-score) + 7 asset-class one-hot dummies. The panel has 17 features × (date × 29 assets).

Two portfolio construction approaches: **Approach B** (core) wraps ML cross-sectional z-scores in a SignalTilt overlay (EW base + tilt_strength=0.5). **Approach A** (extension) feeds ML μ̂ directly into MSR (long-only clip + renormalize, Ledoit-Wolf Σ, lookback=504 days).

Train/validation/test split (paper-locked): `TRAIN_END=2022-12-31`, `TEST_START=2023-01-01`, validation = last 15% of pre-test window.

**Comparison note:** ML strategies are evaluated on true OOS data; paper baselines use full-sample Sharpes. This is an intentional apples-to-pears comparison by design.

## Headline Results (test period 2023–2026, sorted by Sharpe)

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| VMP(MDP(LW)) | 0.149 | 0.058 | 2.422 | -0.048 |
| MSR(RF_μ̂) | 0.209 | 0.081 | 2.394 | -0.068 |
| SignalTilt(XGB) | 0.707 | 0.245 | 2.304 | -0.226 |
| MSR(Lasso_μ̂) | 0.218 | 0.089 | 2.272 | -0.116 |
| SignalTilt(RF) | 0.619 | 0.225 | 2.252 | -0.326 |
| MSR(XGB_μ̂) | 0.077 | 0.034 | 2.180 | -0.026 |
| SignalTilt(Lasso) | 0.467 | 0.187 | 2.140 | -0.228 |
| SWITCH(v2a) | 0.190 | 0.084 | 2.114 | -0.057 |
| EW | 0.222 | 0.101 | 2.037 | -0.122 |
| MSR(LW) | 0.272 | 0.133 | 1.875 | -0.141 |
| SignalTilt(mom_252) | 0.501 | 0.257 | 1.711 | -0.209 |

## Permutation Importance Findings (RF, validation set)

| Feature | Mean Importance |
|---|---|
| vol_252 | 0.01630 |
| vol_60 | 0.01301 |
| mom_63 | 0.01088 |
| atr_14_ratio | 0.00826 |
| mom_252 | 0.00756 |

Momentum (252-day) and volatility features dominate. Asset-class dummies carry moderate importance, confirming within-class heterogeneity is informative. Short-horizon momentum (mom_21) ranks near the bottom, matching its near-zero IC from Session 1.5B.

## Limitations

1. **Single fit:** Model trained 2003–2019, never retrained. A rolling-refit strategy (Session 2c) would better adapt to regime shifts.
2. **Short test window:** 3.3 years yields Sharpe standard errors of ~0.3–0.5; cross-strategy differences may not be statistically significant.
3. **No hyperparameter tuning:** Conservative defaults throughout. Lasso α=1e-4, RF max_depth=8, XGB lr=0.05 set without grid search.
4. **MSR instability:** Approach A amplifies estimation error in μ̂ (Michaud 1989); the long-only clip is a heuristic fix, not a solution.

## Implications

**Session 3 (DL):** MLP/LSTM/Transformer architectures should aim to beat the best Approach B Sharpe here. The weak single-fit finding suggests time-varying architectures (LSTM with rolling context, Transformer attention) may be needed.

**Session 2c (optional rolling refit):** A walk-forward refit (e.g., annual refit on expanding window) would turn the single-fit experiment into a production-grade backtesting framework and likely close the gap vs. classical baselines.

## VMP Overlay + Ensemble Extensions (Session 2c-A)

Two additive enhancements applied to existing single-fit ML strategies:

- **VMP overlay** wraps each of the 6 ML strategies with the same volatility-managed portfolio scaling used in the paper's 31 VMP variants. Target = each strategy's long-run realized vol; lookback 21d, clip (0.25, 1.5).
- **Ensemble predictions** = equal-weighted average of Lasso/RF/XGBoost μ̂. Wired through both SignalTilt and MSR portfolio construction.

Extended comparison: 11 → 19 strategies.

| Strategy | Family | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| MSR(Ensemble_μ̂) | ML (ensemble) | 0.166 | 0.060 | 2.579 | -0.059 |
| VMP(MDP(LW)) | Classical | 0.149 | 0.058 | 2.422 | -0.048 |
| MSR(RF_μ̂) | ML (single-fit) | 0.209 | 0.081 | 2.394 | -0.068 |
| SignalTilt(XGB) | ML (single-fit) | 0.707 | 0.245 | 2.304 | -0.226 |
| VMP(SignalTilt(XGB)) | ML + VMP | 0.720 | 0.250 | 2.292 | -0.205 |

**Headline observations:**

- **MSR(Ensemble_μ̂) tops the leaderboard at Sharpe 2.579**, narrowly beating VMP(MDP(LW)) at 2.422. The ensemble averages out model-specific noise in μ̂ estimates, improving signal quality without any additional training.
- **VMP-wrapping ML strategies provides mixed results.** VMP overlay mildly hurt most ML variants (e.g., VMP(SignalTilt(XGB)) = 2.292 vs SignalTilt(XGB) = 2.304; VMP(MSR(RF_μ̂)) = 2.177 vs MSR(RF_μ̂) = 2.394). The underlying ML strategies already carry high vol; the VMP clipping does not consistently add value here unlike on the classical strategies.
- **SignalTilt(Ensemble) = 2.183 underperforms individual SignalTilt(XGB) = 2.304**, suggesting that averaging the z-scored tilt signals dilutes XGBoost's edge rather than compounding it. The ensemble benefit is concentrated in the MSR (mean-estimate) path.
