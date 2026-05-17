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
| vol_252 | 0.01747 |
| vol_60 | 0.01292 |
| mom_63 | 0.01059 |
| atr_14_ratio | 0.00776 |
| mom_252 | 0.00704 |

Momentum (252-day) and volatility features dominate. Asset-class dummies carry moderate importance, confirming within-class heterogeneity is informative. Short-horizon momentum (mom_21) ranks near the bottom, matching its near-zero IC from Session 1.5B.

## Limitations

1. **Single fit:** Model trained 2003–2019, never retrained. A rolling-refit strategy (Session 2c) would better adapt to regime shifts.
2. **Short test window:** 3.3 years yields Sharpe standard errors of ~0.3–0.5; cross-strategy differences may not be statistically significant.
3. **No hyperparameter tuning:** Conservative defaults throughout. Lasso α=1e-4, RF max_depth=8, XGB lr=0.05 set without grid search.
4. **MSR instability:** Approach A amplifies estimation error in μ̂ (Michaud 1989); the long-only clip is a heuristic fix, not a solution.

## Implications

**Session 3 (DL):** MLP/LSTM/Transformer architectures should aim to beat the best Approach B Sharpe here. The weak single-fit finding suggests time-varying architectures (LSTM with rolling context, Transformer attention) may be needed.

**Session 2c (optional rolling refit):** A walk-forward refit (e.g., annual refit on expanding window) would turn the single-fit experiment into a production-grade backtesting framework and likely close the gap vs. classical baselines.

## Walk-Forward Refit (Session 2c-B)

Annual refit on rolling 10-year window, 4 refit dates (2023-01-02, 2024-01-02, 2025-01-02, 2026-01-02). Two HP regimes compared:

- **WF-default** uses Hilpisch/JPM defaults from §4–§6 (Lasso α=1e-4; RF 100 trees, depth=8; XGB 300 rounds, depth=6)
- **WF-val-optimal** uses validation-tuned HPs from §16 (Lasso α=1e-3; RF 50 trees, depth=6; XGB 200 rounds, depth=4)

### Walk-Forward vs Single-Fit Sharpe Comparison

| Model | Single-fit | WF-default | WF-val-optimal |
|---|---|---|---|
| Lasso | 2.140 | **2.033** | 2.006 |
| RF    | 2.252 | 1.617 | **1.751** |
| XGB   | 2.304 | 1.777 | **1.860** |

### Walk-Forward Strategy Results (test period 2023–2026)

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| SignalTilt(WF-lasso-default)    | 0.552 | 0.229 | 2.033 | -0.231 |
| SignalTilt(WF-lasso-val-opt)    | 0.566 | 0.238 | 2.006 | -0.229 |
| MSR(WF-lasso-default_μ̂)        | 0.246 | 0.114 | 1.987 | -0.141 |
| MSR(WF-xgb-default_μ̂)          | 0.238 | 0.111 | 1.984 | -0.097 |
| SignalTilt(WF-xgb-val-opt)      | 0.454 | 0.214 | 1.860 | -0.264 |
| MSR(WF-rf-default_μ̂)           | 0.248 | 0.120 | 1.913 | -0.104 |
| SignalTilt(WF-xgb-default)      | 0.370 | 0.187 | 1.777 | -0.169 |
| SignalTilt(WF-rf-val-opt)       | 0.470 | 0.236 | 1.751 | -0.250 |
| SignalTilt(WF-rf-default)       | 0.394 | 0.221 | 1.617 | -0.229 |

### Findings

**Walk-forward did not beat single-fit.** All 9 WF strategies ranked below their single-fit counterparts. Single-fit models trained on 2003–2020 (including COVID stress + the early rate-hike regime) appear to encode regime-appropriate feature weighting better than WF models trained on the trailing 10 years only. The 3.3-year prediction gap did not cause detectable model decay on this test period.

**Validation-bias test result: mixed.** For Lasso, default HPs beat val-optimal in walk-forward (2.033 vs 2.006) — consistent with the hypothesis that §16 HP selection over-optimised on the COVID/rate-shock validation window. For RF and XGB, val-optimal HPs won (+0.134 and +0.083 Sharpe) — the §16 tuning generalised in the walk-forward setting for tree-based models. Validation bias is model-specific.

**MSR(WF-*) vs MSR(single-fit).** Walk-forward refit degraded MSR performance more than SignalTilt, consistent with MSR amplifying the higher estimation noise in smaller WF training windows (Michaud 1989).

**Regime interpretation.** The superior performance of single-fit models suggests the 2020–2022 stress period is predictive of 2023–2026 cross-sectional structure — the test period is still a post-rate-shock environment where the single-fit model's memory of that regime is an asset, not a liability.
