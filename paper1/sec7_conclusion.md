# §7 Conclusion

## Three patterns

The 62-strategy horse race on a single 29-asset, 23.3-year universe produces three findings
that survive the full battery of checks in §§3–6.

**Volatility management is the universal lever.** The VMP overlay [@moreira2017volatility]
improves gross Sharpe over the corresponding base strategy for every one of 24 original base
configurations — a 24/24 sign-test result with $p \approx 6 \times 10^{-8}$ (§3.1). The
median lift is +0.194 Sharpe points; the minimum is +0.119. The result replicates on the
held-out 2023–2026 test period and is directionally stable within each sub-period bucket in
Table 2. It is the single empirical statement in the study that can be made without caveats
about timing, parameter choice, or regime sequence.

**Robust estimation matters where the optimizer overfits.** Ledoit-Wolf shrinkage adds +0.164
Sharpe to MSR by preventing concentration on whichever asset led the estimation window (§4.1).
HRP, which avoids matrix inversion entirely, is near-invariant to estimator choice — the same
regularisation operation that benefits MSR does nothing for HRP because HRP has no inversion
step to destabilise (§4.2). The same covariance matrix produces opposite outcomes because the
structural mechanism differs.

**Regime conditioning, derived honestly from training data, adds genuine value.** SWITCH(v2a)
— with routing derived entirely from 2003–2022 training-period regime-conditional Sharpe
analysis and never recalibrated on test data — outperforms the v1 rule by $\Delta = +0.434$
Sharpe (Memmel $z = 2.05$, $p = 0.040$), the strongest statistically significant single result
in the study (§5.2). The training-only derivation recovers the same R5→MSR(sample) assignment
as the full-sample analysis, confirming a persistent late-cycle structural feature rather than
a derivation-window artifact.

## The mechanism-versus-ranking distinction

Stacking all three refinements produces **VMP(SWITCH(v2a))** at full-sample Sharpe 1.608 — the
strongest classical configuration in this study. The correct reading of that number is
supplied by §6.3: within-strategy Sharpe variation across five-year calendar buckets
(MSR(LW) ranging 0.41–1.80) exceeds the full-sample cross-strategy spread (~0.50 Sharpe
points). The headline ranking table describes average performance over one particular 23.3-year
macro sequence, not a stable ordering of strategy quality. The three mechanisms persist across
sub-periods; the specific 1.608 configuration does not need to, and the honest evidence is
that it will not.

## Limitations

All results apply to a specific 29-ticker universe with US equity tilt; cross-sectional
strategies (FF3 factors, BL-Mom) may respond differently in pure small-cap or non-US settings.
Only monthly rebalancing is evaluated. The VMP overlay is treated as costless in the base
analysis; daily exposure scaling via futures overlays carries ~1–3 bps/day in practice. All
Sharpe ratios are computed at $r_f = 0$; at positive risk-free rates the relative ranking of
low-return strategies (GMV(sample), FF3-LowVol) would deteriorate further. The universe
excludes crypto entirely.

## Forward pointers

Two open questions follow directly. First: does learned complexity — ML signal extraction
(Lasso, Random Forest, XGBoost), deep sequence models (MLP, LSTM, Transformer), reinforcement
learning allocators, or LLM-generated Black-Litterman views — add Sharpe above the 1.608
classical ceiling on the same universe and harness? Paper 2 answers this directly, using the
identical 29-asset walk-forward harness and the same 2022-12-31 train/test split.

Second: can the research discipline demonstrated here — a deterministic multi-stage pipeline
with strict temporal barriers, reproducible published artifacts, and human-in-the-loop review
gates — be operationalized as a governed agentic research workflow? Paper 3 documents the
end-to-end build under a forbidden-capabilities substrate and asks whether the governance
constraints that make classical empirical research credible translate to an AI-assisted
research system.

Paper 1 ends here: classical methods, one harness, one verdict on the mechanisms.
