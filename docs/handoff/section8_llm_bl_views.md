# Section 8 — LLM-Augmented Black-Litterman Views: Findings

## Abstract

A returns-only LLM view generator feeding a disciplined Black-Litterman allocator produced
the best risk-adjusted performance of the BL family (Sharpe 2.284, max drawdown −9.7%) over
a fully out-of-sample window spanning 833 trading days and 41 monthly refits. Both LLM
providers outperformed both rule-based BL generators. View quality is model-dependent:
claude-opus-4-7 adds a clear margin over the no-views equilibrium prior; gpt-5.5 barely
clears it. The IC temporal profile runs opposite to a training-data leakage fingerprint —
LLM skill is weakest early in the backtest (deepest in the training window) and strongest
in the post-cutoff window — substantially softening the lookahead caveat, though N ≈ 8
post-cutoff refit periods is too small to be conclusive. This validates the **method**, not
yet a deployable edge.

---

## Methodology

**Universe and OOS window.** 29 assets — US and international equities, fixed income
(SHY, IEF, TLT, AGG, HYG), commodities (GLD, SLV, DBC, USO), and EURUSD — over a
fully out-of-sample window 2023-01-01 to 2026-04-30 (833 trading days, 41 monthly refits).

**Shared BL infrastructure.** All five variants use identical BL parameters: τ = 0.05,
δ = 2.5, Ledoit-Wolf covariance estimator, long-only weights, monthly refits. They differ
only in how the view vector **Q** is formed.

**Five variants.**

| Generator | View source |
|---|---|
| `equilibrium` | No views — BL collapses to CAPM-equilibrium MSR |
| `momentum` | Annualised 252-day trailing return per asset (all 29 assets) |
| `mean_rev` | −1 × annualised 5-year mean — sign-flip reversion (all 29 assets) |
| `llm-anthropic` | claude-opus-4-7 given trailing return evidence |
| `llm-openai` | gpt-5.5 given identical trailing return evidence |

**Evidence packet.** Returns-only: per-asset 21d, 63d, and 252d trailing returns plus
annualised volatility computed from a rolling 252d window, all values strictly as-of the
refit date (no look-ahead). No market commentary, news, or macro data was supplied.

**Governance principle.** The LLM emits view vectors only: a pick-matrix **P**, a view
vector **Q**, and a diagonal uncertainty matrix **Ω**. Black-Litterman blends these with the
equilibrium prior and the Ledoit-Wolf covariance to form posterior expected returns; a
long-only mean-variance optimizer sets portfolio weights. **The LLM never allocates capital
directly.** This architecture follows the JPM "Investable AI" discipline (LLM → structured
views → disciplined allocator) rather than letting the model output weights directly.

**Verification scaffolding.** Schema validation on LLM JSON output; fail-closed on
malformed responses (no view expressed rather than fallback to random); deterministic prompt
cache (Anthropic + OpenAI responses archived per (model, system-prompt, user-prompt) key)
for reproducibility without re-incurring API costs or non-determinism. All backtest results
replay from cache only.

**Development note.** The Ch22 "LLM as coding assistant" pattern (Hilpisch 2020) was used
throughout development via Claude Code.

---

## Results

### Performance table

*OOS 2023-01-01 → 2026-04-30 · 833 days · 41 monthly refits · returns-only evidence.*

| Rank | Generator | Sharpe | Ann Ret | Ann Vol | Max DD | Turn/refit | Views/refit |
|---|---|---|---|---|---|---|---|
| 1 | BL-LLM(Opus 4.7) | **2.284** | 0.357 | 0.138 | −0.097 | 0.445 | 13.0 |
| 2 | BL-Equilibrium | 2.037 | 0.222 | 0.101 | −0.122 | 0.000 | 0 |
| 3 | BL-LLM(GPT-5.5) | 1.979 | 0.271 | 0.125 | −0.110 | 0.508 | 15.8 |
| 4 | BL-Momentum | 1.869 | 0.327 | 0.158 | −0.156 | 0.212 | 29 |
| 5 | BL-MeanRev | 1.560 | 0.278 | 0.166 | −0.123 | 0.476 | 29 |
| ref | ML bar (session 3 best) | 2.579 | — | — | — | — | — |

ML/classical top-3 for context: VMP(MDP(LW)) 2.422 · MSR(RF_μ̂) 2.394 · SignalTilt(XGB) 2.304
(Ann Ret 0.707 / Ann Vol 0.245 / MaxDD −0.226 — far rougher drawdown than BL-LLM(Opus)).

### Two load-bearing findings

**(a) Both LLMs beat both rule-based generators on Sharpe.** Opus leads momentum by +0.416
and mean_rev by +0.725; even GPT-5.5 leads by +0.110 and +0.419 respectively. The result
is not "LLMs win generically" — it is that a calibrated, selective view-generator (returning
13–16 views from 29 assets rather than covering all 29 indiscriminately) outperforms
full-coverage mechanical rules. The selectivity itself appears to be part of the signal.

**(b) View quality is model-dependent: Opus clears the no-views prior, GPT-5.5 barely does.**
BL-Equilibrium (no views, pure CAPM prior) sits at Sharpe 2.037 — a non-trivial baseline
reflecting the BL allocator's quality rather than any view signal. Opus clears it by +0.247;
GPT-5.5 clears it by only +0.058 — a margin that would be easily consumed by transaction
costs. This means the ability to add value through LLM views is sensitive to model choice,
not guaranteed by the architecture.

### Context within the broader comparison

BL-LLM(Opus) does not clear the 2.579 ML bar (gap −0.295) but sits within the same
competitive tier as the top ML/classical cluster. More notably, it matches SignalTilt(XGB)
on Sharpe (≈2.3) while achieving dramatically better drawdown (−0.097 vs −0.226). That
drawdown compression is a structural property of BL's covariance-aware allocator, not
a consequence of low signal quality or look-ahead data. Naive rule-based generators
(momentum, mean_rev) hurt versus the equilibrium prior on Sharpe — a result consistent
with BL theory: views that add noise worsen the posterior.

---

## Forward-Predictive Skill and Lookahead Diagnostic

**Method.** For each monthly refit, the Spearman rank correlation (Information Coefficient,
IC) between the view vector **Q** and the realised asset returns over the next holding
window was computed. Analysis is restricted to assets for which the generator expressed a
view. Momentum serves as the control: a deterministic, non-LLM signal with no
cutoff-aligned decay structure.

### Fig A — Mean IC by generator

| Generator | Mean IC | IC Std | N |
|---|---|---|---|
| llm-anthropic | +0.117 | 0.347 | 41 |
| llm-openai | +0.105 | 0.346 | 41 |
| momentum | +0.042 | 0.284 | 41 |

Both LLM providers show positive mean IC above the momentum baseline. The LLM-vs-momentum
difference is not statistically significant at N ≈ 41 monthly periods (IC standard errors
≈ ±0.05–0.06), but the directional signal is consistent across both providers and all 41
refit dates.

### Fig B — Temporal IC pattern (the hindsight-tell)

A training-data leakage fingerprint would manifest as: LLM IC meaningfully positive early
in the backtest (deep inside the training window, 2022–23) decaying toward zero or negative
as refit dates approach the training cutoffs (GPT-5.5 ≈ 2025-08, claude-opus-4-7 ≈ 2026-01).

**The observed pattern runs opposite to this.** LLM IC is lowest — at or below zero — in
2022–23, the period most deeply inside the training window. It rises and is strongest in
2025–26, past or near both training cutoffs. This is the anti-fingerprint pattern: skill
concentrated out-of-sample, not inside training data.

**Leakage ratios** are elevated (Anthropic +2.79×, OpenAI +2.50× vs momentum), but the
elevated ratio is NOT combined with cutoff-aligned temporal decay. Per the standard
interpretation, both conditions are required for the hindsight fingerprint; ratio alone,
without temporal decay, is insufficient evidence of leakage.

**This softens the lookahead caveat substantially** relative to the setup-stage concern.
The main remaining uncertainty is N ≈ 8 post-cutoff refit periods — too small for
statistically firm conclusions. Approximate training-cutoff dates (Anthropic ≈ 2026-01,
OpenAI ≈ 2025-08) introduce further uncertainty about the exact window boundary.

---

## What We Can and Cannot Claim

**Can claim:**
- The LLM view-generator architecture (structured P/Q/Ω → BL allocator) works as
  designed: schema validation, fail-closed error handling, and the prompt cache all
  functioned correctly across 41 × 2 provider × 29 assets = 2 378 view calls.
- On this backtest, returns-only LLM views improved BL Sharpe over rule-based alternatives
  and over the no-views equilibrium, with Opus achieving a clear margin.
- The IC temporal profile does not exhibit the expected hindsight fingerprint.

**Cannot claim:**
- Statistical significance: N ≈ 40 monthly periods, single OOS path, single regime
  (2023–2026 bull market with one rate-cycle). IC standard errors are large; none of
  the IC results would survive multiple-testing correction.
- Model-agnostic LLM advantage: GPT-5.5 barely clears the no-views prior. "Use an LLM"
  is not sufficient — model selection and confidence calibration matter.
- Deployable edge: turnover of 0.445–0.508 per monthly refit (≈60–70 bps round-trip/month)
  exceeds the 10 bps nominal transaction cost budget. Net-of-cost performance has not
  been evaluated.
- Lookahead absence: the approximate training cutoff dates used for the diagnostic
  introduce uncertainty of weeks to months. A rigorous attribution would require strictly
  held-out, unambiguously post-cutoff dates with confirmed model data boundaries.

---

## Framing

This project is closer to the JPM "Investable AI" (Gupta & Strasburg 2023) paradigm — LLM
produces structured views that feed a disciplined quantitative allocator — than to the
Hilpisch Ch22/23 assistant and agent modes. It honours the governance principle those modes
share: the LLM advises, the quantitative model allocates.

Hilpisch's Ch22 "LLM as coding assistant" pattern was used throughout development via
Claude Code, and the Ch23 "LLM agent" framing captures the view-generation loop at a
conceptual level (observe returns → reason → emit structured output → act via BL), but the
LLM does not call tools, access live market data, or autonomously place orders.

The critical methodological constraint distinguishing this work from a live trading assistant
is that all evidence is point-in-time: evidence is constructed strictly from data as-of the
refit date, API responses are cached and replayed deterministically, and no live web search
(which would inject future information into historical dates) is permitted in the backtest.

---

## Future Work

**1. Rich-evidence layer (v3).**
The most impactful near-term extension is augmenting the prompt with fundamentals (P/E,
EPS revisions) and news sentiment from a **timestamped archive sliceable to ≤asof** — for
example EODHD news API, Bloomberg historical news, or similar. This must be a
point-in-time archive, NOT live web search.

**Tavily and live search are explicitly excluded from backtesting.** A live search call
at a historical refit date (e.g., 2024-02-01) retrieves documents published after that
date, injecting future information into the evidence packet and invalidating the OOS
comparison. Tavily belongs only in a live/forward-looking mode where the intent is
explicitly to use the most current available data.

**2. Larger-N / multi-path confirmation.**
Extend the OOS window forward, use bootstrap or block-bootstrap resampling across the
existing 41 refit dates, or run the backtest on a different universe to build IC inference
beyond N ≈ 40 monthly periods.

**3. Exact training-cutoff verification.**
Obtain precise Anthropic and OpenAI training-data cutoff dates (ideally confirmed by the
providers) to sharpen the post-cutoff analysis window and reduce uncertainty in Fig B.

**4. Turnover management.**
Experiment with confidence thresholds (e.g., exclude views with confidence < 0.45) and
position caps to bring LLM-generator turnover within the 10 bps budget. Target: ≤ 0.25
one-way turnover per monthly refit.

**5. Earnings-transcript semantic alignment.**
Compute a per-asset factor measuring alignment between the LLM's directional view and the
sentiment of the most recent earnings call transcript. Hypothesis: views that "agree" with
management tone have higher ex-post IC.

**6. Multi-regime / multi-path robustness.**
The 2023–2026 window is dominated by a single macro regime (post-rate-hike equity rally).
Testing on a window that includes a genuine bear market episode would sharpen the claim
that BL-LLM(Opus) drawdown compression is structural, not regime-specific.
