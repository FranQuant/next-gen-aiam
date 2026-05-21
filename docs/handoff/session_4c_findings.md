# Session 4c Findings — RL N=29 Walk-Forward Training (REINFORCE+Baseline)

## Status: Pending Colab Execution

Notebook `06a_rl_reinforce.ipynb` (formerly `04f_rl_n29_training.ipynb`) is written and smoke-tested locally.
Full results require Colab GPU (T4/L4 recommended). Fill in bracketed placeholders
`[TBD]` after the Colab run completes.

---

## Methodology Summary

### Algorithm

REINFORCE with learned value baseline (REINFORCE+baseline). The policy is a
`SimplexPolicy` — a per-asset MLP encoder with weights shared across all 29
assets, followed by a softmax to produce portfolio allocations on the simplex.
Architecture: Linear(21→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1) per
asset, with the current portfolio weight appended to each asset's feature vector
(21 = 20 lagged daily returns + 1 weight). Parameter count is independent of N,
so the same policy class scales from N=1 in the sanity notebook (04c) through
N=2/3 in the training notebook (04e) and N=29 here.

The value baseline is a two-layer MLP on the mean-pooled encoder hidden state. It
estimates V(s_t) and is used to reduce the variance of the policy gradient via
advantage estimates A_t = G_t − V(s_t), where G_t is the Monte-Carlo return-to-go
standardized within each episode.

### Environment

`PortfolioEnv` with 29 assets, `lookback=20` (rolling 20-day return windows as
features), transaction cost 10 bps per unit of turnover, and `lambda_risk` as
an ablated parameter. Reward: `gross_return − tc − lambda_risk × portfolio_vol`.
Portfolio vol is computed as `w ⊤ vol_proxy` where `vol_proxy` is the rolling
21-day per-asset standard deviation. The environment operates in discrete time
(one trading day per step); episodes run over the full training window without
artificial truncation.

### Walk-Forward Refit

Monthly refit cadence matching Sessions 1–3. For each refit date D:
- Training window: `[D − 24 months, D − 1 day]` (approximately 504 trading days)
- OOS window: `[D, next_refit_date − 1 day]`
- 10 independent seeds trained per refit date per config
- Ensemble prediction: simple average of 10 deterministic policies (`policy.act()`)
- Transaction cost applied at 10 bps per turnover unit during OOS evaluation

Total: 2 configs × 10 seeds × `[TBD]` refits = `[TBD]` policy fits.

The refit cadence is produced by `generate_refit_dates` from the DL walk-forward
harness (`aiam.dl.walkforward`), ensuring calendar-snapping to valid trading days
and exact alignment with Sessions 1–3 OOS windows.

### Configs

| Parameter | Config A | Config B |
|---|---|---|
| `lambda_risk` | 0.02 | 0.00 |
| `episodes` | 200 | 200 |
| `max_steps_per_episode` | None (full) | None (full) |
| `lr` | 1e-3 | 1e-3 |
| `gamma` | 0.95 | 0.95 |
| `entropy_coef` | 0.01 | 0.01 |
| `grad_clip` | 1.0 | 1.0 |
| `seeds` | 0–9 | 0–9 |

Config A replicates the Session 4b default. Config B tests whether removing the
risk penalty breaks the static-collapse pattern observed with 60-step truncation
at N=1/2/3 — the risk penalty was a natural pull toward equal-weight that could
theoretically dominate the signal in the policy gradient.

---

## Headline Result

**Status: `[STATIC_COLLAPSE_DETECTED / FEATURE_CONDITIONAL_LEARNING / PARTIAL_COLLAPSE — fill in after run]`**

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD | Rank |
|---|---|---|---|---|---|
| RL λ=0.02 ensemble | [TBD] | [TBD] | [TBD] | [TBD] | [TBD]/39 |
| RL λ=0.00 ensemble | [TBD] | [TBD] | [TBD] | [TBD] | [TBD]/39 |
| MSR(Ensemble_μ̂) (ML bar) | 0.166 | 0.060 | 2.579 | −0.059 | 1/39 |
| EW | 0.222 | 0.101 | 2.037 | −0.122 | — |

Gap vs ML bar: Config A `[TBD]`, Config B `[TBD]`.

---

## Static Collapse Analysis

### Background

Session 4b trained REINFORCE+baseline on N=1/2/3 assets with 60-step truncated
rollouts. The result was uniform static collapse: mean turnover ≈ 0.00 across all
seeds and configs, weight standard deviation across time ≈ 0.00–0.01. Policies
converged to approximately equal-weight allocations regardless of features,
suggesting that the gradient signal from 60-step windows was too weak to learn
feature-conditional allocations on a 29-asset universe.

The core hypothesis for Session 4c: full rollouts of ~504 steps (one pass through
a 24-month training window) provide substantially more gradient signal per episode
— 8× more steps — and may enable the policy to detect and exploit the feature
structure that 60-step truncation could not.

A secondary hypothesis: the `lambda_risk = 0.02` penalty may reinforce equal-weight
as a local optimum, because equal-weight minimizes the vol penalty absent any return
signal. Config B eliminates this pull to isolate its contribution.

### Diagnostic Gate

```
STATIC_COLLAPSE_DETECTED if:
  mean_turnover < 0.05 AND weight_std_across_time < 0.01
  for BOTH Config A AND Config B
```

OOS diagnostics:
- Config A: mean_turnover = `[TBD]`, weight_std_across_time = `[TBD]` → `[collapsed / not collapsed]`
- Config B: mean_turnover = `[TBD]`, weight_std_across_time = `[TBD]` → `[collapsed / not collapsed]`

**Verdict: `[TBD — fill in after Colab run]`**

### Interpretation of Likely Outcome

Three scenarios are possible, in decreasing order of prior probability given 4b results:

**Scenario 1 — Static collapse persists (most likely given 4b).** Full rollouts at
N=29 still converge to static allocations. This would indicate that REINFORCE's
variance problem is fundamental: even with 504-step Monte-Carlo returns, the
gradient signal in episode-level rewards is too noisy for a 29-asset allocation
task. The policy gradient for portfolio selection scales poorly with N because
(a) the reward variance scales as O(N²) under cross-asset interactions, and (b)
the policy must learn to differentiate 29 assets from a shared encoder, which
requires long-horizon credit assignment that REINFORCE with a simple value baseline
cannot achieve. This is the honest negative finding the paper should report.

**Scenario 2 — Feature-conditional learning at N=29 (possible).** The policy learns
non-trivial allocations. Mean turnover > 0.05, weight variation visible across
time. This would represent the first successful RL signal on the 29-asset universe
and would be the paper's RL headline result. Sharpe competition with the ML bar
(2.579) would still likely be unfavorable given REINFORCE's sample inefficiency,
but a Sharpe above the EW baseline (2.037) would be a meaningful RL contribution.

**Scenario 3 — Partial collapse.** One config collapses, the other does not. Most
interpretable as a λ effect: if Config A (λ=0.02) collapses and Config B (λ=0.00)
does not, the risk penalty is the mechanism. If B also collapses, λ is not the
cause and the gradient variance interpretation holds for both.

---

## λ Ablation Interpretation

The risk penalty `r_t -= lambda_risk × w_t ⊤ vol_t` acts as a regularizer that
penalizes concentration in high-volatility assets. When `lambda_risk = 0.02` and
the signal is weak, equal-weight is a near-optimal solution under the reward
decomposition: `reward = gross_return − tc − 0.02 × portfolio_vol`. The penalty
magnitude at equal-weight on the 29-asset universe is roughly `0.02 × mean_vol ≈
0.02 × 0.012 ≈ 2.4e-4` per step — comparable to the expected excess return signal
per day. This could suppress exploration and lock the policy into near-equal-weight.

Config B (λ=0.00) removes this pull entirely. The reward collapses to
`gross_return − tc`, which is the Sharpe-maximization objective without vol control.
If this creates instability (high-concentration policies with large drawdowns), the
entropy coefficient (0.01) and value baseline are the only stabilizers.

Expected pattern: Config B should show higher turnover and weight variation than
Config A in either the collapse or the non-collapse scenario, because the
equal-weight attractor is removed. If both collapse, Config B's turnover should
still exceed Config A's slightly, confirming the λ mechanism. If neither collapses,
Config B may show more concentrated and volatile allocations (higher Ann Vol, lower
Sharpe), consistent with unconstrained Sharpe maximization via REINFORCE.

Observed:
- Config A mean_turnover (training): `[TBD]`
- Config B mean_turnover (training): `[TBD]`
- Config A mean_turnover (OOS):      `[TBD]`
- Config B mean_turnover (OOS):      `[TBD]`

---

## Comparison vs ML and DL Paradigms

The full 39-strategy table is at
`data/cache/portfolio_returns/full_comparison_with_rl.csv`. Key reference points:

| Paradigm | Best Strategy | OOS Sharpe |
|---|---|---|
| ML (best) | MSR(Ensemble_μ̂) | 2.579 |
| Classical (best) | VMP(MDP(LW)) | 2.422 |
| DL (best) | MSR(MLP_μ̂) | 2.320 |
| EW (reference) | EW | 2.037 |
| RL Config A | RL λ=0.02 ensemble | [TBD] |
| RL Config B | RL λ=0.00 ensemble | [TBD] |

The ML bar of 2.579 was set by a single-fit ensemble of three tree-based μ̂
estimators (Lasso + RF + XGBoost) fed into MSR optimization. This was a
methodologically favorable setting: (a) single fit on a regime-rich 2003–2022
training set, (b) the test window (2023–2026) happened to reward the regime
memory encoded by the training set, and (c) MSR amplification of well-calibrated
signals. REINFORCE faces structural disadvantages: online credit assignment
over 504-step trajectories, no gradient-level access to the portfolio objective,
and gradient variance that grows with N and T. The fair comparison is not whether
REINFORCE beats the ML bar, but whether it produces results above naive baselines
(EW, Risk Parity) — a much more achievable bar given the algorithm's properties.

If the static collapse result holds, the RL Sharpe will approximate EW (since a
static equal-weight policy produces the EW return). The comparison column for RL
in the paper would then report: "REINFORCE converges to approximately equal-weight
allocations on the 29-asset universe regardless of risk penalty; see static
collapse analysis."

---

## Honest Limitations

### Algorithm

REINFORCE is the first algorithm in the RL policy gradient family and has known
variance problems. Its sample complexity for portfolio tasks with N=29 and T~500
is substantially higher than what 200 training episodes can provide. The gradient
estimate per episode:

```
∇θ J ≈ Σ_t ∇θ log π(a_t|s_t) × Ā_t
```

has variance that grows as O(T²) under naive Monte-Carlo returns. The value
baseline reduces this but does not eliminate it. Modern algorithms — PPO, SAC —
address variance via trust-region constraints (PPO) or off-policy experience replay
(SAC), both of which are strictly more sample-efficient than REINFORCE. The
decision to use REINFORCE-only in Session 4c was deliberate (per the session spec)
to establish a baseline and isolate the N-scaling question before introducing
algorithmic complexity.

### Feature Set

Features are rolling 20-day return windows per asset — the same minimal feature set
as Session 4a/4b. The 17-feature engineered set from Session 1.5B (momentum,
volatility, RSI, ATR, Bollinger, gap, volume signal, forward returns, asset class
one-hot) was not used for the RL walk-forward. There are two reasons: (a) the
`PortfolioEnv`'s default feature construction is return windows, and (b) the
static collapse at N=1/2/3 in Session 4b made feature engineering secondary to
resolving the gradient variance problem. If collapse persists at N=29, richer
features are unlikely to help — the bottleneck is the optimizer, not the signal.
If feature-conditional learning emerges, richer features could be tested in Session
4d.

### Training Budget

200 episodes × 504 steps = 100,800 environment interactions per policy fit.
By contrast, PPO implementations for portfolio tasks in the literature (e.g.,
Jiang et al., 2017; Liu et al., 2020) typically use 10⁶–10⁷ interactions before
convergence. The 440-fit budget was chosen to match the DL walk-forward's
per-session compute envelope, not to match RL convergence requirements. This is a
deliberate comparison: the DL methods achieved results with similar compute
budgets; if REINFORCE cannot, the compute-adjusted comparison favors DL.

### Walk-Forward Interference

Unlike the DL walk-forward (which fits supervised models on static (X, y) pairs),
the RL walk-forward trains each refit's policy from scratch on the 24-month window.
There is no warm-starting from the previous refit's policy. This is conservative
but suboptimal: a policy that learned good structure in the 2021–2022 window should
provide a better initialization for 2022–2024 than random initialization. This is
a Session 4d opportunity if the Session 4c results justify continuation.

### OOS Window

The OOS window (2023-01–2026-04, approximately 840 trading days) was a
post-COVID-normalization, post-rate-shock period characterized by strong equity
performance and unusual macro dynamics. Session 2's ML strategies benefited from
this: their training set (2003–2022) encoded the shock regime, and the test window
rewarded the shock-conditioned weights. RL policies trained on 24-month rolling
windows will see very different regimes depending on refit timing, making regime
consistency harder to achieve than for single-fit models.

---

## Recommendation on Session 4d (PPO)

### Criteria for YES

Session 4d (PPO or SAC) is warranted if ANY of the following hold:
1. Static collapse persists in Session 4c with REINFORCE, but the diagnostics
   show partial gradient signal (e.g., final episode reward trending up even if
   turnover remains low) — suggesting the algorithm, not the signal, is the
   bottleneck.
2. Feature-conditional learning emerges (Scenario 2) but Sharpe is substantially
   below the EW baseline (< 1.5) — suggesting the signal is present but REINFORCE
   is not exploiting it efficiently.
3. Config B (λ=0.00) shows better results than Config A, indicating that the reward
   function design is non-trivial and worth exploring with a more stable optimizer.

### Criteria for NO

Session 4d is NOT warranted if:
1. Static collapse persists in both configs AND the training-period diagnostics show
   no improvement in reward over episodes (flat reward curves) — suggesting the
   environment/problem formulation, not the algorithm, is the bottleneck.
2. The OOS Sharpe in the non-collapsed scenario is below EW (2.037) — suggesting
   the learned policy is worse than a trivial baseline even when the policy is
   not static.

### Author Recommendation (pre-run)

Based on Session 4b's results and the theoretical properties of REINFORCE on
high-dimensional portfolio tasks, the most likely outcome is static collapse at
N=29 even with full rollouts. The gradient variance at T=504, N=29 exceeds what
a simple value baseline can reduce to a usable signal-to-noise ratio. If this
holds, Session 4d (PPO) would need to address the fundamental variance problem
with a more principled approach — either PPO's clipped surrogate or SAC's
entropy-maximization with replay buffer. The paper's RL section would then make
a clear algorithmic progression claim: REINFORCE fails at N=29 due to gradient
variance; PPO succeeds (or fails for different reasons) due to trust-region
stability.

The recommendation to proceed with Session 4d should be made only after reviewing
the Colab output's static collapse verdict and the reward curve trajectories.
A flat reward curve (no improvement over 200 episodes) is a strong signal for
Session 4d. An improving curve that still collapses OOS suggests a different
problem (generalization, not optimization) that PPO may not solve.

---

## Figures

All figures in `results/rl/n29/figures/`:

- `equity_curves_rl.png` — OOS cumulative returns: RL configs + top-3 from comparison table
- `turnover_over_time_rl.png` — Daily turnover (monthly resampled) per config, with threshold line
- `weight_heatmap_rl.png` — Per-asset weight over time (monthly resampled heatmap), one panel per config
- `seed_sharpe_dist_rl.png` — Per-seed OOS Sharpe histogram per config, with ML bar reference line

---

## Artifacts

| File | Description |
|---|---|
| `results/rl/n29/diagnostics_all.parquet` | Per-(config, refit, seed) training diagnostics |
| `results/rl/n29/seed_sharpe_distribution.csv` | Per-seed OOS Sharpe for each config |
| `data/cache/portfolio_returns/full_comparison_with_rl.csv` | 39-strategy comparison table |
| `results/rl/n29/figures/*.png` | 4 diagnostic figures |
| `notebooks/06a_rl_reinforce.ipynb` | Execution notebook (Colab-ready) |
| `src/aiam/rl/walkforward.py` | Walk-forward RL adapter |
| `tests/rl/test_walkforward.py` | 10 interface tests (added to 293-test baseline) |

---

## Commit Reference

Branch: `rl/foundation`  
Session 4b close: commit `a8d3af6` (REINFORCE+baseline trainer, 293 tests)  
Session 4c commit: TBD — to be made after Colab run
