# Session 4 — Reinforcement Learning for Portfolio Construction: Findings

## Summary

Two reinforcement learning paradigms were evaluated on the 29-asset universe
(2003–2026) through the same walk-forward harness used for the Classical, ML,
and DL paradigms: on-policy REINFORCE with a value baseline (Session 4c) and
on-policy PPO (Session 4d). Both converge to a near-equal-weight (≈1/N) static
allocation. Neither exceeds the empirical bar set by MSR(Ensemble_μ̂) at
Sharpe 2.579. The RL paradigm does not improve on the ML ensemble on this
universe, and the static collapse is robust to algorithm, risk-penalty
setting, random seed, and full-length rollouts.

## Methodology

- Universe: 29 assets, 7 classes, 2003-01-02 to 2026-04-30. Test window from
  2023-01-01, monthly walk-forward refit (41 refits), 24-month rolling train.
- Formulation: offline historical replay MDP. State = trailing return features
  + current portfolio weights. Action = continuous simplex weight vector
  (softmax / Dirichlet over 29 assets). Reward = wᵀr_{t+1} − cost·‖Δw‖₁
  − λ·risk_penalty, COST_BPS=10.
- REINFORCE (4c): value baseline, γ=0.95, 200 episodes/seed, full rollouts,
  2 configs (λ=0.02, λ=0.00), 10 seeds, 41 refits = 820 policy fits.
- PPO (4d): clipped surrogate (clip_eps=0.2), GAE (λ=0.95, γ=0.95), K=4 epochs,
  lr=3e-4, entropy_coef=0.01, full rollouts, 1 config (λ=0.02), 5 seeds,
  41 refits = 205 policy fits.
- All training offline; normalizers fit on train and frozen; ensemble = mean
  policy across seeds.

## Results

| Strategy | OOS Sharpe | Ann Ret | Ann Vol | Max DD | Rank (of 39) |
|---|---|---|---|---|---|
| MSR(Ensemble_μ̂) — ML bar | 2.579 | — | — | — | 1 |
| RL REINFORCE (λ=0.02) | 2.0255 | 22.05% | 10.09% | −12.16% | 27 |
| RL REINFORCE (λ=0.00) | 2.0256 | 22.05% | 10.09% | −12.16% | 26 |
| RL PPO (λ=0.02) | 2.0267 | 22.06% | 10.09% | −12.16% | 26 |

Diagnostics (the static-collapse evidence):
- REINFORCE OOS mean turnover 0.00068; weight_std_across_time 0.00003;
  seed spread ±0.0008. Training turnover 1.867 (exploration occurred).
- PPO OOS mean turnover 0.000007; weight_std_across_time
  0.000000; verdict STATIC_COLLAPSE_DETECTED.

## The static collapse, and the ≈1/N mechanism

Both algorithms explored during training (REINFORCE training turnover 1.867)
yet converged to a deterministic, near-constant weight vector at OOS
evaluation. The learned weights cluster tightly around 1/29 ≈ 0.034
(final weight std ≈ 0.006), i.e. the agents rediscover an equal-weight
portfolio. This explains the Sharpe ≈ 2.0, which is approximately where a
1/N portfolio lands on this universe over 2023–2026.

The most parsimonious interpretation: after transaction costs, the
feature-conditional edge on this universe is too small to justify dynamic
rebalancing, so the optimal policy under the reward is approximately static.
PPO reaching the same optimum as REINFORCE — with a stronger optimizer,
clipping, and GAE — confirms the collapse is a property of the problem
(weak conditional signal), not the algorithm.

## Relation to the other paradigms

This mirrors the DL direct-weight track (Notebook 05), where the best deep
model (LSTM, CRRA+shrinkage) reached Sharpe 1.240 versus a Risk Parity
benchmark at 1.247 — a tie — and the Cheng & Wu (JPM 2024) reported advantage
did not reproduce. Two independent paradigms (deep learning, reinforcement
learning) reaching the same conclusion is corroboration: the ML ensemble's
one-shot optimization is the standout, and added model complexity does not
reliably add value on this universe.

## Limitations

The finding is conditional on: (i) the 17-feature state, (ii) the
return−cost−risk reward family, (iii) two on-policy algorithms (REINFORCE,
PPO). It is not a claim that RL can never work for allocation. Untested
levers that could in principle change the result — and the reasons they were
not pursued — are below.

## What would be needed to change the result

The rigorous literature attributes most positive RL-allocation results to
overfitting; the genuine RL successes in finance are in optimal execution,
market making, and hedging, not strategic allocation. Levers that the
literature uses to coax dynamic behavior: differential Sharpe reward
(Moody & Saffell), richer state (sentiment / regime / cross-sectional
signals), synthetic data augmentation (diffusion/GAN crisis scenarios), and
off-policy actor-critics (SAC, TD3). Each adds complexity with low expected
out-of-sample payoff on this universe, and pursuing them to force a win would
risk the overfitting trap. They are noted as future work, not gaps in this
study.

## Verdict

RL is a rigorous negative result in the comparative paper: two paradigms,
identical static collapse to ≈1/N, neither exceeding the ML ensemble at 2.579.
The contribution is the clean, corroborated demonstration — consistent with
the DL track and the careful literature — that sequential decision-making does
not add value over one-shot optimization on this universe.
