# Appendix B — Strategy Zoo

This appendix gives full mathematical specifications for all strategy families in the harness. Section 2 of the paper stated operative one-line formulas; here each family receives the complete optimization problem, any closed-form solution, all parameter values used in this study, implementation notes keyed to `src/aiam/strategy/`, and primary citations.

Common notation throughout: $\mathbf{w} \in \mathbb{R}^N$ is the portfolio weight vector, $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ is the estimated covariance matrix of daily returns (annualised where noted), $\boldsymbol{\mu} \in \mathbb{R}^N$ is the expected return vector, and $N = N(t)$ is the active universe size at rebalance date $t$.

---

## B.1 Equal Weight (EW)

**Problem.** No optimization. Assign uniform weight to all $N(t)$ active assets:
$$w_i = \frac{1}{N(t)}, \quad i = 1, \ldots, N(t).$$

**Parameters.** None.

**Implementation.** `src/aiam/strategy/equal_weight.py`. Serves as the fallback in all other strategies when the solver fails or the lookback window is insufficient.

**Citation.** DeMiguel, Garlappi & Uppal (2009) establish EW as the empirical benchmark that diversified MV strategies struggle to beat.

---

## B.2 Global Minimum Variance (GMV)

**Problem.**
$$\min_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0.$$

**Closed form (unconstrained).** $\mathbf{w}^* \propto \boldsymbol{\Sigma}^{-1} \mathbf{1}$. The long-only constraint renders a closed form unavailable in general; CVXPY/OSQP solves the QP.

**Parameters.**
- Lookback: 252 trading days.
- Estimators: `sample_cov`, `ledoit_wolf_cov` (Ledoit & Wolf 2004 analytical shrinkage), `oas_cov` (Oracle Approximating Shrinkage, Chen, Wiesel & Eldar 2010).
- Solver tolerances: `eps_abs = eps_rel = 1e-8` (OSQP).

**Implementation.** `src/aiam/strategy/global_min_variance.py`. Weekend/holiday rows are filtered (`dayofweek < 5`) before slicing to `lookback+1` rows; tickers with > 10% NaN in the window are dropped.

**Citation.** Markowitz (1952); Ledoit & Wolf (2004) for shrinkage estimators.

---

## B.3 Maximum Sharpe Ratio (MSR)

**Problem.** Maximize the Sharpe ratio with risk-free rate $r_f = 0$:
$$\max_{\mathbf{w}} \; \frac{\boldsymbol{\mu}^\top \mathbf{w}}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0.$$

**Parametric reformulation.** Following Markowitz / Dantzig-Wolfe, the equivalent QP is:
$$\min_{\mathbf{y}} \; \mathbf{y}^\top \boldsymbol{\Sigma} \mathbf{y}
\quad \text{s.t.} \quad \boldsymbol{\mu}^\top \mathbf{y} = 1, \quad \mathbf{y} \geq 0.$$
The weight vector is $\mathbf{w} = \mathbf{y} / \|\mathbf{y}\|_1$. Assets with non-positive expected returns (i.e. $\mu_i \leq 0$) are excluded via the positivity constraint; if no assets have positive $\mu$, the strategy falls back to EW.

**Parameters.**
- Lookback: 252 trading days.
- Mean estimator: sample mean of daily returns, annualised.
- Estimators: `sample_cov`, `ledoit_wolf_cov`.

**Implementation.** `src/aiam/strategy/max_sharpe.py`. Solver: CVXPY/OSQP.

**Citation.** Markowitz (1952); Tobin (1958).

---

## B.4 Most Diversified Portfolio (MDP)

**Problem.** Maximize the Diversification Ratio $\mathrm{DR}(\mathbf{w}) = (\boldsymbol{\sigma}^\top \mathbf{w}) / \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$, where $\sigma_i = \sqrt{\Sigma_{ii}}$ is the asset-level volatility. The equivalent parametric QP is:
$$\min_{\mathbf{y}} \; \mathbf{y}^\top \boldsymbol{\Sigma} \mathbf{y}
\quad \text{s.t.} \quad \boldsymbol{\sigma}^\top \mathbf{y} = 1, \quad \mathbf{y} \geq 0,$$
then $\mathbf{w} = \mathbf{y} / \|\mathbf{y}\|_1$. This is structurally identical to the MSR problem with $\boldsymbol{\sigma}$ replacing $\boldsymbol{\mu}$, exploiting the fact that $\boldsymbol{\sigma}$ is always element-wise positive.

**Parameters.** Lookback: 252 days. Estimators: `sample_cov`, `ledoit_wolf_cov`.

**Implementation.** `src/aiam/strategy/most_diversified.py`. Near-zero-volatility assets ($\sigma_i < 10^{-8}$) are dropped to prevent degenerate QP.

**Citation.** Choueifaty & Coignard (2008).

---

## B.5 Risk Parity / Equal Risk Contribution (RP/ERC)

**Problem.** Find $\mathbf{w}$ such that each asset contributes equally to portfolio variance. The risk contribution of asset $i$ is $\mathrm{RC}_i(\mathbf{w}) = w_i \cdot (\boldsymbol{\Sigma}\mathbf{w})_i / \sigma_p(\mathbf{w})$ where $\sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$. The ERC condition $\mathrm{RC}_i = \sigma_p / N$ for all $i$ is enforced via the least-squares objective:
$$\min_{\mathbf{w}} \; \sum_{i=1}^{N} \!\left( w_i \cdot \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p(\mathbf{w})} - \frac{\sigma_p(\mathbf{w})}{N} \right)^{\!2}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad 10^{-6} \leq w_i \leq 1.$$

**Parameters.** Lookback: 252 days. Estimators: `sample_cov`, `ledoit_wolf_cov`. Solver: SciPy SLSQP with `ftol=1e-9`, `maxiter=500`. Note: covariance scaling does not affect the ERC weights (both $\sigma_p$ and $\mathrm{RC}_i$ scale identically), so the daily (non-annualised) covariance is used directly.

**Implementation.** `src/aiam/strategy/risk_parity.py`.

**Citation.** Qian (2005); Roncalli (2013).

---

## B.6 Hierarchical Risk Parity (HRP)

**Algorithm.** Three steps (Lopez de Prado 2016):

1. **Distance matrix.** From the covariance matrix $\boldsymbol{\Sigma}$ (shrinkage carries through), compute the correlation matrix $\boldsymbol{R}$ and then the distance $d_{ij} = \sqrt{(1 - R_{ij})/2}$.

2. **Hierarchical clustering.** Apply Ward linkage to the upper-triangle distance vector. Extract the leaf order $\pi$ from the dendrogram.

3. **Recursive bisection.** Start with all $N$ assets as one cluster $\mathcal{C}$. Repeatedly split each cluster at its midpoint into left $\mathcal{L}$ and right $\mathcal{R}$ sub-clusters. Compute the cluster variance:
$$v(\mathcal{S}) = \mathbf{w}_{\mathcal{S}}^\top \boldsymbol{\Sigma}_{\mathcal{S}} \mathbf{w}_{\mathcal{S}}, \quad w_j = \frac{1}{|\mathcal{S}|} \; \forall j \in \mathcal{S}.$$
Allocate weight proportionally to inverse variance: $\alpha = 1 - v(\mathcal{L}) / (v(\mathcal{L}) + v(\mathcal{R}))$; multiply weights of $\mathcal{L}$ by $\alpha$ and $\mathcal{R}$ by $(1-\alpha)$. Normalise at the end.

**Parameters.** Lookback: 252 days. Clustering method: `ward` (scipy `linkage`). Estimators: `sample_cov`, `ledoit_wolf_cov`. Fully deterministic — no solver.

**Implementation.** `src/aiam/strategy/hierarchical_risk_parity.py`.

**Citation.** Lopez de Prado (2016, SSRN 2708678).

---

## B.7 Black-Litterman (BL)

**Posterior.** The Black-Litterman model combines equilibrium implied returns $\boldsymbol{\pi}$ with investor views $(P, \mathbf{Q}, \boldsymbol{\Omega})$ via a Gaussian conjugate update. Given the prior $\boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\pi},\, \tau\boldsymbol{\Sigma})$ and view likelihood $\mathbf{Q} = P\boldsymbol{\mu} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})$, the posterior mean and covariance are:

$$\boldsymbol{M}^{-1} = (\tau\boldsymbol{\Sigma})^{-1} + P^\top \boldsymbol{\Omega}^{-1} P,$$
$$\boldsymbol{\mu}_{\mathrm{post}} = \boldsymbol{M} \left[ (\tau\boldsymbol{\Sigma})^{-1} \boldsymbol{\pi} + P^\top \boldsymbol{\Omega}^{-1} \mathbf{Q} \right],$$
$$\boldsymbol{\Sigma}_{\mathrm{post}} = \boldsymbol{\Sigma} + \boldsymbol{M}.$$

The portfolio is then the long-only MSR of $(\boldsymbol{\mu}_{\mathrm{post}},\, \boldsymbol{\Sigma}_{\mathrm{post}})$:
$$\min_{\mathbf{y}} \; \mathbf{y}^\top \boldsymbol{\Sigma}_{\mathrm{post}} \mathbf{y}
\quad \text{s.t.} \quad \boldsymbol{\mu}_{\mathrm{post}}^\top \mathbf{y} = 1, \quad \mathbf{y} \geq 0.$$

**Parameters.** $\tau = 0.05$, $\delta = 2.5$ (risk-aversion coefficient for equilibrium $\boldsymbol{\pi} = \delta \boldsymbol{\Sigma} \mathbf{w}_{\mathrm{mkt}}$). Prior weights $\mathbf{w}_{\mathrm{mkt}} = \mathbf{1}/N$ (equal-weight market portfolio). Lookback: 252 days. Covariance is annualised ($\times 252$) so that $\boldsymbol{\pi}$ and view $\mathbf{Q}$ are on the same annual scale.

**View generators** (defined in `src/aiam/estimators/views.py`):

| Variant | $P$ | $\mathbf{Q}$ | $\boldsymbol{\Omega}$ |
|---|---|---|---|
| **BL-Eq** | $\mathbf{0}_{0\times N}$ | $\emptyset$ | $\emptyset$ — reduces to equilibrium MSR |
| **BL-Mom(LW)** | $I_N$ | Annualised trailing-252d return per asset | $\mathrm{diag}(s_i^2 \times 0.05)$, $s_i^2$ = daily variance |
| **BL-Rev(LW)** | $I_N$ | $-1 \times$ annualised trailing-1260d mean per asset | $\mathrm{diag}(s_i^2 \times 0.05)$, $s_i^2$ = daily variance |

BL-Eq has two estimator variants (sample, LW); BL-Mom and BL-Rev use LW only.

**Implementation.** `src/aiam/strategy/black_litterman.py`. Solver: CVXPY/OSQP on the MSR problem.

**Citation.** Black & Litterman (1992); He & Litterman (1999).

---

## B.8 Time-Series Momentum (TSMOM)

**Signal.** Asset $i$'s momentum signal at date $t$ is the sign of its cumulative return over the past $h$ trading days:
$$s_{i,t} = \mathrm{sign}\!\left( \frac{P_{i,t}}{P_{i,t-h}} - 1 \right), \quad s_{i,t} \in \{-1, 0, +1\}.$$
In the long-only variant, negative signals are set to zero.

**Volatility scaling.** Each asset is sized to deliver a target annualised volatility $\sigma^* = 10\%$:
$$\tilde{w}_{i,t} = s_{i,t} \cdot \frac{\sigma^*}{\hat{\sigma}_{i,t}},$$
where $\hat{\sigma}_{i,t} = \sqrt{252} \cdot \mathrm{std}(r_{i,t-62:t})$ is the 63-day rolling annualised volatility.

**Normalisation.** Weights are normalised by gross exposure: $\mathbf{w}_t = \tilde{\mathbf{w}}_t / \|\tilde{\mathbf{w}}_t\|_1$. If all signals are zero (all assets in drawdown), the strategy falls back to EW.

**Parameters.**

| Configuration | Signal lookback $h$ | Vol lookback | Long-only |
|---|---|---|---|
| TSMOM(12m) | 252 days | 63 days | Yes |
| TSMOM(6m) | 126 days | 63 days | Yes |
| TSMOM-LS(12m) | 252 days | 63 days | No |

**Implementation.** `src/aiam/strategy/tsmom.py`.

**Citation.** Moskowitz, Ooi & Pedersen (2012).

---

## B.9 Factor Portfolios (FF3-style)

**Structure.** `FactorPortfolio` computes a cross-sectional signal, selects the top-tercile assets ($k = \lceil N/3 \rceil$), and weights them by inverse annualised volatility. Three single-factor variants and one multi-factor average:

| Variant | Signal function | Lookback |
|---|---|---|
| **FF3-Mom** | 12-1 momentum: $(R^{12m}_i - R^{1m}_i)$, annualised | 756 days (3y); 252/21 day windows for 12m/1m sub-signals |
| **FF3-LowVol** | Negative 6m realised vol: $-\hat{\sigma}_{i,126}$ | 756 days |
| **FF3-Quality** | Trailing 3y per-asset Sharpe: $\bar{r}_i / \hat{\sigma}_i \times \sqrt{252}$ | 756 days |
| **FF3-Multi** | Equal-weight average of Mom, LowVol, Quality weights | — |

**Inverse-vol weighting within the selected tercile:**
$$w_i = \frac{1/\hat{\sigma}_i}{\sum_{j \in \mathcal{T}} 1/\hat{\sigma}_j}, \quad i \in \mathcal{T},$$
where $\hat{\sigma}_i$ is the annualised 252-day realised volatility computed on the daily returns window.

**Long-Short extension.** `FF3-Mom-LS` takes long top-tercile and short bottom-tercile with equal weight within each leg; gross exposure = 1, net $\approx$ 0.

**Implementation.** `src/aiam/strategy/factor_portfolio.py` (`FactorPortfolio`, `FF3MomLongShort`, `MultiFactorPortfolio`); signals in `src/aiam/estimators/factor_signals.py`.

**Citation.** Fama & French (1993); Jegadeesh (1990) for the 12-1 skip-month adjustment.

---

## B.10 Regime-Switching Strategy (SWITCH)

**Regime engine.** Eight macro indicators are classified monthly into one of eight regimes (0–7) by a Lopez de Prado-style three-factor rule on each indicator's level, first difference, and second difference (convexity). The dominant regime at each month-end is the mode across all eight indicator-level regime readings. The eight indicators are: GDP QoQ growth, CPI MoM inflation, unemployment rate (UNEM), 10y Treasury yield (YC\_10Y), 2y Treasury yield (YC\_2Y), yield curve slope (YC\_STEP), VIX, and S&P 500 MoM return (SPX). Macro data are indexed by publication date (no look-ahead bias). The regime parquet is precomputed once and treated as a data product.

The monthly regime is forward-filled to daily frequency within each strategy; the most recent month-end reading is used at each daily rebalance.

**v1 routing (SWITCH(sample) / SWITCH(LW) in the 62-strategy table).**

| Regime | Strategy |
|---|---|
| 0 | EW |
| 5 | MSR(cov) |
| 1, 2, 3, 4, 6, 7 | MDP(cov) |

Covariance estimator matches the variant suffix (sample or LW).

**v2a routing (reported separately in §5.3, not in the 62-strategy Appendix C table).**

| Regime | Strategy |
|---|---|
| 0 | MSR(LW) |
| 5 | MSR(sample) |
| 1, 2, 3, 4, 6, 7 | MDP(LW) |

The v2a rule was derived on the training period (2003–2022) and evaluated OOS; it replaces the R0→EW baseline with R0→MSR(LW), motivated by the observation that Regime 0 (expansion with accelerating growth) historically rewards high-Sharpe concentration.

**Implementation.** `src/aiam/strategy/switching.py` (routing logic); `scripts/build_switch_v2a_weights.py` (v2a weight assembly); `src/aiam/data/regimes/regime_engine.py` (classification).

---

## B.11 Volatility-Managed Portfolio (VMP) Overlay

**Mechanism.** The VMP overlay applies a time-varying exposure scalar to any base strategy's daily returns. Let $r_t$ be the base strategy daily return. The VMP return is:
$$r^{\mathrm{vmp}}_t = c_t \cdot r_t,$$
$$c_t = \mathrm{clip}\!\left( \frac{\bar{\sigma}}{\hat{\sigma}_{t-1}},\; 0.25,\; 1.50 \right),$$
where $\hat{\sigma}_{t-1} = \sqrt{252} \cdot \mathrm{std}(r_{t-22:t-1})$ is the 21-day lagged realised volatility (lag = 1 day enforces no look-ahead) and $\bar{\sigma}$ is the strategy's own long-run realised volatility computed over the full available sample. The clip $[0.25, 1.50]$ prevents leverage collapse and limits gross notional.

**Parameters.**
- Lookback: 21 trading days.
- Lag: 1 day.
- Clip: $[0.25\times, 1.50\times]$.
- Target $\bar{\sigma}$: strategy-specific long-run realised vol (not a universal target).

VMP is applied to all 31 base strategies, producing 31 VMP variants for a total of 62 configurations.

**Implementation.** `src/aiam/evaluation/vmp_assembly.py` (`assemble_vmp_returns`).

**Citation.** Moreira & Muir (2017).

---

## B.12 Constrained Mean-Variance (MSR\_C, MVO\_C)

**MSR\_C.** Maximum Sharpe with per-asset bounds $[l, u] = [5\%, 40\%]$:
$$\max_{\mathbf{w}} \; \frac{\boldsymbol{\mu}^\top \mathbf{w}}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad 0 \leq w_i \leq 0.40.$$
Solved via SciPy SLSQP (direct Sharpe maximisation). After convergence, positions below 5% are zeroed and weights renormalised (soft lower bound).

**MVO\_C.** Global Minimum Variance with the same bounds $[5\%, 40\%]$:
$$\min_{\mathbf{w}} \; \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}
\quad \text{s.t.} \quad \mathbf{1}^\top \mathbf{w} = 1, \quad 0 \leq w_i \leq 0.40.$$
Solved via CVXPY/OSQP (convex QP). Post-hoc soft lower bound as above.

**Parameters.** Lookback: 252 days. Bounds: $(l, u) = (0.05, 0.40)$. Estimators: sample, LW.

**Implementation.** `src/aiam/strategy/max_sharpe_constrained.py` (MSR\_C); `src/aiam/strategy/mvo_constrained.py` (MVO\_C).

---

## B.13 Long-Short Extensions

Three strategies operate on a net-zero (gross-normalised) long-short basis:

**TSMOM-LS(12m).** As in §B.8 but `long_only=False`. Negative momentum signals produce short positions; gross exposure = 1, net $\approx 0$ in balanced markets. In strongly trending markets (most assets in one direction), the net can deviate from zero within the $[-1, 1]$ range.

**BL-Mom-LS(LW).** Black-Litterman with momentum views and `long_only=False`. Assets with negative posterior expected returns receive negative weights; gross normalised to 1. This makes the portfolio a risk-parity-style L/S rather than a directional long-only.

**FF3-Mom-LS.** As in §B.9; equal long/short within top/bottom terciles, gross = 1, net ≈ 0.

**Note on multi-asset L/S.** In a diversified universe spanning equities, bonds, commodities, and FX, a mechanical long-short strategy does not necessarily reduce portfolio risk relative to a long-only benchmark. The cross-asset correlation structure differs from the intra-equity universe on which classical L/S strategies were developed. The long-short variants are included to benchmark the value of short-side information, not as practical investment alternatives.

**Implementation.** Controlled via `long_only=False` flag in `TSMOM` and `BlackLitterman`; `FF3MomLongShort` in `src/aiam/strategy/factor_portfolio.py`.
