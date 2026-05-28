# §4 When Estimator Choice Matters and When It Doesn't

Section §3 established that VMP lifts every base strategy through the same inverse-vol
mechanism, but the overlay does not explain the performance differences that precede it.
MSR(LW) Sharpe=1.059 versus MSR(sample) Sharpe=0.895 is a gap of +0.164, large enough to
reorder the leaderboard. HRP(LW) Sharpe=1.093 versus HRP(sample) Sharpe=1.045 is a gap of
0.047, small enough to be indistinguishable from noise. The same statistical substitution —
Ledoit-Wolf in place of the sample covariance — produces one of the largest single-method
effects in the study for one family and is effectively zero for another. This section asks why.

## §4.1 The MSR shrinkage benefit

MSR optimizes the portfolio Sharpe ratio directly, with the covariance matrix appearing in
the denominator's curvature. On the 29-asset universe, the sample estimator produces an
ill-conditioned matrix: its dominant eigenvalue corresponds to a correlated equity cluster,
and the optimizer loads heavily on whichever asset had the highest sample Sharpe in the
trailing estimation window — typically a low-vol fixed-income ETF during a trending quarter.
The concentrated position unwinds with mean reversion out of sample (Michaud 1989). The
result is MSR(sample) Sharpe=0.895, among the lower base-strategy results in the table
despite maximizing in-sample Sharpe at each refit (Finding 2).

Ledoit-Wolf regularization shrinks the sample eigenvalues toward their grand mean and the
off-diagonal covariances toward zero by an analytically optimal factor. The extreme
concentration is defanged: the optimizer still tilts toward the highest-in-sample-Sharpe
cluster, but the eigenvalue gaps are smaller and the off-diagonals are pulled back, dampening
the corner solution. MSR(LW) Sharpe=1.059, a +0.164 improvement — the largest
single-estimator substitution effect in the full 62-strategy comparison.

The Memmel (2003) paired test on the 29-asset 2003–2026 sample gives $z=1.13$, $p=0.259$:
directional but not significant at conventional thresholds (Finding R1). This is worth
stating plainly. The economic gap (+0.164 Sharpe) is meaningful, but 23.3 years of daily
returns on 29 assets is not a large sample for paired Sharpe inference — the test is
underpowered. The earlier test on a 30-asset 2008–2026 panel reached clear significance
($z=2.78$, $p=0.005$); the longer pre-GFC period (2003–2007), where momentum and low-vol
regimes were less volatile, dilutes the measured shrinkage benefit in the extended sample.
The directional consistency across both universes — LW ahead in both, significant in one —
is the stronger evidence than either p-value alone.

## §4.2 The HRP near-invariance

HRP allocates through a three-step procedure: compute the correlation matrix, apply Ward
hierarchical clustering on correlation distance to produce a dendrogram, then bisect the
tree recursively — allocating each cluster's capital in inverse proportion to its subtree's
variance (de Prado 2016). No matrix inversion occurs at any step. The covariance matrix
enters only through the diagonal (individual variances) and the pairwise correlations that
define the dendrogram's branch distances.

Because HRP never concentrates on extreme eigenvalues — recursive bisection caps individual
cluster weights structurally — Ledoit-Wolf shrinkage has little to correct. The off-diagonal
smoothing does refine the correlation block structure the dendrogram uses for cluster
boundaries, but the effect on final portfolio weights is second-order. On the 29-asset
sample: HRP(sample) Sharpe=1.045, HRP(LW) Sharpe=1.093, $\Delta=+0.047$ in favor of LW.
Memmel $z=-0.67$, $p=0.506$ — not close to significance (Finding R4).

The sign reversal across samples reinforces the interpretation. In the prior 30-asset
2008–2026 study, HRP(sample) led by +0.037 (sample ahead). A +0.047 LW advantage here and
a −0.037 LW disadvantage there, both non-significant, are consistent with sampling noise
around a true effect near zero. The conservative conclusion is empirically supported: HRP
performance is approximately invariant to shrinkage choice in long multi-asset samples.

A boundary case sharpens the argument. Black-Litterman with no active views (BL-Eq) is an
algebraic limit: when the view specification matrix $P$ is null, the BL posterior mean
reduces to the equilibrium prior regardless of $\Sigma$, because the update term
$P'\Omega^{-1}P$ vanishes. The covariance estimator is literally irrelevant. BL-Eq(sample)
and BL-Eq(LW) produce return series differing by at most $2.8 \times 10^{-8}$ per day —
floating-point rounding only (Finding 10). This is a useful diagnostic boundary: estimator
choice is exactly zero when the optimizer never differentiates on covariance structure.

## §4.3 Same operation, opposite outcomes — the structural reason

The +0.164 MSR lift and the near-zero HRP response point to the same root cause: it is not
the estimator that determines whether shrinkage matters, it is what the optimizer does with
the covariance matrix.

MSR is the canonical noise-amplification machine. Maximizing $\hat{\mu}'\hat{\Sigma}^{-1}\hat{\mu}$
requires inverting $\hat{\Sigma}$, and inversion exposes every estimation error in the sample
matrix — eigenvalue noise is amplified, not absorbed. LW regularization counteracts this
directly: shrinking the eigenspectrum before inversion reduces the condition number and
defangs the concentration. The substitution is consequential because inversion is exactly the
operation that propagates estimation error most aggressively.

HRP is structurally noise-tolerant. Recursive bisection never inverts anything; the
correlation-based clustering and variance-weighted bisection dampen single-asset dominance
geometrically. LW smooths the off-diagonals, but the optimizer that consumes them was
already insensitive to their exact values. The substitution is inconsequential because
hierarchical bisection does not propagate eigenvalue noise. GMV is the extreme version of
the MSR pathology: unconstrained, it corners entirely in the minimum-variance asset (SHY,
iShares 1–3 Year Treasury), producing vol=3.16% and Sharpe=0.958 on a portfolio that is
effectively cash at the prevailing risk-free rate (Finding 1). LW breaks this corner:
GMV(LW) vol=4.01%, Sharpe=0.954, a diversified portfolio at essentially identical Sharpe.

The practitioner implication follows: choosing a robust *method* — HRP, MDP, RP — makes
the choice of *estimator* a second-order decision; use whichever is convenient. Choosing a
noise-amplifying *method* — MSR, unconstrained MVO — makes the estimator a first-order
decision; Ledoit-Wolf is not optional.

---

There is one exception to the pattern — and it is the foundation of §5.

## §4 → §5 Transition: the regime where sample beats LW

The §4 finding holds on average across the full 23-year sample: LW shrinkage helps MSR.
Finding 4 introduces a crack in that generality. In Regime 5 — Low Growth & Contracting,
the late-cycle or early-recession configuration in the 8-regime macro classifier — MSR(sample)
produces conditional Sharpe=1.392 against MSR(LW) conditional Sharpe=1.097. Sample wins
by +0.295 within this single regime. In every other regime, LW matches or beats sample
(Finding 4). Regime 5 covers 924 of 5,868 daily observations (15.7% of the full sample).

Why would the noisy, overfit optimizer outperform the regularized one in a specific regime?
In a late-cycle contraction, a narrow subset of assets — typically short-duration fixed
income and defensive equities — generates the highest realized Sharpe. The sample estimator
concentrates on exactly this cluster, which happens to be correct ex post. LW shrinkage
pulls the portfolio back toward diversification and misses the concentrated trade. The
average disadvantage of sample concentration (the full-sample −0.164 Sharpe gap) reflects
that this regime accounts for only 15.7% of observations; the remaining 84.3% penalize the
noisy concentration, overwhelming the R5 advantage in the aggregate.

The regime-conditional routing that exploits this exception — substituting MSR(sample) for
MDP(LW) in Regime 5 while leaving all other regimes unchanged — is the single move that
lifts SWITCH from Sharpe=1.080 to Sharpe=1.514, the strongest statistically significant
result in the study ($z=2.05$, $p=0.040$). §5 develops the construction, the identification
strategy, and the sub-period evidence.
