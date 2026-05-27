# ML Ensemble MSR Research Handoff

## Executive Summary

This research handoff reviews a deterministic historical ML expected-return plus maximum Sharpe ratio (“MSR”) allocation prototype using the artifact set in `results/ml_ensemble_msr_demo`. The deterministic artifact contract validated successfully, with the expected source files present and no validation errors or missing files reported.

The tested prototype combines machine-learning expected-return forecasts with a long-only MSR allocation approximation. The bounded evidence shows a historical test-period total return of approximately 64.2%, CAGR of approximately 16.6%, annualized volatility of approximately 6.0%, maximum drawdown of approximately -5.9%, and Sharpe ratio of approximately 2.58 over 813 strategy-return observations. These results are economically notable in the historical sample, but they are not deployable evidence on their own.

The most important interpretation is that the return profile must be reviewed alongside implementation assumptions. The baseline excludes transaction costs, does not impose explicit concentration constraints, relies on local cached data, uses a single-fit machine-learning setup, and produces concentrated portfolios with average top-5 weight share near 66.9% and maximum single-asset weight near 47.9%. Human review required. This memo is research-only, provides no investment advice, no target allocations, and no trading recommendations.

## Research Context

This is a historical ML expected-return plus MSR allocation prototype. The deterministic engine owns the strategy mechanics: local artifact generation, model forecasts, portfolio construction, return calculation, metrics, and figures. The OpenAI agent team owns bounded review, critique, synthesis, and communication only.

The reviewed artifact directory is `results/ml_ensemble_msr_demo`. The deterministic run inventory indicates a 29-asset universe, 17 features, a training end date of 2022-12-31, and a test start date of 2023-01-01. The rendered deterministic evidence describes the model components as Lasso, Random Forest, and XGBoost, combined as an equal-weighted ensemble of expected-return forecasts. Portfolio weights are lagged by one trading day before return realization.

No agent loaded external data, discovered new features, changed hyperparameters, modified optimizer rules, changed weights, or made execution decisions.

## Agent Team Interpretation

The Research Manager / Handoff Agent interpretation is that the artifact set is complete enough for a research handoff and that the deterministic evidence supports a coherent historical prototype narrative. However, the handoff should not be read as validation for deployment. The reported performance is conditional on the historical data, local cache, feature set, model setup, and optimizer configuration used by the deterministic engine.

The Data QA Agent interpretation is that the artifact contract passed: `predictions.parquet`, `weights.parquet`, `strategy_returns.parquet`, `metrics.json`, `report.md`, and `run_manifest.json` were present. The bounded summaries showed finite prediction, weight, and strategy-return data, with no null predictions reported. The predictions cover 29 assets from 2023-01-03 to 2026-03-31; weights cover 813 rows and 29 assets over the same date span; strategy returns cover 813 observations from 2023-01-04 to 2026-04-01. A caveat remains that the manifest-level full data date range extends from 2003-01-02 to 2026-04-30, while the bounded test artifacts cover the post-train strategy period. This distinction should be reconciled and documented for human review.

The Quant Strategy Agent would interpret the result as a promising but preliminary single-fit ML ensemble prototype. The reported Sharpe and CAGR are strong in the historical test window, but the research design still requires robustness checks, including rolling or expanding refits, sensitivity to universe definition, benchmark reconciliation, and transaction-cost-aware evaluation. Since the deterministic engine owns the strategy mechanics, this review does not change model features, targets, hyperparameters, or optimizer logic.

The Portfolio Risk Agent would focus on implementation risk. The portfolio is long-only and appears fully invested based on weight row sums near 1.0, but the optimizer creates meaningful concentration. Average effective positions are approximately 6.73, average max single-asset weight is approximately 33.3%, average top-5 weight share is approximately 66.9%, and maximum top-5 share reaches approximately 83.4%. Historical weights are not target allocations.

The Performance Review Agent would note that the return stream is attractive in-sample for the test period but incomplete as an institutional performance case. No transaction costs are included; turnover is non-trivial, with average turnover approximately 4.9% and maximum turnover approximately 44.9%; and benchmark reconciliation remains an open question.

## Performance Interpretation

The reported headline metrics are:

| Metric | Value |
| --- | ---: |
| Total return | 0.641675 |
| CAGR | 0.166087 |
| Arithmetic annualized return | 0.155514 |
| Annualized volatility | 0.060289 |
| Sharpe | 2.579464 |
| Max drawdown | -0.059160 |
| Observations | 813 |

Sharpe is calculated as arithmetic annualized return divided by annualized volatility. In this artifact set, the arithmetic annualized return is approximately 15.6% and annualized volatility is approximately 6.0%, producing a Sharpe ratio of approximately 2.58.

CAGR and arithmetic annualized return are different measures. CAGR reflects compounded growth over the full period, while arithmetic annualized return annualizes the average periodic return. Here, CAGR is approximately 16.6%, while arithmetic annualized return is approximately 15.6%. The difference should not be treated as an error; it reflects distinct calculation conventions.

The strategy-return artifact contains 813 observations, with bounded daily return summary statistics showing a mean return of approximately 0.000617, standard deviation of approximately 0.003798, minimum return of approximately -0.02688, and maximum return of approximately 0.03477. These are historical test-period statistics only.

## Portfolio Construction and Risk Interpretation

The baseline is a long-only MSR allocation prototype using ensemble expected-return forecasts. The weights artifact contains 813 rows across 29 assets, with row sums effectively equal to 1.0 and no negative weights reported in the bounded summary. The average number of nonzero assets is approximately 18.6, but effective diversification is materially lower because the optimizer concentrates capital.

Concentration is a key implementation caveat. The maximum single-asset weight reaches approximately 47.9%. Average top-5 concentration is approximately 66.9%, and maximum top-5 concentration reaches approximately 83.4%. Average effective positions are approximately 6.73. These diagnostics suggest the allocation may be sensitive to forecast error, covariance estimation, optimizer instability, and asset-specific shocks.

Turnover also requires review. Average turnover is approximately 4.9%, median turnover is approximately 4.3%, and maximum turnover is approximately 44.9%. Since the baseline does not include transaction costs, slippage, borrow constraints, taxes, or market-impact assumptions, reported returns may overstate implementable performance.

There are no concentration constraints in the baseline diagnostics. This is an important distinction: concentration metrics are descriptive evidence, not imposed portfolio limits.

## Figures

![Cumulative Returns](figures/cumulative_returns.png)
![Drawdown](figures/drawdown.png)
![Turnover](figures/turnover.png)
![Concentration](figures/concentration.png)
![Top Weights](figures/top_weights.png)

## Methodology Caveats

This is research-only and based on a historical backtest. It is not investment advice, not a target allocation, and not a trading recommendation.

Key caveats for human review include local cache dependence, single-fit ML setup, no transaction costs, optimizer concentration, and benchmark reconciliation. The results depend on local cached inputs and the approved universe definition. The deterministic evidence states no live data call was used. Cache vintage, data lineage, feature availability timing, and survivorship assumptions should be reviewed before any further interpretation.

The single-fit setup also limits inference. Rolling or expanding retraining would help test whether the model behavior is stable through time. The ML ensemble and MSR optimizer may perform differently across regimes, especially if expected-return estimates are noisy.

Benchmark reconciliation remains open. The result should be reconciled against the relevant notebook or published benchmark diagnostics before it is treated as a stable research finding.

## Recommended Next Research Questions

Recommended next research work should remain research-only:

- Add transaction-cost and slippage sensitivity analysis.
- Test explicit concentration constraints and compare risk-adjusted performance.
- Evaluate rolling or expanding refit procedures.
- Reconcile reported metrics against the benchmark and Notebook 03 diagnostics.
- Test sensitivity to cache vintage, universe definition, and date alignment.
- Review whether feature timing and data availability assumptions are free of look-ahead bias.

## Human Review Checklist

Human review required.

- Confirm artifact contract validity and source cache lineage.
- Confirm universe definition, date ranges, and train/test split.
- Review feature columns and feature timing assumptions.
- Reconcile deterministic metrics with benchmark diagnostics.
- Assess concentration, turnover, and transaction-cost sensitivity.
- Confirm historical weights are not interpreted as target allocations.
- Confirm this memo is treated as research-only and not as trading guidance.

## Appendix / Source Artifacts

The deterministic source artifacts are:

- `run_manifest.json`
- `metrics.json`
- `report.md`
- `predictions.parquet`
- `weights.parquet`
- `strategy_returns.parquet`
- `figures/*.png`

Artifact contract status: valid, with no missing files or validation errors reported for `results/ml_ensemble_msr_demo`.