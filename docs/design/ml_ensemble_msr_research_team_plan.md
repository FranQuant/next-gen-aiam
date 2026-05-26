# ML Ensemble MSR Research Team Plan

## 1. Objective

The ML Ensemble MSR research team wraps the deterministic
`MSR(Ensemble_mu_hat)` engine implemented in
`src/aiam/strategy/ml_ensemble_msr.py` and exposed through
`scripts/run_ml_ensemble_msr_research.py`.

The team is a governed research and critique layer. It may trigger a
deterministic run, inspect the generated artifacts, summarize evidence, identify
methodology risks, and assemble a final research handoff memo for human review.
It must not own or alter strategy logic. The deterministic engine remains the
source of truth for data preparation, model fitting, forecasts, portfolio
weights, backtest alignment, metrics, diagnostics, and artifact writing.

## 2. Non-Goals

This workflow is not an autonomous portfolio manager and must not be presented
as investment advice.

Explicitly forbidden:

- Autonomous portfolio management.
- Investment advice.
- Live order placement or brokerage integration.
- Live EODHD calls by default.
- Live API calls by default.
- Feature selection by an LLM.
- Model hyperparameter selection by an LLM.
- Target definition changes by an LLM.
- Optimizer rule or portfolio weight selection by an LLM.
- Arbitrary repository file reading.
- `.env` access.
- Secret, credential, token, or private-key inspection.
- Notebook execution.
- Expanding the artifact read surface without reviewed deterministic code and
  tests.

## 3. Deterministic Engine Boundary

The deterministic code owns:

- Data loading from local cache.
- Feature construction.
- Target construction.
- Lasso, Random Forest, and XGBoost fitting.
- Ensemble prediction.
- MSR weight construction.
- Lagged backtest alignment.
- Performance metrics.
- Turnover diagnostics.
- Concentration diagnostics.
- Artifact writing.

The agents must treat these outputs as immutable evidence. They may critique
the methodology and reporting assumptions, but they must not replace any of the
engine's choices with LLM-generated alternatives.

Current deterministic artifact output:

```text
predictions.parquet
weights.parquet
strategy_returns.parquet
metrics.json
report.md
run_manifest.json
```

The generated manifest currently records the strategy, local-cache date range,
feature columns, model components, covariance lookback, horizon, validation
share, artifact files, metrics, turnover diagnostics, concentration diagnostics,
caveats, and reproducibility notes.

## 4. Agent Team

### Research Manager Agent

Responsibility:

- Coordinate the workflow.
- Decide whether to use an existing artifact directory or trigger the
  deterministic local-cache runner.
- Route artifact summaries to the specialized review agents.
- Assemble the final response from agent outputs and validation results.

Allowed inputs:

- CLI arguments.
- Deterministic tool outputs.
- `run_manifest.json` summaries.
- `metrics.json` summaries.
- Existing `report.md`.
- Agent critique summaries.

Allowed tools:

- `run_ml_ensemble_msr_pipeline(output_dir: str) -> str`
- `summarize_artifact_inventory(output_dir: str) -> dict`
- `load_run_manifest(output_dir: str) -> dict`
- `load_metrics(output_dir: str) -> dict`
- `load_report_markdown(output_dir: str) -> str`
- `render_research_handoff(output_dir: str) -> str`

Forbidden behavior:

- Reading arbitrary files.
- Editing strategy code.
- Choosing model features, targets, hyperparameters, optimizer settings, or
  weights.
- Calling live market-data APIs.
- Calling LLMs unless explicitly invoked through a future `--run-team
  --model-id MODEL_ID` path.
- Producing recommendation or allocation language.

Expected output:

- A workflow status summary.
- A list of artifacts read.
- A consolidated research handoff memo or a clear failure reason.

### Data QA Agent

Responsibility:

- Review `run_manifest.json` for data provenance and schema consistency.
- Check date range, universe size, feature count, feature columns, model
  components, and local-cache assumptions.
- Surface missingness caveats when available from deterministic artifacts.
- Confirm that the run manifest states local-cache-only and no live EODHD call.

Allowed inputs:

- `run_manifest.json`.
- Artifact inventory summary.
- Bounded metadata summaries for `predictions.parquet`, `weights.parquet`, and
  `strategy_returns.parquet`.

Allowed tools:

- `load_run_manifest(output_dir: str) -> dict`
- `summarize_artifact_inventory(output_dir: str) -> dict`
- Future deterministic parquet summary helpers that return shape, columns, date
  range, null counts, and finite-value checks.

Forbidden behavior:

- Loading raw cache inputs unless a reviewed deterministic tool exposes a
  bounded summary.
- Reading `.env`, private caches, notebooks, or arbitrary paths.
- Changing feature definitions or universe membership.
- Inferring fresh data quality claims from unapproved files.

Expected output:

- Data QA findings with evidence references.
- Date range and universe summary.
- Feature-column inventory.
- Data caveats and open checks for human review.

### Quant Strategy Agent

Responsibility:

- Review the deterministic modeling setup.
- Summarize model components, target horizon, feature set, validation share,
  single-fit setup, Sharpe convention, and Notebook 03 reproduction context.
- Check that arithmetic annualized return and CAGR are described separately.
- Critique methodology without changing the strategy.

Allowed inputs:

- `run_manifest.json`.
- `metrics.json`.
- `report.md`.
- Optional bounded summaries of predictions.

Allowed tools:

- `load_run_manifest(output_dir: str) -> dict`
- `load_metrics(output_dir: str) -> dict`
- `load_report_markdown(output_dir: str) -> str`

Forbidden behavior:

- Selecting new features.
- Retuning Lasso, Random Forest, or XGBoost hyperparameters.
- Changing target horizon or target definition.
- Replacing the ensemble rule.
- Making investment recommendations.

Expected output:

- Strategy mechanics summary.
- Metric convention check.
- Reproduction evidence summary.
- Methodology caveats and open questions.

### Portfolio Risk Agent

Responsibility:

- Review risk, turnover, and concentration evidence.
- Summarize annual volatility, max drawdown, total return, one-way turnover,
  Herfindahl concentration, effective number of positions, max single-asset
  weight, top-5 concentration, and transaction-cost absence.
- Highlight optimizer concentration and covariance instability risks.

Allowed inputs:

- `metrics.json`.
- `run_manifest.json`.
- `report.md`.
- Bounded summaries of `weights.parquet` and `strategy_returns.parquet`.

Allowed tools:

- `load_metrics(output_dir: str) -> dict`
- `load_run_manifest(output_dir: str) -> dict`
- `summarize_artifact_inventory(output_dir: str) -> dict`
- Future deterministic weight and return summary helpers.

Forbidden behavior:

- Imposing portfolio constraints.
- Modifying weights.
- Creating target allocations.
- Estimating live execution cost from external data.
- Treating historical weights as recommendations.

Expected output:

- Risk review.
- Turnover and concentration interpretation.
- Transaction-cost and execution caveats.
- Human-review checklist items.

### Research Handoff Agent

Responsibility:

- Produce the final markdown memo.
- Keep evidence, interpretation, caveats, and open questions separated.
- Apply no-advice and human-review language.
- Include deterministic artifact references and metric definitions.

Allowed inputs:

- Research Manager summary.
- Data QA findings.
- Quant Strategy findings.
- Portfolio Risk findings.
- `run_manifest.json`, `metrics.json`, and `report.md` summaries.

Allowed tools:

- `render_research_handoff(output_dir: str) -> str`
- Future deterministic packet validation helper.

Forbidden behavior:

- Introducing uncited claims.
- Producing allocation, trade, or investment advice.
- Hiding caveats.
- Expanding the read surface.

Expected output:

- Final markdown research handoff memo.
- Explicit caveats.
- Open questions.
- Human review checklist.

## 5. Tool Surface

The first implementation should be deterministic Python tools. SDK wrappers can
call these tools later, but they should not own file access, validation, or
research logic.

Recommended Python functions:

```text
run_ml_ensemble_msr_pipeline(output_dir: str) -> str
load_run_manifest(output_dir: str) -> dict
load_metrics(output_dir: str) -> dict
summarize_artifact_inventory(output_dir: str) -> dict
load_report_markdown(output_dir: str) -> str
render_research_handoff(output_dir: str) -> str
```

Additional deterministic helpers should be added before any SDK wrapper:

```text
validate_ml_ensemble_msr_artifact_contract(output_dir: str) -> dict
summarize_predictions_artifact(output_dir: str) -> dict
summarize_weights_artifact(output_dir: str) -> dict
summarize_strategy_returns_artifact(output_dir: str) -> dict
validate_research_handoff(markdown: str) -> dict
```

Design constraints:

- Tools accept an artifact directory, not arbitrary file paths.
- JSON readers load only expected file names.
- Parquet readers return bounded summaries by default.
- Full `predictions.parquet`, `weights.parquet`, and
  `strategy_returns.parquet` contents remain available only to deterministic
  code, not directly to LLM prompts.
- Tool outputs should be JSON-serializable.
- Tool outputs should include caveats and validation status.

## 6. Artifact Contract

### `predictions.parquet`

Purpose:

- Stores deterministic ensemble expected-return predictions indexed by date and
  asset.

Read mode:

- Deterministic tools may read the full file.
- Agents should receive summaries only: shape, date range, asset count, null
  counts, finite-value checks, and limited distribution statistics.

Caveats:

- Predictions are historical research outputs, not forecasts for live trading.
- Agents must not alter or re-rank predictions.

### `weights.parquet`

Purpose:

- Stores deterministic MSR portfolio weights by signal date and asset.

Read mode:

- Deterministic tools may read the full file.
- Agents should receive summaries only: shape, date range, max weight,
  concentration metrics, turnover metrics, and validation checks.

Caveats:

- Historical weights are not target allocations.
- No constraint is imposed by the diagnostic layer.

### `strategy_returns.parquet`

Purpose:

- Stores lagged backtest returns for `MSR(Ensemble_mu_hat)`.

Read mode:

- Deterministic tools may read the full file.
- Agents should receive summaries only: observation count, date range,
  cumulative return, drawdown summary, finite-value checks, and headline
  performance metrics.

Caveats:

- Historical backtest returns do not imply live performance.
- Backtest uses lagged weights but excludes transaction costs in the baseline.

### `metrics.json`

Purpose:

- Stores structured performance metrics, turnover diagnostics, and
  concentration diagnostics.

Read mode:

- Agents may inspect the full JSON payload.

Caveats:

- Sharpe uses arithmetic annualized return over annualized volatility.
- CAGR and arithmetic annualized return are both reported and must not be
  conflated.

### `report.md`

Purpose:

- Stores the deterministic human-readable run report.

Read mode:

- Agents may inspect the full markdown report.

Caveats:

- The report is generated by the deterministic engine and should be cited as
  run evidence, not as an LLM-authored conclusion.

### `run_manifest.json`

Purpose:

- Stores deterministic run metadata, artifact inventory, metrics, diagnostics,
  caveats, and reproducibility notes.

Read mode:

- Agents may inspect the full JSON payload.

Caveats:

- `created_at_utc` may be deterministic rather than wall-clock time.
- Manifest contents should be treated as the artifact contract for the run.

## 7. Recommended Implementation Order

1. Add deterministic artifact inspection tools.
2. Add tests for artifact inspection tools.
3. Add CLI for research handoff generation without LLM.
4. Add OpenAI Agents SDK wrapper with the 5-agent team.
5. Add optional live run mode only after deterministic path is stable.

The first three steps should be fully useful without any LLM dependency.

## 8. OpenAI Agents SDK Design

The future OpenAI Agents SDK layer should be a thin orchestration wrapper over
the deterministic Python tools.

Requirements:

- Use lazy imports so the base package and deterministic CLI do not require the
  SDK.
- Place SDK dependencies in an optional dependency group.
- Keep no live run by default.
- Provide deterministic CLI modes first.
- Add `--run-team --model-id MODEL_ID` as the only LLM-capable path.
- Require an explicit existing or newly generated artifact directory.
- Do not read `.env` inside the workflow; credentials are outside this design.
- Apply guardrails that require research-only, no-advice language.
- Validate final memo text for forbidden recommendation language.
- Use tracing only with bounded, non-sensitive summaries.
- Never send full parquet contents, local cache inputs, private paths, secrets,
  or notebook contents to a model.

Suggested future flow:

1. CLI validates artifact directory or runs deterministic pipeline locally.
2. Deterministic tools build bounded summaries.
3. Research Manager Agent receives only allowed summaries.
4. Specialized agents critique assigned sections.
5. Research Handoff Agent drafts the memo.
6. Deterministic validator checks structure, caveats, artifact references, and
   no-advice language.
7. CLI writes or prints the memo for human review.


## 9A. Output Presentation Standards

The final research handoff should be written as clean GitHub-compatible Markdown.

The output should not be a plain text dump. It should use:

- Clear section headings.
- Compact metric tables.
- Compact artifact inventory tables.
- Bullet-point caveats.
- Human-review checklist items.
- Code fences only for reproducible commands.
- No excessive raw JSON dumps in the final memo.

The preferred demo output is a polished Markdown research memo that can be read in the terminal, GitHub, or a rendered Markdown viewer. Tables should be used for metrics, artifact inventories, risk diagnostics, and agent findings.

The LLM-facing Research Handoff Agent may draft prose, but all numerical tables should come from deterministic artifacts such as `metrics.json`, `run_manifest.json`, `weights.parquet`, and `strategy_returns.parquet`.

This presentation standard is cosmetic and communicative only. It must not change the deterministic strategy, metrics, weights, or research conclusions.

## 9. Final Research Handoff Template

```text
# ML Ensemble MSR Research Handoff

## Executive Summary

## Deterministic Run Inventory

## Strategy Mechanics

## Performance Metrics

## Turnover and Concentration

## Methodology Caveats

## Risk Review

## Open Questions

## Human Review Checklist
```

The memo should include explicit statements that it is research-only, requires
human review, and does not provide investment advice or target allocations.

## 10. Risks

- Single-fit model risk: the current deterministic run uses one fitted setup,
  so apparent out-of-sample performance may not represent model stability.
- No transaction costs: baseline metrics omit costs, slippage, financing, and
  operational constraints.
- Optimizer concentration: MSR optimization can concentrate weights in a subset
  of assets even when diagnostics report concentration clearly.
- Covariance instability: the 504-session lookback covariance estimate may be
  sensitive to regime shifts and sample composition.
- EODHD cache dependence: the run depends on local cached EODHD-derived data
  and should not be interpreted without provenance review.
- Target/feature leakage risk: target timing, rolling features, and train/test
  separation require continuing review even though the deterministic runner
  encodes the current intended alignment.
- Reporting vs recommendation boundary: historical weights, returns, and
  diagnostics must remain research evidence, not implementation guidance.
- LLM hallucination risk: future agents may overstate evidence, invent causal
  explanations, or soften caveats unless bounded by deterministic tools and
  validators.
