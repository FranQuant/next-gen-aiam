# 07b — PCA Dislocation and Macro-Enhanced Momentum Dashboard: Design Plan

## 1. Objective and narrative role

Notebook 07b should extend the completed Notebook 07 baseline from structural representation learning into a tactical diagnostic dashboard. Notebook 07 asks what latent cross-asset structure exists in the canonical 29-asset universe. Notebook 07b should ask which assets are currently rich or cheap versus a rolling PCA reconstruction, and whether trailing momentum and existing macro/regime context confirm or contradict that dislocation.

```text
07  = unsupervised representation learning baseline
07b = PCA dislocation + momentum + macro/regime diagnostic dashboard
```

Notebook 07b is inspired by the JPM PCA dislocation / Tactical Trifecta research, but it must be adapted to the repo's available 29-asset panel, asset-class metadata, and existing regime labels. It should not claim to replicate JPM's broader 100+ asset universe, proprietary positioning, or Macrosynergy-style macro support scores.

07b is a diagnostic dashboard notebook. It is not a strategy, not an allocation engine, not a backtest, and not a replication of JPM positioning or proprietary macro scores.

## 2. Research references and implementation mapping

The reviewed research motivates a three-lens diagnostic: PCA reconstruction residuals identify rich/cheap cross-asset dislocations, momentum helps classify whether price action confirms or contradicts the residual, and macro/regime context helps interpret whether the market backdrop supports the observed behavior. In this repo, the first two lenses can be implemented directly from canonical prices and returns. The third should use existing `dominant_regime` labels as a transparent context overlay only.

Key references:

- Simple lines intertwining: PCA Dislocations, Positioning, and Momentum
- PCA Dislocations, Positioning, and new Macro-enhanced Momentum
- Tactical Trifecta: PCA Dislocations, Positioning, and Momentum NEW Dashboard
- Less Is More: Factor Investing by PCA and Its Variants
- Scaled Factor Portfolio
- Lettau-Pelger RP-PCA
- Autoencoder cluster report

### Directly used in 07b v1

- PCA dislocation: rolling PCA reconstruction residuals on the 29-asset panel.
- Momentum confirmation: trailing-only multi-horizon price momentum.
- Macro/regime context: existing `dominant_regime` labels aligned without future leakage.
- Dashboard/scorecard presentation: latest cross-sectional scorecard, rich/cheap rankings, scatter views, heatmaps, and selected time series.

### Acknowledged but not implemented in 07b v1

- True positioning, including futures positioning, fund betas, surveys, options positioning, ETF flows, or proprietary crowding aggregates.
- JPM's 100+ asset cross-asset universe.
- Macrosynergy-style point-in-time macro support scores.
- RP-PCA.
- Sharpe-scaled PCA.
- Autoencoder cluster portfolios.

### Deferred to later notebooks

```text
07c = PCA variants / RP-PCA / Sharpe-scaled PCA
07d = autoencoder cluster portfolios
```

## 3. Data inputs

Do not fetch web data, introduce new external providers, require JPM-style positioning data, or silently fabricate positioning proxies.

| Input | Path | Expected shape / key columns | Frequency | Role in 07b | Fallback behavior |
| --- | --- | --- | --- | --- | --- |
| Canonical 29-asset price panel | `data/cache/prices_29.parquet` | 5,869 rows x 29 assets, date index, columns matching `UNIVERSE_29`; currently 2003-01-02 to 2026-04-30 | Daily | Primary source for weekly resampling and trailing price momentum | If unavailable, use `data/published/ohlcv_29assets_2003_2026.csv` only as a documented fallback to rebuild adjusted close prices |
| Canonical 29-asset return panel | `data/cache/returns_29_2003_2026.parquet` | 5,869 rows x 29 assets, date index, columns matching `UNIVERSE_29`; currently 2003-01-02 to 2026-04-30 | Daily | Sanity check against price-derived returns; optional source for daily momentum diagnostics | If unavailable, recompute returns from `prices_29.parquet` with explicit checks |
| Asset universe | `src/aiam/data/universe.py` | `UNIVERSE_29`, ordered list of 29 tickers | Static | Canonical asset ordering for all tables and plots | Fail fast if missing, because canonical ordering is part of the repo contract |
| Asset-class mapping | `src/aiam/features/asset_class.py` | `ASSET_CLASS_MAP`, seven classes: `us_single_stock`, `us_sector_etf`, `broad_equity_etf`, `intl_equity_etf`, `fixed_income_etf`, `commodity_etf`, `fx_spot` | Static | Grouping, color, filters, and scorecard metadata | Assign `unknown` only for display if a new asset appears; current 29 assets should all map |
| Macro/regime labels | `data/cache/regime_signals_2003_2026.parquet` | 317 rows x 9 columns, monthly index, `regime_*` columns and `dominant_regime`; currently 2000-01-31 to 2026-05-31 | Monthly | Regime context overlay for weekly/daily dashboard dates | Use `data/published/regime_signals.parquet` if cache is unavailable; otherwise skip regime overlay and mark unavailable |
| Optional Notebook 07 artifacts | `results/notebook_07/*` | Existing PCA, clustering, regime, and inventory artifacts | Static research artifacts | Provenance and consistency checks only | 07b should not require executed Notebook 07 outputs unless the artifact is stable and explicitly optional |

The notebook should write an `input_inventory.csv` capturing each selected input path, existence, shape, date range, and role.

## 4. Core methodology

### 4.1 Sampling frequency

07b v1 should compute weekly returns from daily prices and use weekly returns for rolling PCA dislocation. Daily prices can remain available for momentum diagnostics, but the PCA reconstruction should run on weekly data.

Weekly returns reduce microstructure and single-day noise, lower the autocorrelation created by overlapping daily shocks, and align more closely with the JPM-style five-year weekly rolling PCA design. A weekly design also makes the dashboard cadence naturally diagnostic rather than intraday or trading-system oriented.

### 4.2 Rolling PCA reconstruction

At each weekly date `t`:

```text
1. Use returns from [t - window, t].
2. Standardize assets using only that window.
3. Fit PCA on the standardized window.
4. Retain K components.
5. Reconstruct the latest standardized return vector at t.
6. Compute residual = observed standardized return - reconstructed standardized return.
```

Recommended v1 choices:

- Window: five years of weekly observations, approximately 260 weeks, when enough complete data are available.
- Component count: fixed `K=5`.
- Standardization: fit mean and standard deviation inside each rolling window only.
- Reconstruction: transform the latest standardized vector into the retained PCA space, inverse-transform using the retained components, and compare against the observed standardized vector.
- No-look-ahead rule: the result dated `t` may use data through `t` only. Any later trailing residual z-score for `t` must use residuals observed no later than `t`.

Fixed `K=5` is preferred for v1 because it is easier to compare through time and avoids time-varying threshold artifacts. The notebook can still report explained variance for `K=3`, `K=5`, and the number of components required to reach 80% variance as diagnostics. A variance-threshold `K` can be revisited later if it materially improves interpretability.

### 4.3 PCA dislocation score

The v1 score should use trailing residual standardization:

```text
pca_dislocation_z(asset, t) =
    residual(asset, t) / trailing_std(residual(asset), lookback=156 weeks)
```

Use a minimum observation requirement, for example 52 weekly residuals, before reporting a z-score. If the trailing residual standard deviation is zero or unavailable, set the score to missing and flag it in diagnostics.

Sign convention:

- Positive `pca_dislocation_z` = asset is richer / stronger than PCA reconstruction.
- Negative `pca_dislocation_z` = asset is cheaper / weaker than PCA reconstruction.

The integrated-residual or Ornstein-Uhlenbeck residual approach from the JPM research should be acknowledged as a future enhancement, not implemented in 07b v1.

### 4.4 Momentum diagnostics

Momentum should be trailing-only and used as confirmation or contradiction, not as a trading signal. Recommended v1 features:

- `mom_4w`
- `mom_13w`
- `mom_26w`
- `mom_52w`

These should be computed from weekly prices or weekly total-return index levels derived from daily prices. For each dashboard date, z-score momentum cross-sectionally by horizon and average the horizon z-scores into a simple `momentum_score`. Keep the individual horizons in `momentum_scores.csv` for auditability.

Momentum buckets can be:

```text
positive_momentum: momentum_score >= 0.5
negative_momentum: momentum_score <= -0.5
neutral_momentum: otherwise
```

The thresholds are descriptive bins only and must not be framed as signal triggers.

### 4.5 Macro/regime overlay

Use existing `dominant_regime` labels from the monthly regime artifact. Align monthly labels to weekly or daily dates by sorting the regime index and forward-filling only after the regime observation date. Do not backfill future regime labels into earlier dates.

In v1, regime should enter as:

- A categorical `dominant_regime` context field.
- A coloring/facet variable in plots.
- Optional regime-conditioned summaries of dislocation and momentum distributions.

Do not create a regime-switching rule. Do not create a macro support score unless it is explicitly transparent and based only on existing regime labels. If a field named `macro_context` is included, it should be a plain label such as `regime_1`, `regime_2`, or `regime_unavailable`, not a proprietary-style support score.

### 4.6 Scorecard construction

The latest cross-sectional scorecard should include:

```text
date
asset
asset_class
pca_dislocation_z
dislocation_bucket
momentum_score
momentum_bucket
dominant_regime
macro_context
diagnostic_label
abs_dislocation_rank
```

Suggested dislocation buckets:

```text
rich: pca_dislocation_z >= 1.0
cheap: pca_dislocation_z <= -1.0
neutral: otherwise
```

Suggested diagnostic labels:

```text
rich_with_positive_momentum
rich_with_negative_momentum
cheap_with_positive_momentum
cheap_with_negative_momentum
neutral_or_mixed
```

Labels are diagnostics only. The notebook should avoid buy/sell language and should not imply that any label is a recommendation.

## 5. Proposed notebook architecture

```text
# 07b — PCA Dislocation and Macro-Enhanced Momentum Dashboard

§0 Setup and relationship to Notebook 07
§1 Load canonical 29-asset panel and relevant artifacts
§2 Weekly return panel and rolling PCA design
§3 Rolling PCA reconstruction residuals
§4 PCA dislocation scores
§5 Momentum diagnostics
§6 Macro/regime overlay
§7 Cross-sectional scorecard
§8 Dashboard-style visualizations
§9 What is missing versus JPM
§10 Research interpretation and limitations
```

### §0 Setup and relationship to Notebook 07

- Purpose: Establish 07b as a diagnostic extension of Notebook 07, not a replacement.
- Main computations: Imports, deterministic `ROOT`, `OUT_DIR = results/notebook_07b`, `FIG_DIR`, `HTML_DIR`, seed, plotting defaults, and package path setup.
- Expected figures/tables: None.
- Expected artifacts: Output directories only when implemented.
- Validation checks: Confirm repo root resolution, canonical paths, and no writes outside `results/notebook_07b/`.

### §1 Load canonical 29-asset panel and relevant artifacts

- Purpose: Load prices, returns, asset metadata, regime labels, and optional Notebook 07 artifacts for consistency checks.
- Main computations: Reindex to `UNIVERSE_29`, inspect missingness, date ranges, and asset-class coverage.
- Expected figures/tables: Input inventory and asset inventory.
- Expected artifacts: `input_inventory.csv`.
- Validation checks: 29 columns, ordered columns, non-empty date range, all assets mapped to an asset class.

### §2 Weekly return panel and rolling PCA design

- Purpose: Convert daily price history into the weekly panel used for rolling PCA.
- Main computations: Resample prices to weekly close, compute weekly returns, choose complete-case or documented missing-history handling, define five-year window and fixed `K=5`.
- Expected figures/tables: Weekly coverage table and rolling-window availability counts.
- Expected artifacts: Optional coverage diagnostics inside `rolling_pca_reconstruction_diagnostics.csv`.
- Validation checks: Weekly dates are monotonic; returns use past/current prices only; no future fills.

### §3 Rolling PCA reconstruction residuals

- Purpose: Estimate rolling PCA reconstruction residuals for each asset.
- Main computations: Window-local standardization, PCA fit, latest-vector reconstruction, residual calculation, explained variance and reconstruction error diagnostics.
- Expected figures/tables: Rolling explained variance and reconstruction error summary.
- Expected artifacts: `pca_reconstruction_residuals.parquet`, `rolling_pca_reconstruction_diagnostics.csv`.
- Validation checks: Each result date uses only its historical window; no global scaler; residual matrix columns match `UNIVERSE_29`.

### §4 PCA dislocation scores

- Purpose: Convert residuals into interpretable rich/cheap z-scores.
- Main computations: Trailing residual volatility, z-score calculation, dislocation buckets, latest absolute-dislocation ranks.
- Expected figures/tables: Latest dislocation table and distribution summary.
- Expected artifacts: `pca_dislocation_scores.csv`.
- Validation checks: Minimum residual history enforced; sign convention documented; missing scores flagged rather than imputed.

### §5 Momentum diagnostics

- Purpose: Add trailing price-action context.
- Main computations: 4w, 13w, 26w, and 52w momentum; cross-sectional z-scores; composite `momentum_score`; momentum buckets.
- Expected figures/tables: Latest momentum ranking and horizon-level diagnostics.
- Expected artifacts: `momentum_scores.csv`.
- Validation checks: Momentum is trailing-only; no centered windows; individual horizons retained.

### §6 Macro/regime overlay

- Purpose: Add existing macro/regime context without creating a new macro model.
- Main computations: Point-in-time alignment of `dominant_regime` to weekly score dates; optional regime-conditioned summaries.
- Expected figures/tables: Regime coverage table and optional regime-conditioned dislocation/momentum summary.
- Expected artifacts: `macro_regime_overlay.csv`.
- Validation checks: No backfilling from future regime dates; missing regime labels remain explicit.

### §7 Cross-sectional scorecard

- Purpose: Combine dislocation, momentum, asset class, and regime context into the latest dashboard table.
- Main computations: Join latest dislocation and momentum records, attach metadata/regime labels, assign buckets and diagnostic labels, rank by absolute dislocation.
- Expected figures/tables: Latest scorecard sorted by `abs_dislocation_rank`.
- Expected artifacts: `pca_dislocation_scorecard.csv`.
- Validation checks: Explicit scorecard date; one row per asset; no hidden positioning proxy; no strategy or allocation columns.

### §8 Dashboard-style visualizations

- Purpose: Produce static dashboard figures mirroring the useful presentation layer of the research.
- Main computations: Plot rolling PCA diagnostics, latest rich/cheap bars, dislocation-vs-momentum scatter, macro/regime heatmap, selected asset time series, and optional asset-class panels.
- Expected figures/tables: Six dashboard figures described in Section 7.
- Expected artifacts: Figure PNGs under `results/notebook_07b/figures/`.
- Validation checks: Axis labels and titles use diagnostic language; color encodings are documented; figures save deterministically.

### §9 What is missing versus JPM

- Purpose: Make implementation gaps explicit.
- Main computations: Build a methodology gap table contrasting JPM-style requirements with repo-native 07b v1.
- Expected figures/tables: Gap table.
- Expected artifacts: `methodology_gap_table.csv`.
- Validation checks: Positioning and proprietary macro scores are marked unavailable, not proxied.

### §10 Research interpretation and limitations

- Purpose: Close with safe interpretation boundaries.
- Main computations: None beyond summarizing dashboard findings.
- Expected figures/tables: Optional compact findings table.
- Expected artifacts: `html/07b_pca_dislocation_macro_momentum_dashboard.html` after notebook export.
- Validation checks: No buy/sell recommendations, no performance claims, and no portfolio construction.

## 6. Proposed artifacts

Use:

```text
results/notebook_07b/
results/notebook_07b/figures/
results/notebook_07b/html/
```

Required artifacts:

```text
input_inventory.csv
rolling_pca_reconstruction_diagnostics.csv
pca_reconstruction_residuals.parquet
pca_dislocation_scores.csv
momentum_scores.csv
macro_regime_overlay.csv
pca_dislocation_scorecard.csv
methodology_gap_table.csv
figures/rolling_pca_reconstruction_diagnostics.png
figures/latest_dislocation_bar.png
figures/dislocation_vs_momentum_scatter.png
figures/macro_regime_scorecard_heatmap.png
figures/selected_asset_dislocation_timeseries.png
html/07b_pca_dislocation_macro_momentum_dashboard.html
```

Optional artifacts:

```text
figures/asset_class_dislocation_panel.png
figures/momentum_horizon_heatmap.png
figures/regime_conditioned_dislocation_summary.png
```

Required artifacts should be deterministic outputs from the notebook. Optional artifacts should be added only if they improve reviewability without changing the notebook into a strategy or app.

## 7. Visualization design

### 1. Rolling PCA explained variance / reconstruction error

- X-axis: Weekly rolling window end date.
- Y-axis: Cumulative explained variance for retained components and reconstruction error, preferably on separate panels.
- Grouping/coloring: Lines for `K=3`, `K=5`, optional 80% component count, and reconstruction RMSE.
- Interpretation: Shows whether the rolling PCA basis is stable enough for residual diagnostics.
- Artifact path: `results/notebook_07b/figures/rolling_pca_reconstruction_diagnostics.png`.

### 2. Latest rich/cheap PCA dislocation bar chart

- X-axis: `pca_dislocation_z`.
- Y-axis: Asset, sorted by score or absolute score.
- Grouping/coloring: Asset class or rich/cheap sign.
- Interpretation: Identifies assets currently richer/stronger or cheaper/weaker than rolling PCA reconstruction.
- Artifact path: `results/notebook_07b/figures/latest_dislocation_bar.png`.

### 3. Dislocation vs momentum scatter

- X-axis: `pca_dislocation_z`.
- Y-axis: `momentum_score`.
- Grouping/coloring: Asset class, with optional marker shape or annotation for `dominant_regime`.
- Interpretation: Separates rich/cheap assets whose trailing momentum confirms or contradicts the dislocation.
- Artifact path: `results/notebook_07b/figures/dislocation_vs_momentum_scatter.png`.

### 4. Macro/regime-colored scorecard heatmap

- X-axis: Diagnostic fields such as dislocation bucket, momentum bucket, and macro context.
- Y-axis: Asset, grouped by asset class.
- Grouping/coloring: Heatmap colors for z-scored values and categorical annotation for `dominant_regime`.
- Interpretation: Provides a compact dashboard view of dislocation, momentum, and regime context.
- Artifact path: `results/notebook_07b/figures/macro_regime_scorecard_heatmap.png`.

### 5. Selected asset dislocation z-score time series

- X-axis: Weekly date.
- Y-axis: `pca_dislocation_z`.
- Grouping/coloring: One line per selected asset, with horizontal reference bands at -2, -1, 0, 1, and 2.
- Interpretation: Shows whether current dislocations are persistent, recent, or mean-reverting historically.
- Artifact path: `results/notebook_07b/figures/selected_asset_dislocation_timeseries.png`.

### 6. Optional asset-class panel view

- X-axis: Asset within each class or date, depending on panel design.
- Y-axis: `pca_dislocation_z` or `momentum_score`.
- Grouping/coloring: Facets by asset class.
- Interpretation: Helps compare dislocations within related groups without implying allocation decisions.
- Artifact path: `results/notebook_07b/figures/asset_class_dislocation_panel.png`.

## 8. Streamlit dashboard design, but no implementation yet

Future app path:

```text
apps/pca_dislocation_dashboard.py
```

Streamlit should come after Notebook 07b artifacts are stable. The app should read static artifacts from `results/notebook_07b/`, expose filters and interactive views only, and should not recompute methodology. The notebook remains the source of methodology, artifact generation, and research interpretation.

Potential components:

- Latest scorecard table.
- Asset-class filter.
- Date selector.
- Rich/cheap dislocation bar chart.
- Dislocation vs momentum scatter.
- Regime filter.
- Selected asset time series.

## 9. Validation checks

- No look-ahead in rolling PCA.
- All standardization fitted inside each rolling window only.
- Momentum trailing-only.
- Macro regime labels aligned without future leakage.
- Scorecard date is explicit.
- No hidden positioning proxy.
- No buy/sell language.
- No portfolio construction.
- No strategy performance metrics.
- Deterministic artifact paths.
- Notebook reruns from repo root.
- Tests pass before and after notebook execution.
- Weekly returns are computed from historical daily prices only.
- Residual z-scores use trailing residual history only.
- Missing regime or residual values are surfaced explicitly.

## 10. Explicit non-goals

- No buy/sell recommendations.
- No allocation engine.
- No SWITCH/VMP overlay.
- No portfolio backtest.
- No transaction-cost claims.
- No Sharpe claims.
- No true JPM positioning replication.
- No RP-PCA / Sharpe-scaled PCA implementation in 07b v1.
- No autoencoder cluster portfolios in 07b v1.
- No Streamlit app in this design task.
- No modifications to Notebook 07 or source code.

## 11. Implementation checklist

- [ ] Confirm branch and clean working tree.
- [ ] Run tests.
- [ ] Create notebook.
- [ ] Create `results/notebook_07b` folders.
- [ ] Load data.
- [ ] Implement weekly returns.
- [ ] Implement rolling PCA.
- [ ] Save residuals and diagnostics.
- [ ] Implement dislocation scores.
- [ ] Implement momentum diagnostics.
- [ ] Align macro regimes.
- [ ] Build scorecard.
- [ ] Save figures/artifacts.
- [ ] Export HTML.
- [ ] Run tests.
- [ ] Inspect git status.

## 12. Final recommendation

Proceed with a separate Notebook 07b. Implement the static notebook first, using weekly rolling PCA dislocation scores, trailing momentum diagnostics, and existing regime labels as a transparent macro context overlay. Do not implement Streamlit until the notebook artifacts stabilize. Defer positioning, RP-PCA / Sharpe-scaled PCA, and autoencoders to later work.
