# PCA Dislocation and Macro-Enhanced Momentum - Research Review

## 1. Current project baseline

Notebook 07 is now the baseline unsupervised representation learning notebook for the canonical 29-asset universe. It implements correlation structure, PCA on standardized asset returns, component interpretation, eigenportfolio-style diagnostic portfolios, hierarchical correlation-distance clustering, rolling PCA stability diagnostics, PCA factor behavior by existing `dominant_regime`, and a reference-style snapshot clustering appendix.

It should remain a baseline representation notebook because its role is structural: it explains how the investment universe co-moves when no forecast target, allocation rule, policy objective, or trade recommendation is imposed. Its PCA and clustering outputs are descriptive diagnostics rather than tactical signals.

Tactical PCA dislocation work should be separated into Notebook 07b because it changes the research question. Instead of asking "what is the latent structure of the 29-asset universe?", 07b asks "which assets are currently rich or cheap versus a rolling PCA reconstruction, and do momentum and macro/regime context confirm or contradict that dislocation?" Keeping the notebooks separate preserves the clean interpretation boundary:

```text
07  = unsupervised representation learning baseline
07b = PCA dislocation + momentum + macro/regime diagnostic dashboard
```

## 2. Report inventory

| Report / paper | Local filename if found | Main idea | Relevant to Notebook 07b? | Requires new data? | Implementation priority |
| --- | --- | --- | --- | --- | --- |
| Risk Premia Portfolio Construction via Uncorrelated Clusters Introducing Autoencoders as an extension of Principal Components | `/Users/frasagui/Desktop/JPM_Risk_Premia_Portfoli_2021-01-28_3626916.pdf` | Linear autoencoders can form constrained, low-correlation cluster portfolios as a flexible extension of PCA and hierarchical clustering. | Indirectly; useful for future nonlinear or constrained cluster research, not 07b v1. | Yes for direct replication: JPM risk premia strategy panel and portfolio constraints. | Low for 07b; high for later `07d_autoencoder_cluster_portfolios.ipynb`. |
| Simple lines intertwining: PCA Dislocations, Positioning, and Momentum | `/Users/frasagui/Desktop/JPM_Cross_Asset_Systemat_2025-05-01_4972387.pdf` | Rolling PCA residual dislocations are mean-reversion candidates; momentum and positioning act as confirmation/timing/crowding layers. | Yes; core 07b inspiration. | Yes for true positioning and broad 100+ index cross-asset universe; no for simplified dislocation and momentum. | High. |
| PCA Dislocations, Positioning, and *new* Macro-enhanced Momentum | `/Users/frasagui/Desktop/JPM_Cross_Asset_Systemat_2025-07-10_5027167.pdf` | Adds macro support scores to momentum, especially when price action and macro conditions disagree. | Yes; core 07b macro/regime overlay inspiration. | Yes for Macrosynergy-style point-in-time macro factors; no for simplified regime proxy. | High. |
| Less Is More: Factor Investing by PCA and Its Variants | `/Users/frasagui/Desktop/JPM_Cross_Asset_Systemat_2025-10-29_5113985.pdf` | Compares standard PCA with RP-PCA and Sharpe-scaled PCA, incorporating first- and second-moment information for factor investing. | Partly; belongs mainly to a later PCA variants notebook. | Partly: can prototype on 29 assets, but robust claims need broader histories and validation. | Medium later; low for 07b v1. |
| Tactical Trifecta: PCA Dislocations, Positioning, and Momentum *NEW* Dashboard | `/Users/frasagui/Desktop/JPM_Cross_Asset_Systemat_2026-01-08_5169477.pdf` | Turns PCA dislocation, positioning, and macro-enhanced momentum into a weekly dashboard/scorecard with scatter and ranking views. | Yes; best template for 07b artifact and dashboard shape. | Yes for positioning; no for static scorecard and repo-native dashboard visuals. | High. |
| Scaled Factor Portfolio | `/Users/frasagui/Desktop/ssrn-5392547.pdf` | Sharpe-scaled PCA weights factor inputs by Sharpe ratio before extracting PCs for high-dimensional factor investing. | Indirectly; supports later PCA variants discussion. | Partly: toy prototype possible; high-dimensional factor investing claims need a factor zoo. | Medium later; low for 07b v1. |
| Lettau-Pelger Risk-Premium PCA / Factors that Fit the Time Series and Cross-Section of Stock Returns | `/Users/frasagui/Desktop/ssrn-3211106.pdf` | RP-PCA generalizes PCA by penalizing pricing errors and can find weak high-Sharpe factors standard PCA misses. | Indirectly; important for later RP-PCA design. | Partly: toy implementation possible; robust asset-pricing use needs larger cross-sections. | Medium later; low for 07b v1. |

## 3. Key ideas by report

### 3.1 Risk Premia Portfolio Construction via Uncorrelated Clusters / Autoencoders

The JPM autoencoder report treats autoencoders as a flexible extension of principal components for portfolio construction. The key implementation idea is not a generic nonlinear black box: the report emphasizes linear autoencoders, long-only constraints, out-of-sample low-correlation cluster objectives, and regularization for robustness.

This is valuable because Notebook 07 already has PCA and hierarchical clusters, but neither imposes portfolio constraints nor explicitly optimizes cluster portfolios for low realized correlation. Autoencoders could eventually bridge representation learning and implementable cluster baskets. That is not the right first step for 07b, because 07b should be a tactical dislocation dashboard, not a new portfolio construction engine. The likely future home is `07d_autoencoder_cluster_portfolios.ipynb`.

### 3.2 Simple lines intertwining: PCA Dislocations, Positioning, and Momentum

The May 2025 JPM report frames PCA dislocations as mean-reversion residuals. It uses a broad cross-asset panel, rolling five-year weekly PCA, Lasso selection of relevant PCs for each asset, integrated residuals, and an Ornstein-Uhlenbeck residual model to produce z-scores for rich/cheap conditions.

Momentum is used as timing or confirmation: a dislocation is more interesting when price action is beginning to move in the expected direction. Positioning adds crowding or ownership context, distinguishing stretched-but-supported moves from crowded moves that may be vulnerable to reversal. The strongest version requires all three lenses to agree; two-of-three agreement is treated as weaker but still informative.

With the current repo, we can replicate the spirit but not the full JPM implementation. We can compute rolling PCA reconstruction residuals and momentum scores for the 29-asset panel. We cannot honestly replicate JPM positioning because the repo does not have futures non-commercial positioning, fund betas, surveys, options positioning, or proprietary crowding aggregates.

### 3.3 PCA Dislocations, Positioning, and new Macro-enhanced Momentum

The July 2025 JPM report adds macro-enhanced momentum. The concept is that price momentum should be interpreted together with macro support or headwinds. Macro context helps most when price action and the macro environment disagree, because the disagreement can warn that a trend is less reliable or that a dislocation has a fundamental explanation.

The report uses Macrosynergy-style macro support scores for equities and government bonds, including growth, inflation, liquidity, real carry, labor slack, money/credit, sentiment, and real-estate-related inputs. Those exact inputs are not present in the repo. A simplified local proxy is available: the existing macro-regime engine and `dominant_regime` labels can condition or color momentum diagnostics, provided alignment avoids future leakage. In 07b v1 this should be presented as a regime overlay, not as a claim that the repo has recreated JPM or Macrosynergy macro scores.

### 3.4 Less Is More: Factor Investing by PCA and Its Variants

The October 2025 JPM report compares standard PCA with RP-PCA and Sharpe-scaled PCA. Standard PCA uses second-moment co-movement, while RP-PCA and Sharpe-scaled PCA incorporate first-moment information as well. The report's main claim is that fewer, higher-quality PCs can be more useful for factor investing than a larger orthogonal PCA basis, even if some orthogonality is sacrificed.

This likely belongs in a later 07c rather than 07b v1. Notebook 07b should focus on rolling dislocation, momentum, macro/regime overlays, and dashboard artifacts. PCA variants introduce a different research question: factor extraction for portfolio construction and expected returns. A descriptive appendix in 07b can mention them, but production comparisons should wait.

### 3.5 Tactical Trifecta: PCA Dislocations, Positioning, and Momentum NEW Dashboard

The January 2026 JPM report is the strongest dashboard reference. It describes a weekly dashboard where clients view PCA dislocations, positioning, and macro-enhanced momentum from cross-asset and asset-class perspectives. The visible structure includes a latest cross-asset scorecard, rich/cheap rankings, a dislocation-versus-positioning scatter, and momentum coloring or confirmation.

A simplified repo-native dashboard could include a latest scorecard with asset, asset class, PCA dislocation z-score, trailing momentum, macro/regime context, and a combined diagnostic label such as "cheap with positive momentum" or "expensive with negative momentum". It should replace the positioning axis with momentum or explicitly leave positioning as missing. Suitable first figures are a dislocation bar chart, dislocation-vs-momentum scatter colored by regime or asset class, scorecard heatmap, and selected asset dislocation time series.

### 3.6 Scaled Factor Portfolio

The Scaled Factor Portfolio paper supports the Sharpe-scaled PCA idea. It scales factors by their Sharpe ratios before extracting common PCs, so the resulting components use both co-movement and expected-return information. This is relevant to factor zoo and high-dimensional factor investing problems where ordinary PCA may emphasize high-variance but low-premium directions.

For this repo, it supports a later PCA variants section, not immediate 07b work. A 29-asset ETF/single-stock panel can demonstrate mechanics, but it is too small and mixed to support strong factor-zoo conclusions. It also raises estimation and validation risks because Sharpe scaling uses first moments that are noisy and easy to overfit.

### 3.7 Lettau-Pelger RP-PCA

Lettau and Pelger's RP-PCA incorporates pricing-error and expected-return information into PCA-style factor estimation. It is designed to identify factors that explain both time-series co-movement and cross-sectional expected returns. A key motivation is that standard PCA can miss weak high-Sharpe factors because it focuses on variance explained.

The relevance for this project is portfolio construction and factor modeling, not the first 07b dashboard. RP-PCA should be treated as a later enhancement with explicit train/validation/OOS discipline, because expected-return information can leak into tactical claims if not handled carefully.

## 4. What can be implemented now with the existing repo

Assumed current repo assets:

- canonical 29-asset prices and returns
- asset metadata and asset-class labels
- existing macro-regime and `dominant_regime` labels
- Notebook 07 PCA, clustering, rolling PCA, regime, and snapshot artifacts
- no genuine JPM-style positioning dataset

### Implementable now

- Rolling PCA reconstruction residuals fitted only on historical windows.
- Asset-level PCA dislocation z-scores from residual history.
- Trailing-only momentum scores, such as 21-day, 63-day, 126-day, and 252-day returns or volatility-adjusted momentum.
- Macro/regime overlay using existing `dominant_regime` labels aligned without future leakage.
- Dashboard-style scorecard with asset, asset class, dislocation z-score, momentum score, regime label, and diagnostic agreement flags.
- Cross-sectional bar charts for latest rich/cheap dislocation rankings.
- Scatter plot of dislocation versus momentum, colored by macro context or asset class.
- Selected asset time series of dislocation z-score and thresholds.
- Rolling PCA reconstruction diagnostics, including explained variance, reconstruction error, and number of components.

### Partially implementable now

- Macro-enhanced momentum using regime labels as a simplified macro proxy rather than true macro support scores.
- PCA variant comparison if kept descriptive and clearly labeled exploratory.
- RP-PCA or Sharpe-scaled PCA toy prototype, but not a production-ready factor investing module.
- Two-of-three scorecard logic if the third leg is explicitly marked as missing positioning, not silently substituted.

### Requires new data

- True positioning.
- Futures non-commercial positioning.
- Fund beta estimates.
- Survey positioning.
- Option-market positioning.
- ETF flows and crowding proxies.
- Cross-asset derivative signals.
- Macrosynergy-style point-in-time macro support scores.
- Broad JPM-like 100+ index universe across equity indices, rates, curves, breakevens, credit, FX, commodities, and volatility.

## 5. Proposed Notebook 07b architecture

Proposed notebook path:

```text
notebooks/07b_pca_dislocation_macro_momentum_dashboard.ipynb
```

Recommended structure:

```text
# 07b - PCA Dislocation and Macro-Enhanced Momentum Dashboard

Section 0 Setup and relationship to Notebook 07
Section 1 Load canonical 29-asset panel and relevant artifacts
Section 2 Rolling PCA reconstruction model
Section 3 PCA dislocation scores
Section 4 Momentum diagnostics
Section 5 Macro/regime-enhanced momentum overlay
Section 6 Cross-sectional scorecard
Section 7 Dashboard-style visualizations
Section 8 What is missing versus JPM: positioning and richer macro data
Section 9 Research interpretation and limitations
```

Section 0 Setup and relationship to Notebook 07:
Purpose is to define 07b as a tactical diagnostic extension, not a replacement for 07. Main computation is imports, deterministic paths, and output folders. Expected outputs are environment/path printouts only. Artifacts: none beyond directory creation when implemented.

Section 1 Load canonical 29-asset panel and relevant artifacts:
Purpose is to load prices, returns, asset metadata, Notebook 07 artifacts if useful, and existing regime labels. Main computation is asset ordering, date alignment, complete-history diagnostics, and optional weekly/monthly resampling. Expected tables: input inventory and date coverage. Artifacts: `input_inventory.csv` if needed.

Section 2 Rolling PCA reconstruction model:
Purpose is to estimate historical PCA models without look-ahead. Main computation is rolling-window standardization, PCA fit, component selection, reconstruction, and residual calculation at each window end. Expected figures/tables: rolling explained variance and reconstruction error. Artifacts: `rolling_pca_reconstruction_diagnostics.csv`.

Section 3 PCA dislocation scores:
Purpose is to convert residuals into rich/cheap z-scores. Main computation is residual standardization using trailing residual history and latest cross-sectional ranking. Expected figures/tables: latest dislocation table and z-score distributions. Artifacts: `pca_dislocation_scores.csv`.

Section 4 Momentum diagnostics:
Purpose is to compute trailing-only price confirmation. Main computation is multi-horizon trailing returns, volatility-adjusted momentum, or a simple normalized momentum composite. Expected figures/tables: latest momentum ranking and momentum time series. Artifacts: `momentum_scores.csv`.

Section 5 Macro/regime-enhanced momentum overlay:
Purpose is to add macro context using existing regime labels as a simplified proxy. Main computation is point-in-time alignment of `dominant_regime`, regime-conditioned momentum summaries, and optional regime support/penalty mapping kept transparent and descriptive. Expected figures/tables: regime-colored momentum tables and a regime support matrix. Artifacts: `macro_enhanced_momentum_scores.csv`.

Section 6 Cross-sectional scorecard:
Purpose is to combine dislocation, momentum, and macro/regime context into an inspectable dashboard table. Main computation is agreement flags and diagnostic categories without trading recommendations. Expected figures/tables: latest scorecard sorted by absolute dislocation or combined diagnostic score. Artifacts: `pca_dislocation_scorecard.csv`.

Section 7 Dashboard-style visualizations:
Purpose is to mirror the useful structure of the JPM dashboard with repo-native data. Main computation is plotting. Expected figures: latest dislocation bar chart, dislocation-vs-momentum scatter, macro-enhanced scorecard heatmap, selected asset dislocation time series, and rolling PCA explained variance. Artifacts: PNG files under `figures/`.

Section 8 What is missing versus JPM: positioning and richer macro data:
Purpose is to make missing data explicit. Main computation is none, except possibly a gap table. Expected table: available versus unavailable signal inventory. Artifacts: optional `methodology_gap_table.csv`.

Section 9 Research interpretation and limitations:
Purpose is to document safe interpretation. Main computation is none. Expected output is concise findings and limitations text. Artifacts: exported HTML only.

## 6. Proposed artifacts for 07b

Suggested output directories:

```text
results/notebook_07b/
results/notebook_07b/figures/
results/notebook_07b/html/
```

Suggested artifacts:

```text
pca_dislocation_scores.csv
momentum_scores.csv
macro_enhanced_momentum_scores.csv
pca_dislocation_scorecard.csv
rolling_pca_reconstruction_diagnostics.csv
figures/dislocation_bar_latest.png
figures/dislocation_vs_momentum_scatter.png
figures/macro_enhanced_scorecard_heatmap.png
figures/selected_asset_dislocation_timeseries.png
figures/rolling_pca_reconstruction_diagnostics.png
html/07b_pca_dislocation_macro_momentum_dashboard.html
```

## 7. Streamlit / interactive dashboard recommendation

A Streamlit app is worth implementing, but only after Notebook 07b produces stable deterministic artifacts. The notebook should be the source of methodology and artifact generation. Streamlit should only read those artifacts, expose filters, and display tables/figures. The app should not become the source of calculations, model choices, or research claims.

Proposed app path:

```text
apps/pca_dislocation_dashboard.py
```

Possible components:

- latest scorecard table
- asset-class filter
- PCA dislocation bar chart
- dislocation vs momentum scatter
- macro regime filter
- selected asset time series
- rolling PCA explained variance panel

## 8. Methodological safeguards

- No buy/sell recommendations in v1.
- No portfolio construction.
- No backtested strategy claims.
- No imitation of JPM positioning without data.
- No look-ahead in rolling PCA.
- Residuals and dislocations must be computed only from historical windows.
- Macro labels must be aligned without future leakage.
- Momentum must be trailing-only.
- Full-sample diagnostics must be labeled descriptive.
- Standardization inside rolling models must use only each historical window.
- Outputs are diagnostics, not signals for live trading.

## 9. Recommended implementation sequence

```text
Phase 07b-0: research memo and design plan
Phase 07b-1: static notebook with rolling PCA dislocations + momentum + macro overlay
Phase 07b-2: optional Streamlit app reading notebook artifacts
Phase 07b-3: later positioning data integration
Phase 07b-4: later PCA variants / RP-PCA / Sharpe-scaled PCA comparison
Phase 07b-5: later autoencoder cluster portfolio notebook
```

## 10. Final recommendation

Create 07b as a separate notebook. Do not merge this work into Notebook 07. The first version should implement rolling PCA dislocation scores, momentum diagnostics, and a macro/regime overlay using the current 29-asset panel and existing regime labels. Positioning should remain an explicit missing-data placeholder until a genuine positioning dataset is added.

Streamlit should come after deterministic notebook artifacts are stable. PCA variants, RP-PCA, Sharpe-scaled PCA, and autoencoder cluster portfolios should be later notebooks or appendices, not 07b v1.
