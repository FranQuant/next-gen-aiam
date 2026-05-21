# 07 — Unsupervised Representation Learning: Design Plan

## 1. Objective and narrative role

Notebook 07 will add an unsupervised structural diagnostic layer after the repository's supervised ML, deep learning, and reinforcement learning sections. Its role is to explain what the canonical 29-asset universe looks like when no return forecast, target label, allocation rule, or policy objective is imposed.

The notebook should use unsupervised learning to describe latent risk structure: correlation geometry, PCA factors, eigenportfolios, asset clusters, and the stability of those representations through time. The narrative should treat these methods as ways to understand the investment universe and diagnose diversification, not as a new alpha model.

Notebook 07 should therefore answer questions such as:

- Which assets move together after standardization?
- How concentrated is cross-asset variance in the first few principal components?
- What economic exposures do the leading PCA loadings resemble?
- Which groups emerge from correlation-distance clustering?
- How stable are PCA factors and clusters through rolling windows?
- Do existing macro regimes help interpret factor behavior without becoming a strategy-selection rule?

## 2. Reference mapping

- Book reference: Chapter 19 — Unsupervised Learning and Representation Learning.
- Repo notebook target: `notebooks/07_unsupervised_representation_learning.ipynb`.
- Companion themes: asset clustering, dimensionality reduction, eigenportfolios, latent factors, and regime interpretation.

The repo numbering differs from the book: Notebook 06 is reserved for reinforcement learning, while Notebook 07 is the unsupervised representation learning notebook. The implementation should rebuild the intellectual content in the repo's current data model and storytelling style instead of copying the older Chapter 19 notebook directly.

Regime detection is already represented elsewhere through `data/cache/regime_signals_2003_2026.parquet`, published regime signals, SWITCH/VMP artifacts, and practitioner analytics. Notebook 07 may use `dominant_regime` only as an optional interpretive overlay, for example to group or color PCA factor returns by existing monthly regimes. It should not infer a new regime system or turn regimes into a hidden strategy selector.

## 3. Data inputs

Primary inputs should come from the canonical 29-asset panel:

- `data/cache/prices_29.parquet`: daily adjusted price panel, 5,869 rows x 29 assets, indexed by daily `date`. This is useful for sanity checks and, if needed, recomputing returns.
- `data/cache/returns_29_2003_2026.parquet`: daily return panel, 5,869 rows x 29 assets, indexed by daily `date`. This should be the primary PCA, correlation, eigenportfolio, and clustering input.
- `data/cache/prices_29_ohlcv_2003_2026.parquet`: long OHLCV panel indexed by `(date, ticker)`. This is not required for first-version PCA, but can document provenance and support future volume-aware diagnostics.
- `data/published/ohlcv_29assets_2003_2026.csv`: published reproduction-grade OHLCV source. Prefer cache parquets for the notebook's first version, with this CSV as provenance or fallback documentation.
- `src/aiam/data/universe.py`: `UNIVERSE_29` gives the canonical asset ordering.
- `src/aiam/features/asset_class.py`: `ASSET_CLASS_MAP` and `asset_class_one_hot()` provide the seven asset-class labels used for ordered heatmaps and group summaries.

Optional regime overlay inputs:

- `data/cache/regime_signals_2003_2026.parquet`: monthly regime signal cache with `regime_*` columns and `dominant_regime`.
- `data/published/regime_signals.parquet`: reproduction-grade monthly regime signals with the same column structure.

The first implementation should load from the repo root, follow the existing notebook pattern of defining `ROOT`, and make all paths deterministic.

## 4. Methods included

First-version methods should be interpretable and reproducible:

- Return correlation matrix using the canonical 29 daily return columns.
- Asset-class ordered correlation heatmap using `ASSET_CLASS_MAP`.
- PCA on an explicitly standardized return matrix, or equivalently on the return correlation matrix, with missing-data handling documented before fitting.
- Explained variance diagnostics, including per-component and cumulative explained variance.
- PCA loadings table by asset and component, with asset-class labels attached.
- Eigenportfolio construction from leading PCA loading vectors.
- Eigenportfolio return series computed as transparent linear combinations of asset returns.
- Cluster diagnostics using correlation distance, preferably hierarchical clustering with dendrogram/order outputs.
- Rolling PCA diagnostics, such as rolling explained variance and/or loading stability against a reference component.
- Optional regime overlay: summarize PCA or eigenportfolio behavior by existing `dominant_regime` after aligning monthly regimes to daily dates without future leakage.

Do not include a neural autoencoder in the first version. Autoencoders, nonlinear embeddings, and latent representation models can be listed as future extensions after the PCA/clustering diagnostic baseline is stable.

## 5. Proposed notebook architecture

### §0 Setup and project context

- Purpose: Establish the notebook as a diagnostic unsupervised representation layer after ML, DL, and RL.
- Main computations: Import libraries, define `ROOT`, import `UNIVERSE_29` and `ASSET_CLASS_MAP`, set `OUT_DIR = ROOT / "results" / "notebook_07"` and `FIG_DIR = OUT_DIR / "figures"`.
- Expected figures/tables: None beyond environment/path printouts.
- Saved outputs: Create output directories only when the notebook is implemented.

### §1 Load canonical 29-asset panel

- Purpose: Load the canonical universe and confirm dimensions, date range, asset order, and missing-history behavior.
- Main computations: Read `data/cache/prices_29.parquet` and `data/cache/returns_29_2003_2026.parquet`; reindex columns to `UNIVERSE_29`; attach asset-class metadata from `ASSET_CLASS_MAP`.
- Expected figures/tables: Asset inventory table with ticker, asset class, first valid return date, last valid return date, and non-null observation count.
- Saved outputs: Optional `asset_inventory.csv` under `results/notebook_07/` if useful for auditability.

### §2 Return matrix and standardization

- Purpose: Make preprocessing choices explicit before any unsupervised fitting.
- Main computations: Select the analysis window; document how assets with shorter histories are handled; compute a standardized return matrix using in-sample mean and standard deviation for full-sample diagnostics, and rolling-window standardization for rolling diagnostics.
- Expected figures/tables: Missingness summary by asset; standardization summary showing mean/std checks.
- Saved outputs: None required in the first version.

### §3 Correlation structure and asset-class map

- Purpose: Show the raw geometry of the 29-asset universe before PCA.
- Main computations: Compute the return correlation matrix; order assets by canonical asset class; optionally compute average within-class and between-class correlations.
- Expected figures/tables: Asset-class ordered correlation heatmap; within/between asset-class correlation table.
- Saved outputs: Figure files under `results/notebook_07/figures/`, for example `correlation_heatmap.png` and optionally `asset_class_correlation_summary.csv`.

### §4 PCA on asset returns

- Purpose: Estimate the leading latent linear factors of the standardized return matrix.
- Main computations: Fit PCA to standardized daily returns after documented missing-data filtering; calculate eigenvalues, explained variance ratio, cumulative explained variance, and loadings.
- Expected figures/tables: Scree plot; cumulative explained variance plot; loadings table for the leading components; top positive/negative loading table by component.
- Saved outputs: `pca_explained_variance.csv`, `pca_loadings.csv`, and figures such as `pca_scree.png` and `pca_cumulative_variance.png`.

### §5 Eigenportfolio construction and interpretation

- Purpose: Translate PCA loading vectors into inspectable factor-mimicking portfolios while avoiding alpha claims.
- Main computations: Convert leading component loadings into signed and/or normalized eigenportfolio weights; compute daily eigenportfolio returns as `returns @ weights`; report gross exposure, net exposure, and sign convention.
- Expected figures/tables: Eigenportfolio weight bar charts by component; eigenportfolio cumulative return charts; table of annualized descriptive statistics.
- Saved outputs: `eigenportfolio_returns.parquet`, optional `eigenportfolio_weights.csv`, and figures under `results/notebook_07/figures/`.

### §6 Clustering as diversification map

- Purpose: Present asset clusters as a diversification map rather than an allocation rule.
- Main computations: Convert correlations to distances, for example `sqrt(0.5 * (1 - corr))`; run hierarchical clustering; choose a small, documented number of clusters or distance threshold for interpretation.
- Expected figures/tables: Dendrogram; clustered correlation heatmap; cluster label table with ticker and asset class.
- Saved outputs: `cluster_labels.csv`, `clustered_correlation_heatmap.png`, and `asset_dendrogram.png`.

### §7 Rolling PCA stability diagnostics

- Purpose: Show whether the latent structure is stable or sample-dependent through time.
- Main computations: Fit PCA over rolling historical windows only; track leading explained variance, cumulative variance for the first `k` components, and loading similarity to a reference vector using sign-adjusted cosine similarity or correlation.
- Expected figures/tables: Rolling explained-variance chart; rolling loading-stability chart; optional heatmap of component/asset loading drift.
- Saved outputs: `rolling_pca_diagnostics.csv` and figures such as `rolling_pca_explained_variance.png` and `rolling_pca_loading_stability.png`.

### §8 Optional: PCA factors by existing macro regime

- Purpose: Use existing macro regimes as an interpretive overlay for factor behavior.
- Main computations: Load `data/cache/regime_signals_2003_2026.parquet`; align monthly `dominant_regime` to daily dates using only known month-end classifications; join to eigenportfolio returns; summarize mean, volatility, count, and distribution by regime.
- Expected figures/tables: Boxplot or violin plot of eigenportfolio returns by `dominant_regime`; regime-conditional descriptive table.
- Saved outputs: Optional `pca_factor_regime_summary.csv` and figure files under `results/notebook_07/figures/`.

### §9 Research interpretation and limitations

- Purpose: Close the notebook with careful interpretation and explicit limits.
- Main computations: None beyond summarizing diagnostics.
- Expected figures/tables: Compact findings table separating structural observations from non-claims.
- Saved outputs: Optional `notebook_07_findings.csv` only if the implementation benefits from a structured summary.

## 6. Outputs and artifact policy

Notebook 07 should write research artifacts under:

```text
results/notebook_07/
```

Suggested first-version outputs:

- `results/notebook_07/pca_loadings.csv`
- `results/notebook_07/pca_explained_variance.csv`
- `results/notebook_07/eigenportfolio_returns.parquet`
- `results/notebook_07/eigenportfolio_weights.csv`
- `results/notebook_07/cluster_labels.csv`
- `results/notebook_07/rolling_pca_diagnostics.csv`
- `results/notebook_07/pca_factor_regime_summary.csv`, only if the optional regime overlay is implemented
- `results/notebook_07/figures/correlation_heatmap.png`
- `results/notebook_07/figures/pca_scree.png`
- `results/notebook_07/figures/pca_cumulative_variance.png`
- `results/notebook_07/figures/eigenportfolio_weights_pc1.png`
- `results/notebook_07/figures/eigenportfolio_returns.png`
- `results/notebook_07/figures/asset_dendrogram.png`
- `results/notebook_07/figures/clustered_correlation_heatmap.png`
- `results/notebook_07/figures/rolling_pca_explained_variance.png`
- `results/notebook_07/figures/rolling_pca_loading_stability.png`

Do not write first-version Notebook 07 outputs to `data/cache/` unless a later implementation decision establishes a strong repo convention for reusable cache artifacts. The outputs should be treated as notebook research artifacts, closer to the newer `results/notebook_05/` convention than the core strategy cache.

## 7. Validation checks

Implementation should include explicit checks for:

- No forward-return labels, supervised targets, or future realized returns used in PCA or clustering fits.
- PCA/clustering fitted only on the intended historical sample, and rolling diagnostics fitted only on data available through each rolling window end date.
- Standardization logic is explicit: full-sample diagnostics may use full-sample standardization if labeled descriptive; rolling diagnostics must standardize within each historical window.
- Missing data handling is documented before fitting, including shorter histories for assets such as `GOOGL.US`, `FXI.US`, `GLD.US`, `DBC.US`, `USO.US`, and `HYG.US`.
- Eigenportfolio weights have documented normalization, gross exposure, net exposure, and sign convention.
- PCA sign indeterminacy is handled when comparing components across rolling windows.
- Rolling diagnostics avoid look-ahead by using historical windows and writing each result at the window end date.
- Optional regime overlay aligns monthly regimes to daily dates without future leakage, preferably by using known month-end regime labels and forward-filling only after the monthly observation date.
- Notebook can be rerun from the repo root and, if opened from `notebooks/`, still resolves `ROOT` correctly.
- Artifact paths are deterministic and contained under `results/notebook_07/`.
- Re-running the notebook overwrites or updates only Notebook 07 artifacts.

## 8. Risks and caveats

- PCA is sample-dependent; leading components can change as the estimation window changes.
- PCA signs are indeterminate, so positive/negative loadings must be interpreted after choosing a documented sign convention.
- Eigenportfolios are diagnostic factor portfolios, not automatically tradable or cost-aware allocation strategies.
- Cluster membership can be unstable across windows, linkage choices, and distance definitions.
- Correlation structure changes through time, especially around crises and rate shocks.
- Unsupervised representations describe structure; they are not evidence of predictive alpha.
- Regime overlays are descriptive only. They must not become hidden strategy selection, SWITCH replacement, VMP timing, or a new allocation rule.
- Full-sample visual diagnostics are acceptable when clearly labeled descriptive, but any stability or time-varying claim should use historical windows.

## 9. Explicit non-goals

- No new SWITCH strategy.
- No new allocation engine.
- No VMP overlay.
- No supervised target construction.
- No OOS performance claim.
- No promotion to `src/aiam/unsupervised/` yet.
- No README updates yet.
- No modifications to Notebook 05.
- No modifications to Notebook 06 or reinforcement learning work.
- No modifications to package source code under `src/aiam/` in the planning step.

## 10. Implementation checklist

- Confirm branch is `unsupervised/representation-learning`.
- Confirm Python resolves to the repo `.venv`.
- Run `PYTHONPATH=src python -m pytest -q`.
- Create `notebooks/07_unsupervised_representation_learning.ipynb`.
- Create `results/notebook_07/` and `results/notebook_07/figures/`.
- Implement §0 through §9 in order.
- Save deterministic CSV, parquet, and figure artifacts under `results/notebook_07/`.
- Export HTML for review.
- Run `PYTHONPATH=src python -m pytest -q`.
- Inspect `git status --short` and confirm only intended Notebook 07 files/artifacts changed.
