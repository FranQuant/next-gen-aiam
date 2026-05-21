# PCA Dislocation Dashboard

Run from the repository root:

```bash
python apps/pca_dislocation_dashboard.py
```

Or:

```bash
python -m apps.pca_dislocation_dashboard
```

The dashboard requires `dash`, `plotly`, and `pandas`.

Notebook 07b generates the deterministic artifacts read by the app:

- `results/notebook_07b/pca_dislocation_scorecard.csv`
- `results/notebook_07b/pca_dislocation_scores.csv`
- `results/notebook_07b/momentum_scores.csv`
- `results/notebook_07b/macro_regime_overlay.csv`
- `results/notebook_07b/rolling_pca_reconstruction_diagnostics.csv`
- `results/notebook_07b/methodology_gap_table.csv`

The app is an interactive viewer only. It does not recompute PCA, momentum,
regime labels, diagnostic labels, or methodology outputs.
