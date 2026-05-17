"""Insert §16 HP sensitivity diagnostic cells into notebooks/03_ml_strategies.ipynb."""
import json

nb = json.load(open("notebooks/03_ml_strategies.ipynb"))
print(f"Starting cells: {len(nb['cells'])}")


def mk_code(src: str):
    parts = src.lstrip("\n").split("\n")
    lines = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def mk_md(src: str):
    parts = src.lstrip("\n").split("\n")
    lines = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


s16_md = mk_md("""
## §16 Hyperparameter Sensitivity Diagnostic

The defaults used in §4–§6 come from authoritative sources: JPM ML Quant Equity Risk Factors (Feb 2025) for the RF specs (100 trees, max_depth=8, min_samples_leaf=50), Hilpisch §14 for tree-based defaults, and standard daily-return regime for Lasso (α=1e-4). This section confirms whether these defaults are near-optimal on the validation window (~2019-10 → 2022-12-30) or whether a small grid search reveals materially better hyperparameters. Validation metrics: rank IC (cross-sectional Spearman correlation between predicted and forward returns).
""")

s16_lasso = mk_code("""
from sklearn.linear_model import Lasso
from aiam.ml.workflow import apply_standardizer
from aiam.evaluation.ic import information_coefficient

val_mask_long = X_full.index.get_level_values('Date').isin(val_dates)
X_val_long = X_full.loc[val_mask_long, FEATURE_COLS]
y_val_long = y_full.loc[val_mask_long]

lasso_grid = []
for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
    m = Lasso(alpha=alpha, random_state=42, max_iter=10_000)
    m.fit(lasso_strat._X_train, lasso_strat._y_train)
    val_arr = apply_standardizer(X_val_long, lasso_strat._center, lasso_strat._scale, FEATURE_COLS).values
    pred = pd.Series(m.predict(val_arr), index=X_val_long.index, name='pred')
    ic = information_coefficient(pred.unstack('Asset'), y_val_long.unstack('Asset'), method='spearman').mean()
    lasso_grid.append({'alpha': alpha, 'val_IC': round(ic, 5), 'n_nonzero': int((m.coef_ != 0).sum())})

lasso_df = pd.DataFrame(lasso_grid)
print('Lasso HP grid (validation set):')
print(lasso_df.to_string(index=False))
best = lasso_df.loc[lasso_df['val_IC'].idxmax()]
default_ic = lasso_df.loc[lasso_df['alpha'] == 1e-4, 'val_IC'].iloc[0]
print(f'\\nDefault (α=1e-4): val_IC = {default_ic:.5f}')
print(f'Best alpha: {best["alpha"]:.0e}, val_IC = {best["val_IC"]:.5f}, Δ = {best["val_IC"] - default_ic:+.5f}')
""")

s16_rf = mk_code("""
from sklearn.ensemble import RandomForestRegressor

rf_grid = []
for (n_est, depth) in [(50, 6), (100, 8), (200, 10)]:
    m = RandomForestRegressor(n_estimators=n_est, max_depth=depth, min_samples_leaf=50,
                              n_jobs=-1, random_state=42)
    m.fit(rf_strat._X_train, rf_strat._y_train)
    val_arr = apply_standardizer(X_val_long, rf_strat._center, rf_strat._scale, FEATURE_COLS).values
    pred = pd.Series(m.predict(val_arr), index=X_val_long.index)
    ic = information_coefficient(pred.unstack('Asset'), y_val_long.unstack('Asset'), method='spearman').mean()
    rf_grid.append({'n_estimators': n_est, 'max_depth': depth, 'val_IC': round(ic, 5)})

rf_df = pd.DataFrame(rf_grid)
print('Random Forest HP grid (validation set):')
print(rf_df.to_string(index=False))
best = rf_df.loc[rf_df['val_IC'].idxmax()]
default_ic = rf_df.loc[(rf_df['n_estimators'] == 100) & (rf_df['max_depth'] == 8), 'val_IC'].iloc[0]
print(f'\\nDefault (100 trees, depth=8): val_IC = {default_ic:.5f}')
print(f'Best: ({int(best["n_estimators"])}, depth={int(best["max_depth"])}), val_IC = {best["val_IC"]:.5f}, Δ = {best["val_IC"] - default_ic:+.5f}')
""")

s16_xgb = mk_code("""
import xgboost as xgb

xgb_grid = []
for (n_est, lr, depth) in [(200, 0.05, 4), (300, 0.05, 6), (500, 0.03, 6), (300, 0.10, 4)]:
    m = xgb.XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth,
                         early_stopping_rounds=20, random_state=42, n_jobs=-1, tree_method='hist')
    m.fit(xgb_strat._X_train, xgb_strat._y_train,
          eval_set=[(xgb_strat._X_val, xgb_strat._y_val)], verbose=False)
    val_arr = apply_standardizer(X_val_long, xgb_strat._center, xgb_strat._scale, FEATURE_COLS).values
    pred = pd.Series(m.predict(val_arr), index=X_val_long.index)
    ic = information_coefficient(pred.unstack('Asset'), y_val_long.unstack('Asset'), method='spearman').mean()
    xgb_grid.append({'n_est': n_est, 'lr': lr, 'depth': depth,
                     'best_iter': m.best_iteration, 'val_IC': round(ic, 5)})

xgb_df = pd.DataFrame(xgb_grid)
print('XGBoost HP grid (validation set):')
print(xgb_df.to_string(index=False))
best = xgb_df.loc[xgb_df['val_IC'].idxmax()]
default_ic = xgb_df.loc[(xgb_df['n_est']==300)&(xgb_df['lr']==0.05)&(xgb_df['depth']==6), 'val_IC'].iloc[0]
print(f'\\nDefault (300, lr=0.05, depth=6): val_IC = {default_ic:.5f}')
print(f'Best: ({int(best["n_est"])}, lr={best["lr"]}, depth={int(best["depth"])}), val_IC = {best["val_IC"]:.5f}, Δ = {best["val_IC"] - default_ic:+.5f}')
""")

s16_plot = mk_code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Lasso
labels_l = [f'α={a:.0e}' for a in lasso_df['alpha']]
colors_l = [FAMILY_COLORS['ML single-fit'] if a != 1e-4 else '#2ca02c' for a in lasso_df['alpha']]
axes[0].bar(labels_l, lasso_df['val_IC'], color=colors_l, alpha=0.85, edgecolor='white')
axes[0].set_title('Lasso — α sensitivity', fontsize=10)
axes[0].set_ylabel('Validation rank IC', fontsize=9)
axes[0].set_xlabel('alpha (green = default)', fontsize=8)
axes[0].spines[['top', 'right']].set_visible(False)

# RF
labels_r = [f'({n},d{d})' for n, d in zip(rf_df['n_estimators'], rf_df['max_depth'])]
colors_r = ['#2ca02c' if (n == 100 and d == 8) else FAMILY_COLORS['ML single-fit']
            for n, d in zip(rf_df['n_estimators'], rf_df['max_depth'])]
axes[1].bar(labels_r, rf_df['val_IC'], color=colors_r, alpha=0.85, edgecolor='white')
axes[1].set_title('RF — (n_est, depth) sensitivity', fontsize=10)
axes[1].set_xlabel('(n_est, depth) (green = default)', fontsize=8)
axes[1].spines[['top', 'right']].set_visible(False)

# XGB
labels_x = [f'({r["n_est"]},{r["lr"]},{r["depth"]})' for _, r in xgb_df.iterrows()]
colors_x = ['#2ca02c' if (r['n_est']==300 and r['lr']==0.05 and r['depth']==6)
            else FAMILY_COLORS['ML single-fit'] for _, r in xgb_df.iterrows()]
axes[2].bar(labels_x, xgb_df['val_IC'], color=colors_x, alpha=0.85, edgecolor='white')
axes[2].set_title('XGB — (n_est, lr, depth) sensitivity', fontsize=10)
axes[2].set_xlabel('(n_est, lr, depth) (green = default)', fontsize=8)
axes[2].tick_params(axis='x', labelsize=7)
axes[2].spines[['top', 'right']].set_visible(False)

plt.suptitle('Validation-set Rank IC by Hyperparameter Configuration', fontsize=11)
plt.tight_layout()
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'hp_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/hp_sensitivity.png')
""")

# Placeholder summary — will be updated after execution
s16_summary = mk_md("""
**HP Sensitivity Findings** — *[auto-filled after execution]*

- Lasso: TBD
- RF: TBD
- XGBoost: TBD

**Decision:** TBD
""")

# Insert at position 43 (after cell 42 = §15 extended comparison code)
new_cells = [s16_md, s16_lasso, s16_rf, s16_xgb, s16_plot, s16_summary]
for i, cell in enumerate(reversed(new_cells)):
    nb['cells'].insert(43, cell)

print(f"Final cells: {len(nb['cells'])}")
json.dump(nb, open("notebooks/03_ml_strategies.ipynb", "w"), indent=1)
print("Saved.")
