"""Add 9 visualization cells to notebooks/03_ml_strategies.ipynb."""
import json
from pathlib import Path

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


# ── Color/family mapping (insert after cell 11 — splits) ──────────────────
color_family_cell = mk_code("""
STRATEGY_FAMILY = {
    'EW': 'Classical', 'MSR(LW)': 'Classical', 'SWITCH(v2a)': 'Classical', 'VMP(MDP(LW))': 'Classical',
    'SignalTilt(mom_252)': 'Classical',
    'SignalTilt(Lasso)': 'ML single-fit', 'SignalTilt(RF)': 'ML single-fit', 'SignalTilt(XGB)': 'ML single-fit',
    'MSR(Lasso_μ̂)': 'ML single-fit', 'MSR(RF_μ̂)': 'ML single-fit', 'MSR(XGB_μ̂)': 'ML single-fit',
    'VMP(SignalTilt(Lasso))': 'ML + VMP', 'VMP(SignalTilt(RF))': 'ML + VMP', 'VMP(SignalTilt(XGB))': 'ML + VMP',
    'VMP(MSR(Lasso_μ̂))': 'ML + VMP', 'VMP(MSR(RF_μ̂))': 'ML + VMP', 'VMP(MSR(XGB_μ̂))': 'ML + VMP',
    'SignalTilt(Ensemble)': 'ML ensemble', 'MSR(Ensemble_μ̂)': 'ML ensemble',
}
FAMILY_COLORS = {
    'Classical': '#1f4e79', 'ML single-fit': '#d62728',
    'ML + VMP': '#ff7f0e', 'ML ensemble': '#2ca02c',
}
print('Strategy families:', {v: sum(1 for x in STRATEGY_FAMILY.values() if x == v) for v in FAMILY_COLORS})
""")

# ── Plot 6 — Feature correlation heatmap (insert after cell 6 — feature computation) ──
plot6_cell = mk_code("""
fig, ax = plt.subplots(figsize=(13, 11))
corr = feature_panel[FEATURE_COLS].dropna().corr().values
labels = FEATURE_COLS
cmap = plt.cm.RdBu_r
im = ax.pcolormesh(corr, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(labels)) + 0.5)
ax.set_yticks(np.arange(len(labels)) + 0.5)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr[i, j]
        ax.text(j + 0.5, i + 0.5, f'{v:.2f}', ha='center', va='center',
                fontsize=4.5, color='white' if abs(v) > 0.6 else 'black')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
ax.set_title(f'Feature correlation matrix ({len(labels)} features)', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/feature_correlation.png')
""")

# ── Plot 5 — Predicted vs Realized scatter, 1×3 (insert after cell 17 — XGB training) ──
plot5_cell = mk_code("""
from scipy.stats import spearmanr

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
rng = np.random.default_rng(42)
for ax, (name, strat) in zip(axes, [('Lasso', lasso_strat), ('RF', rf_strat), ('XGB', xgb_strat)]):
    test_mask_p = strat.predictions.index.get_level_values('Date') >= pd.Timestamp(TEST_START)
    pred_test = strat.predictions[test_mask_p]
    real_test = y_full.reindex(pred_test.index).dropna()
    pred_aligned = pred_test.reindex(real_test.index)
    rank_ic = spearmanr(pred_aligned.values, real_test.values)[0]
    if len(real_test) > 5000:
        idx = rng.choice(len(real_test), 5000, replace=False)
        p, r = pred_aligned.iloc[idx], real_test.iloc[idx]
    else:
        p, r = pred_aligned, real_test
    color = FAMILY_COLORS['ML single-fit']
    ax.scatter(p.values, r.values, alpha=0.15, s=3, color=color, rasterized=True)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('Predicted z-score', fontsize=9)
    ax.set_title(f'{name}  Rank-IC={rank_ic:.3f}', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Realized 21d return', fontsize=9)
plt.suptitle('Predicted vs Realized (test period 2023–2026, 5000 samples each)', fontsize=11)
plt.tight_layout()
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'predicted_vs_realized.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/predicted_vs_realized.png')
""")

# ── Plot 7 — Perm importance with error bars, n_repeats=10 (REPLACE cell 19) ──
plot7_cell = mk_code("""
from sklearn.inspection import permutation_importance as _pi

t0 = time.time()
_pi_res = _pi(rf_strat.model, rf_strat._X_val, rf_strat._y_val, n_repeats=10, random_state=42)
rf_importance = pd.Series(_pi_res.importances_mean, index=rf_strat._feature_cols)
imp_std = pd.Series(_pi_res.importances_std, index=rf_strat._feature_cols)
print(f'Permutation importance (n_repeats=10): {time.time()-t0:.1f}s')

top15_idx = rf_importance.sort_values().tail(15).index
top15_mean = rf_importance.reindex(top15_idx)
top15_std = imp_std.reindex(top15_idx)
colors = [FAMILY_COLORS['ML single-fit'] if v >= 0 else '#aaaaaa' for v in top15_mean.values]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(top15_mean)), top15_mean.values, xerr=top15_std.values,
        color=colors, ecolor='#444444', capsize=3, alpha=0.85)
ax.set_yticks(range(len(top15_mean)))
ax.set_yticklabels(top15_mean.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.7)
ax.set_xlabel('Mean Importance ± σ (validation set, n_repeats=10)')
ax.set_title('Random Forest — Permutation Importance with Error Bars')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'rf_permutation_importance.png', dpi=150, bbox_inches='tight')
fig.savefig(ROOT / 'docs/figures/rf_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/rf_permutation_importance.png')
print('\\nTop 5 features by RF permutation importance:')
print(rf_importance.sort_values(ascending=False).head(5).to_string())
""")

# ── Plot 8 — Weight evolution by asset class (insert after cell 22 — Approach B) ──
plot8_cell = mk_code("""
from aiam.features.asset_class import ASSET_CLASS_MAP as _ACM

_CLASSES = ['us_single_stock', 'us_sector_etf', 'broad_equity_etf', 'intl_equity_etf',
            'fixed_income_etf', 'commodity_etf', 'fx_spot']
_CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']

sample_dates = eval_dates[::21]
wt_records = []
for d in sample_dates:
    w = xgb_strat.predict_weights(panel, d)
    row = {cls: 0.0 for cls in _CLASSES}
    for asset, wt in w.items():
        cls = _ACM.get(asset, 'us_single_stock')
        row[cls] = row.get(cls, 0.0) + float(wt)
    row['Date'] = d
    wt_records.append(row)
wt_df = pd.DataFrame(wt_records).set_index('Date')[_CLASSES]

fig, ax = plt.subplots(figsize=(11, 4))
ax.stackplot(wt_df.index, wt_df.values.T, labels=_CLASSES, colors=_CLASS_COLORS, alpha=0.85)
ax.set_ylabel('Portfolio weight', fontsize=10)
ax.set_title('SignalTilt(XGB) — Asset class allocation (monthly sample, test period 2023–2026)', fontsize=11)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.set_ylim(0, 1)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'weight_evolution_xgb.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/weight_evolution_xgb.png')
""")

# ── Viz setup: comparison_returns dict (append after cell 38) ──────────────
viz_setup_cell = mk_code("""
comparison_returns = {
    'EW': ret_ew,
    'SignalTilt(mom_252)': ret_tilt_mom,
    'SignalTilt(Lasso)': ret_lasso,
    'SignalTilt(RF)': ret_rf,
    'SignalTilt(XGB)': ret_xgb,
    'MSR(Lasso_μ̂)': ret_msr_lasso,
    'MSR(RF_μ̂)': ret_msr_rf,
    'MSR(XGB_μ̂)': ret_msr_xgb,
    'MSR(LW)': msr_lw_test,
    'SWITCH(v2a)': sw_test,
    'VMP(MDP(LW))': vmp_mdp_test,
    **vmp_returns,
    'SignalTilt(Ensemble)': ret_signaltilt_ens,
    'MSR(Ensemble_μ̂)': ret_msr_ens,
}
fig_dir = ROOT / 'docs/figures/session2'
fig_dir.mkdir(parents=True, exist_ok=True)
print(f'comparison_returns: {len(comparison_returns)} strategies')
print(f'Session 2 figures dir: {fig_dir}')
""")

# ── Plot 1 — Cumulative wealth (log y), top 6 strategies ──────────────────
plot1_cell = mk_code("""
top6 = extended.head(6).index.tolist()
ls_cycle = {fam: iter([('-', 2.0), ('--', 1.6), (':', 1.4)]) for fam in FAMILY_COLORS}

fig, ax = plt.subplots(figsize=(11, 5))
for strat in top6:
    r = comparison_returns.get(strat, pd.Series(dtype=float)).dropna()
    if len(r) == 0:
        continue
    wealth = (1 + r).cumprod()
    fam = STRATEGY_FAMILY.get(strat, 'Classical')
    color = FAMILY_COLORS[fam]
    try:
        ls, lw = next(ls_cycle[fam])
    except StopIteration:
        ls, lw = '-', 1.2
    ax.plot(wealth.index, wealth.values, label=strat, color=color, ls=ls, lw=lw)
ax.set_yscale('log')
ax.set_ylabel('Cumulative wealth (log, $1→)', fontsize=10)
ax.set_title('Cumulative wealth, top 6 strategies — test period 2023–2026', fontsize=11)
ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(fig_dir / 'cum_wealth_top6.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/cum_wealth_top6.png')
""")

# ── Plot 2 — Underwater drawdown, top 5 ───────────────────────────────────
plot2_cell = mk_code("""
top5 = extended.head(5).index.tolist()

fig, ax = plt.subplots(figsize=(11, 4))
for strat in top5:
    r = comparison_returns.get(strat, pd.Series(dtype=float)).dropna()
    if len(r) == 0:
        continue
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    fam = STRATEGY_FAMILY.get(strat, 'Classical')
    color = FAMILY_COLORS[fam]
    ax.fill_between(dd.index, dd.values, 0, alpha=0.20, color=color)
    ax.plot(dd.index, dd.values, label=strat, color=color, lw=1.3)
ax.set_ylabel('Drawdown', fontsize=10)
ax.set_title('Underwater drawdown, top 5 strategies — test period 2023–2026', fontsize=11)
ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(fig_dir / 'drawdown_top5.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/drawdown_top5.png')
""")

# ── Plot 3 — Rolling 126-day Sharpe, top 5 ────────────────────────────────
plot3_cell = mk_code("""
top5 = extended.head(5).index.tolist()

fig, ax = plt.subplots(figsize=(11, 4))
for strat in top5:
    r = comparison_returns.get(strat, pd.Series(dtype=float)).dropna()
    if len(r) == 0:
        continue
    roll_sh = r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(TRADING_DAYS)
    fam = STRATEGY_FAMILY.get(strat, 'Classical')
    ax.plot(roll_sh.index, roll_sh.values, label=strat, color=FAMILY_COLORS[fam], lw=1.3)
ax.axhline(0, color='black', lw=0.7, ls='--')
ax.axhline(1, color='gray', lw=0.5, ls=':')
ax.set_ylabel('Rolling 126-day Sharpe', fontsize=10)
ax.set_title('Rolling 126-day Sharpe, top 5 strategies — test period 2023–2026', fontsize=11)
ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(fig_dir / 'rolling_sharpe_top5.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/rolling_sharpe_top5.png')
""")

# ── Plot 4 — Sharpe vs Ann Vol scatter, 19 strategies ─────────────────────
plot4_cell = mk_code("""
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 6))
for strat, row in extended.iterrows():
    fam = STRATEGY_FAMILY.get(strat, 'Classical')
    color = FAMILY_COLORS[fam]
    marker = '*' if strat == 'MSR(Ensemble_μ̂)' else 'o'
    size = 150 if strat == 'MSR(Ensemble_μ̂)' else 55
    ax.scatter(row['Ann Vol'], row['Sharpe'], color=color, marker=marker, s=size, zorder=3, alpha=0.85)
    ax.annotate(strat, (row['Ann Vol'], row['Sharpe']),
                fontsize=5.5, xytext=(3, 3), textcoords='offset points', ha='left')
handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=handles, fontsize=8, loc='upper right')
ax.set_xlabel('Annualized Volatility', fontsize=10)
ax.set_ylabel('Sharpe Ratio', fontsize=10)
ax.set_title('Sharpe vs Volatility — 19 strategies (test period 2023–2026)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(fig_dir / 'sharpe_vs_vol_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/sharpe_vs_vol_scatter.png')
""")

# ── Plot 9 — Calendar-year Sharpe heatmap ─────────────────────────────────
plot9_cell = mk_code("""
years = [2023, 2024, 2025, 2026]
cal_sharpe = {}
for strat, r in comparison_returns.items():
    r = r.dropna()
    cal_sharpe[strat] = {}
    for yr in years:
        r_yr = r[r.index.year == yr]
        if len(r_yr) > 20:
            cal_sharpe[strat][yr] = r_yr.mean() / r_yr.std() * np.sqrt(TRADING_DAYS)
        else:
            cal_sharpe[strat][yr] = np.nan

cal_df = pd.DataFrame(cal_sharpe).T.reindex(extended.index)[years]
n_strats = len(cal_df)

fig, ax = plt.subplots(figsize=(6, max(7, n_strats * 0.45)))
vmin, vmax = -2, 4
im = ax.pcolormesh(cal_df.values, cmap=plt.cm.RdYlGn, vmin=vmin, vmax=vmax)
ax.set_xticks(np.arange(len(years)) + 0.5)
ax.set_xticklabels([str(y) for y in years], fontsize=10)
ax.set_yticks(np.arange(n_strats) + 0.5)
ax.set_yticklabels(cal_df.index, fontsize=8)
for i in range(n_strats):
    for j in range(len(years)):
        val = cal_df.values[i, j]
        if not np.isnan(val):
            text_color = 'black' if -1.5 < val < 3.5 else 'white'
            ax.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=text_color)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label='Sharpe')
ax.set_title('Calendar-year Sharpe (sorted by overall Sharpe)', fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(fig_dir / 'calendar_year_sharpe.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: docs/figures/session2/calendar_year_sharpe.png')
""")


# ── Apply insertions in REVERSE index order ────────────────────────────────

# 1. Extend with 6 cells after cell 38 (§15 extended comparison code)
nb['cells'].extend([viz_setup_cell, plot1_cell, plot2_cell, plot3_cell, plot4_cell, plot9_cell])

# 2. Insert plot8 after cell 22 (Approach B walk-forward code)
nb['cells'].insert(23, plot8_cell)

# 3. Replace cell 19 (permutation importance plot) with error-bar version
nb['cells'][19] = plot7_cell

# 4. Insert plot5 after cell 17 (XGB training)
nb['cells'].insert(18, plot5_cell)

# 5. Insert color_family after cell 11 (splits)
nb['cells'].insert(12, color_family_cell)

# 6. Insert plot6 after cell 6 (feature computation)
nb['cells'].insert(7, plot6_cell)

print(f"Final cells: {len(nb['cells'])}")
json.dump(nb, open("notebooks/03_ml_strategies.ipynb", "w"), indent=1)
print("Saved.")
