#!/usr/bin/env python
"""Generate 4 publication-quality figures for docs/results.md."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import warnings

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

FAMILY_COLORS = {
    "Classical MV":    "#1f77b4",
    "Constrained MV":  "#17becf",
    "Diversification": "#2ca02c",
    "Regime":          "#9467bd",
    "TSMOM":           "#ff7f0e",
    "Black-Litterman": "#d62728",
    "FF3 Factor":      "#8c564b",
    "Long-Short":      "#bcbd22",
    "EW (benchmark)":  "#333333",
}

# Crisis / stress periods for shading
CRISES = [
    (pd.Timestamp("2003-01-03"), pd.Timestamp("2003-04-30"), "Dot-com\nRecov."),
    (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), "GFC"),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), "COVID"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31"), "Rate\nShock"),
]

OUT = "docs/figures"
import os
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
base = pd.read_parquet("data/cache/portfolio_returns/58strategies_29assets_2003_2026.parquet")
vmp  = pd.read_parquet("data/cache/portfolio_returns/58strategies_vmp_29assets_2003_2026.parquet")
sw_oos = pd.read_parquet("data/cache/portfolio_returns/switch_v2a_oos_29assets.parquet")
regime_sig = pd.read_parquet("data/cache/regime_signals_2003_2026.parquet")["dominant_regime"].dropna()

# Forward-fill monthly regime to daily business-day index
regime_daily = regime_sig.resample("B").ffill().reindex(base.index, method="ffill")

# SWITCH(v2a) — training-only-derived rule (proper OOS version)
switch_v2a = sw_oos["SWITCH_v2a_train_only"].reindex(base.index)
switch_v2a.name = "SWITCH(v2a)"

# ── Helper functions ────────────────────────────────────────────────────────────
def ann_sharpe(rets):
    return rets.mean() / rets.std() * np.sqrt(252)

def max_dd(rets):
    cum = (1 + rets).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def cum_wealth(rets):
    return (1 + rets).cumprod()

# ── Figure 1: Cumulative Wealth ─────────────────────────────────────────────────
print("Figure 1: Cumulative Wealth...")

F1_STRATEGIES = [
    ("EW",                     "EW (benchmark)",          FAMILY_COLORS["EW (benchmark)"],
     dict(lw=2.4, ls="--", zorder=5)),
    ("VMP(MSR(ledoit_wolf))",  "VMP(MSR(LW))",            FAMILY_COLORS["Classical MV"],
     dict(lw=1.8, zorder=4)),
    ("VMP(BL-Mom(LW))",        "VMP(BL-Mom(LW))",         FAMILY_COLORS["Black-Litterman"],
     dict(lw=1.8, zorder=4)),
    (None,                     "SWITCH(v2a)",              FAMILY_COLORS["Regime"],
     dict(lw=1.6, zorder=3)),
    ("BL-Mom(LW)",             "BL-Mom(LW)",              FAMILY_COLORS["Black-Litterman"],
     dict(lw=1.2, ls=":", zorder=3)),
    ("VMP(GMV(sample))",       "VMP(GMV(sample)) [art.]", FAMILY_COLORS["Classical MV"],
     dict(lw=1.2, ls="-.", zorder=2)),
    ("FF3-LowVol",             "FF3-LowVol",              FAMILY_COLORS["FF3 Factor"],
     dict(lw=1.5, zorder=3)),
]

fig, ax = plt.subplots(figsize=(12, 6))

for col, label, color, kw in F1_STRATEGIES:
    if col is None:
        rets = switch_v2a
    elif col in base.columns:
        rets = base[col]
    elif col in vmp.columns:
        rets = vmp[col]
    else:
        print(f"  WARNING: column '{col}' not found, skipping.")
        continue
    cw = cum_wealth(rets)
    ax.semilogy(cw.index, cw.values, color=color, label=label, **kw)

# Shade crises
for start, end, _ in CRISES:
    ax.axvspan(start, end, color="grey", alpha=0.10, zorder=0)

# Crisis text labels
fig.canvas.draw()
ylim = ax.get_ylim()
y_label = np.exp(np.log(ylim[0]) + 0.015 * (np.log(ylim[1]) - np.log(ylim[0])))
for start, end, label in CRISES:
    mid = start + (end - start) / 2
    ax.text(mid, y_label, label, fontsize=6.5, ha="center", va="bottom",
            color="dimgray", style="italic")

# BL-Mom(LW) trough annotation — computed from data, not hardcoded
bl = base["BL-Mom(LW)"]
cum_bl = cum_wealth(bl)
dd_bl = (cum_bl - cum_bl.cummax()) / cum_bl.cummax()
trough_date = dd_bl.idxmin()
trough_val  = cum_bl[trough_date]
maxdd_bl    = dd_bl.min() * 100
ax.annotate(
    f"BL-Mom(LW)\n{maxdd_bl:.2f}% max DD",
    xy=(trough_date, trough_val),
    xytext=(trough_date + pd.Timedelta(days=300), trough_val * 0.62),
    fontsize=7.5,
    color=FAMILY_COLORS["Black-Litterman"],
    arrowprops=dict(arrowstyle="-", color=FAMILY_COLORS["Black-Litterman"], lw=0.8),
    ha="left",
)

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative wealth (log scale, 1 = initial $1)")
ax.set_title("Cumulative Wealth, January 2003 – May 2026", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2g}×"))
ax.legend(ncol=2, loc="upper left", framealpha=0.85)

fig.tight_layout()
fig.savefig(f"{OUT}/cumulative_wealth.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/cumulative_wealth.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved cumulative_wealth.{png,svg}")

# ── Figure 2: Sharpe vs Max Drawdown scatter ───────────────────────────────────
print("Figure 2: Sharpe vs Drawdown scatter...")

FAMILY_MAP = {
    "EW":                  "EW (benchmark)",
    "GMV(sample)":         "Classical MV",
    "GMV(ledoit_wolf)":    "Classical MV",
    "GMV(oas)":            "Classical MV",
    "MSR(sample)":         "Classical MV",
    "MSR(ledoit_wolf)":    "Classical MV",
    "MDP(sample)":         "Diversification",
    "MDP(ledoit_wolf)":    "Diversification",
    "RP(sample)":          "Diversification",
    "RP(ledoit_wolf)":     "Diversification",
    "HRP(sample)":         "Diversification",
    "HRP(ledoit_wolf)":    "Diversification",
    "SWITCH(sample)":      "Regime",
    "SWITCH(ledoit_wolf)": "Regime",
    "TSMOM(12m)":          "TSMOM",
    "TSMOM(6m)":           "TSMOM",
    "BL-Eq(sample)":       "Black-Litterman",
    "BL-Eq(LW)":           "Black-Litterman",
    "BL-Mom(LW)":          "Black-Litterman",
    "BL-Rev(LW)":          "Black-Litterman",
    "FF3-Mom":             "FF3 Factor",
    "FF3-LowVol":          "FF3 Factor",
    "FF3-Quality":         "FF3 Factor",
    "FF3-Multi":           "FF3 Factor",
    "MSR_C(ledoit_wolf)":  "Constrained MV",
    "MSR_C(sample)":       "Constrained MV",
    "MVO_C(ledoit_wolf)":  "Constrained MV",
    "MVO_C(sample)":       "Constrained MV",
    "TSMOM-LS(12m)":       "Long-Short",
    "BL-Mom-LS(LW)":       "Long-Short",
    "FF3-Mom-LS":          "Long-Short",
}

def display_name(col):
    return col.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)")

records = []
for col in base.columns:
    r = base[col]
    records.append({
        "strategy": display_name(col),
        "family":   FAMILY_MAP.get(col, "EW (benchmark)"),
        "sharpe":   ann_sharpe(r),
        "maxdd":    max_dd(r) * 100,
        "is_vmp":   False,
    })
for col in vmp.columns:
    r = vmp[col]
    base_col = col[4:-1]  # strip "VMP(" prefix and ")" suffix
    records.append({
        "strategy": display_name(col),
        "family":   FAMILY_MAP.get(base_col, "EW (benchmark)"),
        "sharpe":   ann_sharpe(r),
        "maxdd":    max_dd(r) * 100,
        "is_vmp":   True,
    })

df_stats = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(11, 7))

for family, grp in df_stats.groupby("family"):
    color = FAMILY_COLORS.get(family, "#aaaaaa")
    base_pts = grp[~grp.is_vmp]
    vmp_pts  = grp[grp.is_vmp]
    if len(base_pts):
        ax.scatter(base_pts.maxdd, base_pts.sharpe,
                   color=color, marker="o", s=50, zorder=4,
                   label=f"{family} (base)", edgecolors=color)
    if len(vmp_pts):
        ax.scatter(vmp_pts.maxdd, vmp_pts.sharpe,
                   color="none", marker="o", s=70, zorder=4,
                   edgecolors=color, linewidths=1.6,
                   label=f"{family} (VMP)")

ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=1)
ax.axvline(-20.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=1)
ax.text(-19.5, df_stats.sharpe.min() * 0.98,
        "Max DD = −20%", fontsize=7, color="dimgray", va="bottom")
ax.text(df_stats.maxdd.max() * 0.5, 1.01, "Sharpe = 1.0",
        fontsize=7, color="dimgray", va="bottom", ha="center")

# Top-5 Sharpe annotations
top5 = df_stats.nlargest(5, "sharpe")
offsets = {
    "VMP(GMV(sample))":        (4, 0.03),
    "VMP(MDP(sample))":        (2, 0.02),
    "VMP(SWITCH(sample))":     (-8, 0.04),
    "VMP(SWITCH(LW))":         (-6, -0.06),
    "VMP(MDP(LW))":            (2, -0.05),
    "VMP(GMV(LW))":            (4, 0.02),
    "MDP(LW)":                 (2, 0.02),
    "GMV(sample)":             (4, -0.04),
}
for _, row in top5.iterrows():
    name = row.strategy
    dx, dy = offsets.get(name, (2, 0.02))
    ax.annotate(
        name,
        xy=(row.maxdd, row.sharpe),
        xytext=(row.maxdd + dx, row.sharpe + dy),
        fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.7, color="gray"),
        ha="left" if dx >= 0 else "right",
    )

ax.set_xlabel("Maximum Drawdown (%)")
ax.set_ylabel("Sharpe Ratio (annualized)")
ax.set_title("Sharpe vs. Maximum Drawdown — All 62 Strategies", fontsize=11, fontweight="bold")

family_patches = [
    mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_COLORS
]
base_marker = plt.Line2D([0], [0], marker="o", color="gray", ls="",
                         markersize=6, markerfacecolor="gray", label="Base strategy")
vmp_marker  = plt.Line2D([0], [0], marker="o", color="gray", ls="",
                         markersize=8, markerfacecolor="none",
                         markeredgewidth=1.6, label="VMP variant")
ax.legend(handles=family_patches + [base_marker, vmp_marker],
          ncol=2, fontsize=7.5, loc="lower right", framealpha=0.85)

fig.tight_layout()
fig.savefig(f"{OUT}/sharpe_vs_drawdown.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/sharpe_vs_drawdown.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved sharpe_vs_drawdown.{png,svg}")

# ── Compute regime-conditional Sharpe and save parquet ─────────────────────────
print("Computing regime-conditional Sharpe (2003-2026)...")

HEATMAP_STRATS = [
    "EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)",
    "MSR(sample)", "MSR(ledoit_wolf)",
    "MDP(sample)", "MDP(ledoit_wolf)",
    "RP(sample)", "RP(ledoit_wolf)",
    "HRP(sample)", "HRP(ledoit_wolf)",
]
HEATMAP_DISPLAY = {
    "EW":              "EW",
    "GMV(sample)":     "GMV(sample)",
    "GMV(ledoit_wolf)":"GMV(LW)",
    "GMV(oas)":        "GMV(OAS)",
    "MSR(sample)":     "MSR(sample)",
    "MSR(ledoit_wolf)":"MSR(LW)",
    "MDP(sample)":     "MDP(sample)",
    "MDP(ledoit_wolf)":"MDP(LW)",
    "RP(sample)":      "RP(sample)",
    "RP(ledoit_wolf)": "RP(LW)",
    "HRP(sample)":     "HRP(sample)",
    "HRP(ledoit_wolf)":"HRP(LW)",
}

regimes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
hm_data = {}
ndays   = {}

for col in HEATMAP_STRATS:
    r_series = base[col]
    row = {}
    for k in regimes:
        mask = (regime_daily == k).reindex(base.index, fill_value=False)
        sub = r_series[mask]
        row[k] = ann_sharpe(sub) if len(sub) >= 21 else np.nan
    hm_data[HEATMAP_DISPLAY[col]] = row

for k in regimes:
    mask = (regime_daily == k).reindex(base.index, fill_value=False)
    ndays[k] = mask.sum()

df_hm = pd.DataFrame(hm_data).T
df_hm.columns = [int(c) for c in df_hm.columns]
df_hm.to_parquet("data/cache/regime_conditional_sharpe.parquet")
print(f"  → saved regime_conditional_sharpe.parquet  shape={df_hm.shape}")
print(f"  n_days per regime: { {int(k): int(v) for k,v in ndays.items()} }")

# ── Figure 3: Regime Conditional Heatmap ──────────────────────────────────────
print("Figure 3: Regime conditional heatmap...")

REGIME_LABELS = {
    0: "R0\nExpansion",
    1: "R1\nRecovery",
    2: "R2\nNeutral",
    3: "R3\nSlow\nGrowth",
    4: "R4\nStress",
    5: "R5\nLow &\nContracting",
    6: "R6\nCrisis",
    7: "R7\nContraction",
}
SPARSE_THRESHOLD = 252

strat_order = list(HEATMAP_DISPLAY.values())
reg_order   = [0, 1, 2, 3, 4, 5, 6, 7]

Z = np.full((len(strat_order), len(reg_order)), np.nan)
for i, strat in enumerate(strat_order):
    for j, k in enumerate(reg_order):
        if strat in df_hm.index:
            Z[i, j] = df_hm.loc[strat, k]

vmax = min(max(abs(np.nanmin(Z)), abs(np.nanmax(Z))), 3.0)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(Z, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, label="Annualized Sharpe", shrink=0.8)

for i in range(len(strat_order)):
    for j, k in enumerate(reg_order):
        val = Z[i, j]
        n   = ndays[float(k)]
        if np.isnan(val):
            txt = "—"
        elif n < SPARSE_THRESHOLD:
            txt = f"{val:.2f}*"
        else:
            txt = f"{val:.2f}"
        color = "white" if (not np.isnan(val) and abs(val) > vmax * 0.6) else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                color=color if not np.isnan(val) else "dimgray")

# Hatch sparse columns
for j, k in enumerate(reg_order):
    if ndays[float(k)] < SPARSE_THRESHOLD:
        for i in range(len(strat_order)):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  fill=False, hatch="////", edgecolor="white",
                                  linewidth=0, alpha=0.35, zorder=3)
            ax.add_patch(rect)

# Gold borders: training-only-derived v2a selection rule cells
highlight_cells = [
    (strat_order.index("MSR(LW)"),     reg_order.index(0)),  # R0 → MSR(LW)
    (strat_order.index("MSR(sample)"), reg_order.index(5)),  # R5 → MSR(sample)
]
for (row_i, col_j) in highlight_cells:
    rect = FancyBboxPatch(
        (col_j - 0.48, row_i - 0.48), 0.96, 0.96,
        boxstyle="square,pad=0", fill=False,
        edgecolor="gold", linewidth=2.5, zorder=5,
    )
    ax.add_patch(rect)

ax.set_xticks(range(len(reg_order)))
ax.set_xticklabels([REGIME_LABELS[k] for k in reg_order], fontsize=7.5)
ax.set_yticks(range(len(strat_order)))
ax.set_yticklabels(strat_order, fontsize=8)
ax.set_title(
    "Regime-Conditional Annualized Sharpe — 12 Base Strategies × 8 Regimes (2003–2026)\n"
    "(*sparse: n < 252 days; gold border = training-only-derived SWITCH(v2a) selection rule)",
    fontsize=9.5, fontweight="bold",
)

fig.tight_layout()
fig.savefig(f"{OUT}/regime_conditional_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/regime_conditional_heatmap.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved regime_conditional_heatmap.{png,svg}")

# ── Figure 4: VMP Exposure Mechanism ─────────────────────────────────────────
print("Figure 4: VMP exposure mechanism...")

msr_lw = base["MSR(ledoit_wolf)"]
target_vol   = msr_lw.std() * np.sqrt(252)
roll_vol     = msr_lw.rolling(21).std() * np.sqrt(252)
roll_vol_lag = roll_vol.shift(1)
exposure     = (target_vol / roll_vol_lag).clip(0.25, 1.5)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"hspace": 0.08})
ax_vol, ax_exp = axes

ax_vol.plot(roll_vol.index, roll_vol.values * 100,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="21-day realized vol")
ax_vol.axhline(target_vol * 100, color="black", ls="--", lw=1.0,
               label=f"Long-run vol ({target_vol*100:.1f}%)")
for start, end, label in CRISES:
    ax_vol.axvspan(start, end, color="grey", alpha=0.10, zorder=0)
ax_vol.set_ylabel("Realized Vol (%, ann.)")
ax_vol.set_title(
    "MSR(LW): 21-day Realized Volatility and VMP Exposure Multiplier (2003–2026)",
    fontsize=11, fontweight="bold",
)
ax_vol.legend(fontsize=8, loc="upper right")

cap_low  = exposure <= 0.251
cap_high = exposure >= 1.499

ax_exp.plot(exposure.index, exposure.values,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="VMP exposure")
ax_exp.fill_between(exposure.index, 0.25, exposure.values,
                    where=cap_low, color="#d62728", alpha=0.35, label="Vol cap active (0.25×)")
ax_exp.fill_between(exposure.index, exposure.values, 1.5,
                    where=cap_high, color="#2ca02c", alpha=0.35, label="Max exposure (1.5×)")
ax_exp.axhline(0.25, color="#d62728", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.50, color="#2ca02c", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.00, color="black", ls=":", lw=0.7, alpha=0.5)

for start, end, label in CRISES:
    ax_exp.axvspan(start, end, color="grey", alpha=0.10, zorder=0)

fig.canvas.draw()
ylim_e = ax_exp.get_ylim()
for start, end, label in CRISES:
    mid = start + (end - start) / 2
    ax_exp.text(mid, ylim_e[0] + 0.01 * (ylim_e[1] - ylim_e[0]),
                label, fontsize=6.5, ha="center", va="bottom",
                color="dimgray", style="italic")

ax_exp.set_ylabel("Exposure multiplier")
ax_exp.set_xlabel("Date")
ax_exp.legend(fontsize=8, loc="upper right", ncol=2)
ax_exp.set_ylim(0.15, 1.65)

fig.tight_layout()
fig.savefig(f"{OUT}/vmp_exposure_mechanism.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/vmp_exposure_mechanism.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved vmp_exposure_mechanism.{png,svg}")

print("\nAll 4 figures generated successfully.")
print(f"Output directory: {OUT}")
