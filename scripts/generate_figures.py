#!/usr/bin/env python
"""Generate 4 publication-quality figures for docs/results.md."""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, LogFormatter
import matplotlib.patheffects as pe
from pathlib import Path
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
    "Diversification": "#2ca02c",
    "Regime":          "#9467bd",
    "TSMOM":           "#ff7f0e",
    "Black-Litterman": "#d62728",
    "FF3 Factor":      "#8c564b",
    "EW (benchmark)":  "#333333",
}

# Crisis periods
CRISES = [
    (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), "GFC"),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), "COVID"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31"), "Rate\nShock"),
]

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
base = pd.read_parquet("data/cache/portfolio_returns/24strategies_2008_2026.parquet")
vmp  = pd.read_parquet("data/cache/portfolio_returns/24strategies_vmp_2008_2026.parquet")
regime_sig = pd.read_parquet("data/cache/regime_signals.parquet")["dominant_regime"].dropna()

# Forward-fill monthly regime to daily business-day index
regime_daily = regime_sig.resample("B").ffill()
regime_daily = regime_daily.reindex(base.index, method="ffill")

# ── Build SWITCH(v2a) ──────────────────────────────────────────────────────────
# R0→MSR(LW), R5→MSR(sample), others→MDP(LW)
r = regime_daily.reindex(base.index, method="ffill")
switch_v2a = pd.Series(index=base.index, dtype=float)
switch_v2a[r == 0.0] = base.loc[r == 0.0, "MSR(ledoit_wolf)"]
switch_v2a[r == 5.0] = base.loc[r == 5.0, "MSR(sample)"]
mask_other = ~r.isin([0.0, 5.0])
switch_v2a[mask_other] = base.loc[mask_other, "MDP(ledoit_wolf)"]
switch_v2a.name = "SWITCH(v2a)"

# ── Helper functions ────────────────────────────────────────────────────────────
def ann_sharpe(rets):
    return rets.mean() / rets.std() * np.sqrt(252)

def ann_return(rets):
    return rets.mean() * 252

def ann_vol(rets):
    return rets.std() * np.sqrt(252)

def max_dd(rets):
    cum = (1 + rets).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def cum_wealth(rets):
    return (1 + rets).cumprod()

def shade_crises(ax, alpha=0.10):
    y0, y1 = ax.get_ylim()
    for start, end, label in CRISES:
        ax.axvspan(start, end, color="grey", alpha=alpha, zorder=0)
    # Re-apply label after shading (called again after plot)

def add_crisis_labels(ax, y_frac=0.02, log_scale=False):
    y0, y1 = ax.get_ylim()
    for start, end, label in CRISES:
        mid = start + (end - start) / 2
        if log_scale:
            ypos = np.exp(np.log(y0) + y_frac * (np.log(y1) - np.log(y0)))
        else:
            ypos = y0 + y_frac * (y1 - y0)
        ax.text(mid, ypos, label, fontsize=6.5, ha="center", va="bottom",
                color="dimgray", style="italic")

# ── Figure 1: Cumulative Wealth ─────────────────────────────────────────────────
print("Figure 1: Cumulative Wealth...")

# Display-name mapping (keep short for legend)
F1_STRATEGIES = [
    ("EW",                    "EW (benchmark)",          FAMILY_COLORS["EW (benchmark)"],
     dict(lw=2.4, ls="--", zorder=5)),
    ("VMP(MSR(ledoit_wolf))", "VMP(MSR(LW))",             FAMILY_COLORS["Classical MV"],
     dict(lw=1.8, zorder=4)),
    ("VMP(BL-Mom(LW))",       "VMP(BL-Mom(LW))",         FAMILY_COLORS["Black-Litterman"],
     dict(lw=1.8, zorder=4)),
    (None,                    "SWITCH(v2a)",              FAMILY_COLORS["Regime"],
     dict(lw=1.6, zorder=3)),     # None → use switch_v2a
    ("BL-Mom(LW)",            "BL-Mom(LW)",              FAMILY_COLORS["Black-Litterman"],
     dict(lw=1.2, ls=":", zorder=3)),
    ("VMP(GMV(sample))",      "VMP(GMV(sample)) [art.]", FAMILY_COLORS["Classical MV"],
     dict(lw=1.2, ls="-.", zorder=2)),
    ("FF3-LowVol",            "FF3-LowVol",              FAMILY_COLORS["FF3 Factor"],
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

# Shade crises first, then re-draw (ax.get_ylim is not stable until after first plot)
for start, end, _ in CRISES:
    ax.axvspan(start, end, color="grey", alpha=0.10, zorder=0)

# Crisis text labels — manual y placement after semilogy
ax.set_ylim(bottom=None)
fig.canvas.draw()  # force layout to settle ylim
ylim = ax.get_ylim()
y_label = np.exp(np.log(ylim[0]) + 0.015 * (np.log(ylim[1]) - np.log(ylim[0])))
for start, end, label in CRISES:
    mid = start + (end - start) / 2
    ax.text(mid, y_label, label, fontsize=6.5, ha="center", va="bottom",
            color="dimgray", style="italic")

# BL-Mom(LW) trough annotation
bl = base["BL-Mom(LW)"]
cum_bl = cum_wealth(bl)
dd_bl = (cum_bl - cum_bl.cummax()) / cum_bl.cummax()
trough_date = dd_bl.idxmin()
trough_val = cum_bl[trough_date]
ax.annotate(
    "BL-Mom(LW)\n−50.85% max DD",
    xy=(trough_date, trough_val),
    xytext=(trough_date + pd.Timedelta(days=300), trough_val * 0.62),
    fontsize=7.5,
    color=FAMILY_COLORS["Black-Litterman"],
    arrowprops=dict(arrowstyle="-", color=FAMILY_COLORS["Black-Litterman"], lw=0.8),
    ha="left",
)

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative wealth (log scale, 1 = initial $1)")
ax.set_title("Cumulative Wealth, January 2008 – May 2026", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2g}×"))
ax.legend(ncol=2, loc="upper left", framealpha=0.85)

fig.tight_layout()
fig.savefig(OUT / "cumulative_wealth.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "cumulative_wealth.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved cumulative_wealth.{png,svg}")

# ── Figure 2: Sharpe vs Max Drawdown scatter ───────────────────────────────────
print("Figure 2: Sharpe vs Drawdown scatter...")

# Compute stats for all 48 strategies
FAMILY_MAP = {
    "EW":              "EW (benchmark)",
    "GMV(sample)":     "Classical MV",
    "GMV(ledoit_wolf)":"Classical MV",
    "GMV(oas)":        "Classical MV",
    "MSR(sample)":     "Classical MV",
    "MSR(ledoit_wolf)":"Classical MV",
    "MDP(sample)":     "Diversification",
    "MDP(ledoit_wolf)":"Diversification",
    "RP(sample)":      "Diversification",
    "RP(ledoit_wolf)": "Diversification",
    "HRP(sample)":     "Diversification",
    "HRP(ledoit_wolf)":"Diversification",
    "SWITCH(sample)":  "Regime",
    "SWITCH(ledoit_wolf)": "Regime",
    "TSMOM(12m)":      "TSMOM",
    "TSMOM(6m)":       "TSMOM",
    "BL-Eq(sample)":   "Black-Litterman",
    "BL-Eq(LW)":       "Black-Litterman",
    "BL-Mom(LW)":      "Black-Litterman",
    "BL-Rev(LW)":      "Black-Litterman",
    "FF3-Mom":         "FF3 Factor",
    "FF3-LowVol":      "FF3 Factor",
    "FF3-Quality":     "FF3 Factor",
    "FF3-Multi":       "FF3 Factor",
}

# Display name mapping for annotations
DISPLAY_NAME = {c: c.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)") for c in base.columns}
DISPLAY_NAME.update({c: c.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)") for c in vmp.columns})

records = []
for col in base.columns:
    r = base[col]
    family = FAMILY_MAP.get(col, "EW (benchmark)")
    records.append({
        "strategy": DISPLAY_NAME[col],
        "family":   family,
        "sharpe":   ann_sharpe(r),
        "maxdd":    max_dd(r) * 100,   # in %
        "is_vmp":   False,
    })
for col in vmp.columns:
    r = vmp[col]
    base_col = col.replace("VMP(", "").rstrip(")")
    family = FAMILY_MAP.get(base_col, "EW (benchmark)")
    records.append({
        "strategy": DISPLAY_NAME[col],
        "family":   family,
        "sharpe":   ann_sharpe(r),
        "maxdd":    max_dd(r) * 100,
        "is_vmp":   True,
    })

df_stats = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(11, 7))

# Plot by family
for family, grp in df_stats.groupby("family"):
    color = FAMILY_COLORS[family]
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

# Reference lines
ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=1)
ax.axvline(-20.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=1)
ax.text(-19.5, ax.get_ylim()[0] * 0.99 if ax.get_ylim()[0] > 0 else 0.55,
        "Max DD = −20%", fontsize=7, color="dimgray", va="bottom")
ax.text(df_stats.maxdd.max() * 0.5, 1.01, "Sharpe = 1.0",
        fontsize=7, color="dimgray", va="bottom", ha="center")

# Top-5 Sharpe annotations
top5 = df_stats.nlargest(5, "sharpe")
offsets = {
    "VMP(GMV(sample))":    (4, 0.03),
    "VMP(MDP(sample))":    (2, 0.02),
    "VMP(SWITCH(sample))": (-8, 0.04),
    "VMP(SWITCH(LW))":     (-6, -0.06),
    "VMP(MDP(LW))":        (2, -0.05),
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
ax.set_title("Sharpe vs. Maximum Drawdown — All 48 Strategies", fontsize=11, fontweight="bold")

# Compact legend: families + marker type
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
fig.savefig(OUT / "sharpe_vs_drawdown.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "sharpe_vs_drawdown.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved sharpe_vs_drawdown.{png,svg}")

# ── Compute regime-conditional Sharpe and save parquet ─────────────────────────
print("Computing regime-conditional Sharpe...")

# 12 non-SWITCH base strategies (first 12 cols of 14strategies file)
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
hm_data = {}   # strategy_display → {regime → sharpe}
ndays   = {}   # regime → n_days

for col in HEATMAP_STRATS:
    r_series = base[col]
    row = {}
    for k in regimes:
        mask = (regime_daily == k).reindex(base.index, fill_value=False)
        sub = r_series[mask]
        if len(sub) >= 21:
            row[k] = ann_sharpe(sub)
        else:
            row[k] = np.nan
    hm_data[HEATMAP_DISPLAY[col]] = row

for k in regimes:
    mask = (regime_daily == k).reindex(base.index, fill_value=False)
    ndays[k] = mask.sum()

df_hm = pd.DataFrame(hm_data).T  # shape: regimes × strategies → transpose to strategies × regimes
df_hm.columns = [int(c) for c in df_hm.columns]

# Save parquet
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
SPARSE_THRESHOLD = 252  # trading days

# Build array: rows = strategies, cols = regimes
strat_order = list(HEATMAP_DISPLAY.values())  # 12 strategies
reg_order   = [0, 1, 2, 3, 4, 5, 6, 7]

Z = np.full((len(strat_order), len(reg_order)), np.nan)
for i, strat in enumerate(strat_order):
    for j, k in enumerate(reg_order):
        Z[i, j] = df_hm.loc[strat, k] if strat in df_hm.index else np.nan

vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)))
vmax = min(vmax, 3.0)  # cap for coloring

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(Z, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, label="Annualized Sharpe", shrink=0.8)

# Cell annotations
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
        fontsize = 7
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                color=color if not np.isnan(val) else "dimgray")

# Hatch sparse columns
for j, k in enumerate(reg_order):
    if ndays[float(k)] < SPARSE_THRESHOLD:
        for i in range(len(strat_order)):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, hatch="////", edgecolor="white",
                linewidth=0, alpha=0.35, zorder=3,
            )
            ax.add_patch(rect)

# Highlight SWITCH(v2a) motivating cells: MSR(LW) @ R0, MSR(sample) @ R5
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
    "Regime-Conditional Annualized Sharpe — 12 Base Strategies × 8 Regimes\n"
    "(*sparse: n < 252 days; gold border = SWITCH(v2a) selection rule)",
    fontsize=9.5, fontweight="bold",
)

fig.tight_layout()
fig.savefig(OUT / "regime_conditional_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "regime_conditional_heatmap.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved regime_conditional_heatmap.{png,svg}")

# ── Figure 4: VMP Exposure Mechanism ─────────────────────────────────────────
print("Figure 4: VMP exposure mechanism...")

msr_lw = base["MSR(ledoit_wolf)"]
target_vol = msr_lw.std() * np.sqrt(252)
roll_vol   = msr_lw.rolling(21).std() * np.sqrt(252)
# Lag by 1 day (no lookahead)
roll_vol_lag = roll_vol.shift(1)
exposure    = (target_vol / roll_vol_lag).clip(0.25, 1.5)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"hspace": 0.08})
ax_vol, ax_exp = axes

# ── Top panel: rolling 21-day realized vol ──
ax_vol.plot(roll_vol.index, roll_vol.values * 100,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="21-day realized vol")
ax_vol.axhline(target_vol * 100, color="black", ls="--", lw=1.0, label=f"Long-run vol ({target_vol*100:.1f}%)")
# Crisis shading
for start, end, label in CRISES:
    ax_vol.axvspan(start, end, color="grey", alpha=0.10, zorder=0)
ax_vol.set_ylabel("Realized Vol (%, ann.)")
ax_vol.set_title(
    "MSR(LW): 21-day Realized Volatility and VMP Exposure Multiplier",
    fontsize=11, fontweight="bold",
)
ax_vol.legend(fontsize=8, loc="upper right")

# ── Bottom panel: VMP exposure ──
# Shade where exposure == 0.25 (red) and == 1.5 (green)
cap_low  = exposure <= 0.251
cap_high = exposure >= 1.499

ax_exp.plot(exposure.index, exposure.values,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="VMP exposure")

# Red fill where exposure == lower clip
ax_exp.fill_between(exposure.index, 0.25, exposure.values,
                    where=cap_low, color="#d62728", alpha=0.35, label="Vol cap active (0.25×)")
# Green fill where exposure == upper clip
ax_exp.fill_between(exposure.index, exposure.values, 1.5,
                    where=cap_high, color="#2ca02c", alpha=0.35, label="Max exposure (1.5×)")

ax_exp.axhline(0.25, color="#d62728", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.50, color="#2ca02c", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.00, color="black", ls=":", lw=0.7, alpha=0.5)

# Crisis shading on bottom panel too
for start, end, label in CRISES:
    ax_exp.axvspan(start, end, color="grey", alpha=0.10, zorder=0)

# Crisis text labels on bottom panel
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
fig.savefig(OUT / "vmp_exposure_mechanism.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "vmp_exposure_mechanism.svg", bbox_inches="tight")
plt.close(fig)
print("  → saved vmp_exposure_mechanism.{png,svg}")

print("\nAll 4 figures generated successfully.")
print(f"Output directory: {OUT.resolve()}")
