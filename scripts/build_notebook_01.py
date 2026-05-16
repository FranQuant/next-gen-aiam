#!/usr/bin/env python3
"""Build notebooks/01_static_baselines.ipynb programmatically."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

NB_PATH = Path("notebooks/01_static_baselines.ipynb")
NB_PATH.parent.mkdir(exist_ok=True)

cells = []

def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""# Static Baselines: Paper Reproduction + Practitioner Analytics

**next-gen-aiam** · `notebooks/01_static_baselines.ipynb`

This notebook has two parts:

1. **Paper reproduction** — every table and figure in the paper, computed from cached parquets, with a reproducibility-assertion cell at the end.
2. **Practitioner analytics** — rolling Sharpe, calendar-year heatmap, drawdown, CVaR, turnover, weight distribution, allocation pie/timeline/snapshots, strategy correlation matrix, bootstrap CIs, and pairwise Sharpe tests.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
code("""\
import os, sys, warnings
from pathlib import Path

# Work from project root regardless of where Jupyter was launched
ROOT = Path(__file__).resolve().parents[1] if '__file__' in dir() else Path.cwd()
while ROOT.name != 'next-gen-aiam' and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
print('Working dir:', Path.cwd())
""")

code("""\
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

from aiam.evaluation.transaction_costs import apply_costs, compute_turnover
from aiam.harness.horse_race import _weights_path

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.20,
    "grid.linestyle":     "--",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.labelsize":     10,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
})

FAMILY_COLORS = {
    "Classical MV":    "#1f77b4",
    "Diversification": "#2ca02c",
    "Regime Switch":   "#9467bd",
    "TSMOM":           "#ff7f0e",
    "Black-Litterman": "#d62728",
    "Factor":          "#8c564b",
    "EW (benchmark)":  "#333333",
}

FAMILY_MAP = {
    "EW":               "EW (benchmark)",
    "GMV(sample)":      "Classical MV",
    "GMV(ledoit_wolf)": "Classical MV",
    "GMV(oas)":         "Classical MV",
    "MSR(sample)":      "Classical MV",
    "MSR(ledoit_wolf)": "Classical MV",
    "MDP(sample)":      "Diversification",
    "MDP(ledoit_wolf)": "Diversification",
    "RP(sample)":       "Diversification",
    "RP(ledoit_wolf)":  "Diversification",
    "HRP(sample)":      "Diversification",
    "HRP(ledoit_wolf)": "Diversification",
    "SWITCH(sample)":   "Regime Switch",
    "SWITCH(ledoit_wolf)": "Regime Switch",
    "TSMOM(12m)":       "TSMOM",
    "TSMOM(6m)":        "TSMOM",
    "BL-Eq(sample)":    "Black-Litterman",
    "BL-Eq(LW)":        "Black-Litterman",
    "BL-Mom(LW)":       "Black-Litterman",
    "BL-Rev(LW)":       "Black-Litterman",
    "FF3-Mom":          "Factor",
    "FF3-LowVol":       "Factor",
    "FF3-Quality":      "Factor",
    "FF3-Multi":        "Factor",
}

DISPLAY = {
    "EW":                "EW",
    "GMV(sample)":       "GMV(sample)",
    "GMV(ledoit_wolf)":  "GMV(LW)",
    "GMV(oas)":          "GMV(OAS)",
    "MSR(sample)":       "MSR(sample)",
    "MSR(ledoit_wolf)":  "MSR(LW)",
    "MDP(sample)":       "MDP(sample)",
    "MDP(ledoit_wolf)":  "MDP(LW)",
    "RP(sample)":        "RP(sample)",
    "RP(ledoit_wolf)":   "RP(LW)",
    "HRP(sample)":       "HRP(sample)",
    "HRP(ledoit_wolf)":  "HRP(LW)",
    "SWITCH(sample)":    "SWITCH(sample)",
    "SWITCH(ledoit_wolf)": "SWITCH(LW)",
    "TSMOM(12m)":        "TSMOM(12m)",
    "TSMOM(6m)":         "TSMOM(6m)",
    "BL-Eq(sample)":     "BL-Eq(sample)",
    "BL-Eq(LW)":         "BL-Eq(LW)",
    "BL-Mom(LW)":        "BL-Mom(LW)",
    "BL-Rev(LW)":        "BL-Rev(LW)",
    "FF3-Mom":           "FF3-Mom",
    "FF3-LowVol":        "FF3-LowVol",
    "FF3-Quality":       "FF3-Quality",
    "FF3-Multi":         "FF3-Multi",
}

CRISES = [
    (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), "GFC"),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"), "COVID"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31"), "Rate Shock"),
]

COST_BPS = 10.0
TRADING_DAYS = 252
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# ── Load returns ──────────────────────────────────────────────────────────────
base = pd.read_parquet("data/cache/portfolio_returns/24strategies_2008_2026.parquet")
vmp  = pd.read_parquet("data/cache/portfolio_returns/24strategies_vmp_2008_2026.parquet")
regime_sig = pd.read_parquet("data/cache/regime_signals.parquet")["dominant_regime"].dropna()
regime_conditional_sharpe = pd.read_parquet("data/cache/regime_conditional_sharpe.parquet")
prices = pd.read_parquet("data/cache/prices_30.parquet")

# ── Forward-fill regime to daily ──────────────────────────────────────────────
regime_daily = regime_sig.resample("B").ffill().reindex(base.index, method="ffill")

# ── Build SWITCH(v2a) (R0→MSR(LW), R5→MSR(sample), others→MDP(LW)) ───────────
r = regime_daily
switch_v2a = pd.Series(index=base.index, dtype=float)
switch_v2a[r == 0.0] = base.loc[r == 0.0, "MSR(ledoit_wolf)"]
switch_v2a[r == 5.0] = base.loc[r == 5.0, "MSR(sample)"]
mask_other = ~r.isin([0.0, 5.0])
switch_v2a[mask_other] = base.loc[mask_other, "MDP(ledoit_wolf)"]
switch_v2a.name = "SWITCH(v2a)"

# ── Build VMP(SWITCH(v2a)) ────────────────────────────────────────────────────
target_vol = switch_v2a.std() * np.sqrt(TRADING_DAYS)
roll_vol   = switch_v2a.rolling(21).std() * np.sqrt(TRADING_DAYS)
exposure   = (target_vol / roll_vol.shift(1)).clip(0.25, 1.5)
vmp_switch_v2a = (switch_v2a * exposure).rename("VMP(SWITCH(v2a))")

# ── SPY daily returns ─────────────────────────────────────────────────────────
spy_ret = prices["SPY.US"].pct_change().dropna()

# ── Load all 24 weight DataFrames ─────────────────────────────────────────────
WEIGHT_COLS = list(base.columns)
weights_cache = {}
for col in WEIGHT_COLS:
    p = _weights_path(col)
    if p.exists():
        weights_cache[col] = pd.read_parquet(p)

print(f"base: {base.shape}, vmp: {vmp.shape}")
print(f"regime_daily: {regime_daily.shape}, prices: {prices.shape}")
print(f"SWITCH(v2a) sharpe: {switch_v2a.mean()/switch_v2a.std()*np.sqrt(252):.3f}")
print(f"VMP(SWITCH(v2a)) sharpe: {vmp_switch_v2a.mean()/vmp_switch_v2a.std()*np.sqrt(252):.3f}")
print(f"Weights loaded: {len(weights_cache)}/{len(WEIGHT_COLS)}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
code("""\
def ann_sharpe(s):
    return s.mean() / s.std() * np.sqrt(TRADING_DAYS)

def ann_return(s):
    return (1 + s).prod() ** (TRADING_DAYS / len(s)) - 1

def ann_vol(s):
    return s.std() * np.sqrt(TRADING_DAYS)

def max_dd(s):
    c = (1 + s).cumprod()
    return ((c - c.cummax()) / c.cummax()).min()

def calmar(s):
    dd = max_dd(s)
    return ann_return(s) / abs(dd) if dd != 0 else np.nan

def cum_wealth(s):
    return (1 + s).cumprod()

def shade_crises(ax, alpha=0.10):
    for start, end, _ in CRISES:
        ax.axvspan(start, end, color="grey", alpha=alpha, zorder=0)

def net_sharpe_for(col, cost_bps=10.0):
    rets = base[col] if col in base.columns else vmp[col]
    # Use base weights for both base and VMP variants
    base_col = col.replace("VMP(", "").rstrip(")") if col.startswith("VMP(") else col
    w = weights_cache.get(base_col)
    if w is None:
        return np.nan
    net = apply_costs(rets, w, cost_bps=cost_bps)
    return ann_sharpe(net)

def turnover_for(col):
    base_col = col.replace("VMP(", "").rstrip(")") if col.startswith("VMP(") else col
    w = weights_cache.get(base_col)
    if w is None:
        return np.nan
    return compute_turnover(w).dropna().mean() * 100

def strategy_returns(col):
    \"\"\"Fetch returns for any column name (base, VMP, or special strategies).\"\"\"
    if col == "SWITCH(v2a)":
        return switch_v2a
    if col == "VMP(SWITCH(v2a))":
        return vmp_switch_v2a
    if col in base.columns:
        return base[col]
    if col in vmp.columns:
        return vmp[col]
    raise KeyError(f"Unknown strategy: {col}")

# Precompute full stats for all 48 base + vmp strategies
ALL_STRATS = []
for col in base.columns:
    s = base[col]
    ALL_STRATS.append({
        "col": col, "display": DISPLAY[col], "is_vmp": False,
        "family": FAMILY_MAP.get(col, "EW (benchmark)"),
        "ann_ret": ann_return(s)*100, "ann_vol": ann_vol(s)*100,
        "sharpe": ann_sharpe(s), "max_dd": max_dd(s)*100,
        "calmar": calmar(s), "turnover": turnover_for(col),
        "net_sharpe": net_sharpe_for(col),
    })
for col in vmp.columns:
    s = vmp[col]
    base_col = col.replace("VMP(", "").rstrip(")")
    ALL_STRATS.append({
        "col": col, "display": col.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)"),
        "is_vmp": True,
        "family": FAMILY_MAP.get(base_col, "EW (benchmark)"),
        "ann_ret": ann_return(s)*100, "ann_vol": ann_vol(s)*100,
        "sharpe": ann_sharpe(s), "max_dd": max_dd(s)*100,
        "calmar": calmar(s), "turnover": turnover_for(col),
        "net_sharpe": net_sharpe_for(col),
    })

df_all = pd.DataFrame(ALL_STRATS)
df_base = df_all[~df_all.is_vmp].copy()
df_vmp  = df_all[df_all.is_vmp].copy()
print(f"Total strategies: {len(df_all)}  (base={len(df_base)}, vmp={len(df_vmp)})")
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 HEADER
# ─────────────────────────────────────────────────────────────────────────────
md("---\n## Part 1 — Paper Reproduction\n\nAll tables and figures from the paper, computed from cache.")

# ─── §Appendix A: Master Table ───
md("""### Appendix A — Full 48-Strategy Comparison Table

*Paper reference: Appendix A.*
Families: Classical MV, Diversification, Regime, TSMOM, Black-Litterman, FF3 Factor.
Each base strategy followed by its VMP variant. Numbers should match the paper exactly.
""")

code("""\
FAMILY_ORDER = [
    ("Classical MV",    ["EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)", "MSR(sample)", "MSR(ledoit_wolf)"]),
    ("Diversification", ["MDP(sample)", "MDP(ledoit_wolf)", "RP(sample)", "RP(ledoit_wolf)", "HRP(sample)", "HRP(ledoit_wolf)"]),
    ("Regime Switch",   ["SWITCH(sample)", "SWITCH(ledoit_wolf)"]),
    ("TSMOM",           ["TSMOM(12m)", "TSMOM(6m)"]),
    ("Black-Litterman", ["BL-Eq(sample)", "BL-Eq(LW)", "BL-Mom(LW)", "BL-Rev(LW)"]),
    ("Factor",          ["FF3-Mom", "FF3-LowVol", "FF3-Quality", "FF3-Multi"]),
]

rows = []
for fam, cols in FAMILY_ORDER:
    for col in cols:
        s = base[col]
        vmp_col = f"VMP({col})"
        sv = vmp[vmp_col]
        for is_v, series, cname in [(False, s, col), (True, sv, vmp_col)]:
            disp = cname.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)")
            rows.append({
                "Family": fam if not is_v else "",
                "Strategy": disp,
                "Ann Ret": f"{ann_return(series)*100:.2f}%",
                "Ann Vol": f"{ann_vol(series)*100:.2f}%",
                "Sharpe":  f"{ann_sharpe(series):.3f}",
                "Max DD":  f"{max_dd(series)*100:.2f}%",
                "Calmar":  f"{calmar(series):.3f}",
                "Turnover": f"{turnover_for(cname):.2f}%",
                "Net Sharpe": f"{net_sharpe_for(cname):.3f}",
            })

master_table = pd.DataFrame(rows)
pd.set_option("display.max_rows", 60)
pd.set_option("display.max_colwidth", 30)
display(master_table.to_string(index=False))
""")

# ─── §3 Rankings ───
md("""### §3.2 — Rankings

*Paper reference: §3.2 Results → Rankings.*
""")

code("""\
print("── Top 10 by Sharpe (all 48 strategies) ──")
top10_sharpe = df_all.nlargest(10, "sharpe")[["display","sharpe","ann_ret","max_dd"]].reset_index(drop=True)
top10_sharpe.index += 1
display(top10_sharpe.rename(columns={"display":"Strategy","ann_ret":"Ann Ret (%)","max_dd":"Max DD (%)"}))

print()
print("── Top 5 by Annualized Return ──")
top5_ret = df_all.nlargest(5, "ann_ret")[["display","ann_ret","sharpe"]].reset_index(drop=True)
top5_ret.index += 1
display(top5_ret.rename(columns={"display":"Strategy","ann_ret":"Ann Ret (%)"}))

print()
print("── Bottom 5 by Sharpe (base strategies only) ──")
bot5 = df_base.nsmallest(5, "sharpe")[["display","sharpe","ann_ret"]].reset_index(drop=True)
bot5.index += 1
display(bot5.rename(columns={"display":"Strategy","ann_ret":"Ann Ret (%)"}))
""")

# ─── §3.3 TC sensitivity ───
md("""### §3.3 — Transaction-Cost Sensitivity

*Paper reference: §3.3 Transaction-Cost Sensitivity.*
Uniform 10 bps round-trip cost applied per unit of one-way turnover.
""")

code("""\
print("── Top 10 by Net Sharpe (10 bps) ──")
top10_net = df_all.nlargest(10, "net_sharpe")[["display","sharpe","net_sharpe","turnover"]].reset_index(drop=True)
top10_net.index += 1
display(top10_net.rename(columns={"display":"Strategy","sharpe":"Gross Sharpe","net_sharpe":"Net Sharpe","turnover":"Turnover (%)"}))

print()
print("── Top 5 by Sharpe Degradation (base strategies only) ──")
df_base2 = df_base.copy()
df_base2["degradation"] = df_base2["sharpe"] - df_base2["net_sharpe"]
top5_deg = df_base2.nlargest(5, "degradation")[["display","sharpe","net_sharpe","turnover","degradation"]].reset_index(drop=True)
top5_deg.index += 1
display(top5_deg.rename(columns={"display":"Strategy","sharpe":"Gross Sharpe","net_sharpe":"Net Sharpe","turnover":"Turnover (%)","degradation":"Degradation"}))
""")

# ─── Figure 1 ───
md("""### Figure 1 — Cumulative Wealth

*Paper reference: §3.1, Figure 1 caption.*
Log-y axis. Crisis shading: GFC, COVID, Rate Shock.
""")

code("""\
F1_STRATS = [
    ("EW",                    "EW (benchmark)",     FAMILY_COLORS["EW (benchmark)"],  dict(lw=2.4, ls="--", zorder=5)),
    ("VMP(MSR(ledoit_wolf))", "VMP(MSR(LW))",        FAMILY_COLORS["Classical MV"],    dict(lw=1.8, zorder=4)),
    ("VMP(BL-Mom(LW))",       "VMP(BL-Mom(LW))",    FAMILY_COLORS["Black-Litterman"], dict(lw=1.8, zorder=4)),
    (None,                    "SWITCH(v2a)",          FAMILY_COLORS["Regime Switch"],          dict(lw=1.6, zorder=3)),
    ("BL-Mom(LW)",            "BL-Mom(LW)",          FAMILY_COLORS["Black-Litterman"], dict(lw=1.2, ls=":", zorder=3)),
    ("VMP(GMV(sample))",      "VMP(GMV(sample))",   FAMILY_COLORS["Classical MV"],    dict(lw=1.2, ls="-.", zorder=2)),
    ("FF3-LowVol",            "FF3-LowVol",          FAMILY_COLORS["Factor"],      dict(lw=1.5, zorder=3)),
]

fig, ax = plt.subplots(figsize=(12, 6))
for col, label, color, kw in F1_STRATS:
    rets = switch_v2a if col is None else (
        base[col] if col in base.columns else vmp[col]
    )
    cw = cum_wealth(rets)
    ax.semilogy(cw.index, cw.values, color=color, label=label, **kw)

shade_crises(ax)
fig.canvas.draw()
ylim = ax.get_ylim()
y_label = np.exp(np.log(ylim[0]) + 0.015 * (np.log(ylim[1]) - np.log(ylim[0])))
for start, end, label in CRISES:
    ax.text(start + (end - start)/2, y_label, label, fontsize=6.5, ha="center", va="bottom",
            color="dimgray", style="italic")

# BL-Mom(LW) trough annotation
bl = base["BL-Mom(LW)"]
cum_bl = cum_wealth(bl)
dd_bl  = (cum_bl - cum_bl.cummax()) / cum_bl.cummax()
trough_date = dd_bl.idxmin()
ax.annotate("BL-Mom(LW)\\n−50.85% max DD",
    xy=(trough_date, cum_bl[trough_date]),
    xytext=(trough_date + pd.Timedelta(days=300), cum_bl[trough_date] * 0.62),
    fontsize=7.5, color=FAMILY_COLORS["Black-Litterman"],
    arrowprops=dict(arrowstyle="-", color=FAMILY_COLORS["Black-Litterman"], lw=0.8), ha="left")

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative wealth (log scale, 1 = initial $1)")
ax.set_title("Cumulative Wealth, January 2008 – May 2026", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2g}×"))
ax.legend(ncol=2, loc="upper left", framealpha=0.85)
fig.tight_layout()
plt.show()
""")

# ─── Figure 2 ───
md("""### Figure 2 — Sharpe vs Maximum Drawdown Scatter

*Paper reference: §3.2 Rankings, Figure 2 caption.*
Filled circles = base strategies; open rings = VMP variants. Color = family.
""")

code("""\
fig, ax = plt.subplots(figsize=(11, 7))

for family, grp in df_all.groupby("family"):
    color = FAMILY_COLORS[family]
    base_pts = grp[~grp.is_vmp]
    vmp_pts  = grp[grp.is_vmp]
    if len(base_pts):
        ax.scatter(base_pts.max_dd, base_pts.sharpe, color=color, marker="o",
                   s=50, zorder=4, edgecolors=color, label=f"{family} (base)")
    if len(vmp_pts):
        ax.scatter(vmp_pts.max_dd, vmp_pts.sharpe, color="none", marker="o",
                   s=70, zorder=4, edgecolors=color, linewidths=1.6, label=f"{family} (VMP)")

ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6)
ax.axvline(-20.0, color="black", ls="--", lw=0.8, alpha=0.6)
ax.text(-19.5, df_all.sharpe.min() + 0.02, "Max DD = −20%", fontsize=7, color="dimgray", va="bottom")
ax.text(df_all.max_dd.max() * 0.5, 1.01, "Sharpe = 1.0", fontsize=7, color="dimgray", va="bottom", ha="center")

# Annotate top-5 by Sharpe
top5_pts = df_all.nlargest(5, "sharpe")
offsets = {"VMP(GMV(sample))": (3, 0.02), "VMP(MDP(sample))": (2, 0.02),
           "VMP(SWITCH(sample))": (-8, 0.04), "VMP(SWITCH(LW))": (-6, -0.06),
           "VMP(MDP(LW))": (2, -0.05)}
for _, row in top5_pts.iterrows():
    dx, dy = offsets.get(row.display, (2, 0.02))
    ax.annotate(row.display, xy=(row.max_dd, row.sharpe),
                xytext=(row.max_dd + dx, row.sharpe + dy), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="gray"),
                ha="left" if dx >= 0 else "right")

family_patches = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_COLORS]
base_mkr = plt.Line2D([0],[0], marker="o", color="gray", ls="", markersize=6,
                       markerfacecolor="gray", label="Base strategy")
vmp_mkr  = plt.Line2D([0],[0], marker="o", color="gray", ls="", markersize=8,
                       markerfacecolor="none", markeredgewidth=1.6, label="VMP variant")
ax.legend(handles=family_patches + [base_mkr, vmp_mkr], ncol=2, fontsize=7.5,
          loc="lower right", framealpha=0.85)
ax.set_xlabel("Maximum Drawdown (%)")
ax.set_ylabel("Sharpe Ratio (annualized)")
ax.set_title("Sharpe vs. Maximum Drawdown — All 48 Strategies", fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── Figure 3 ───
md("""### Figure 3 — Regime-Conditional Sharpe Heatmap

*Paper reference: Finding 5 (§3 Findings), Figure 3 caption.*
12 base strategies × 8 regimes. Diverging red–blue. Gold borders = SWITCH(v2a) selection cells.
""")

code("""\
REGIME_LABELS = {
    0: "R0\\nExpansion", 1: "R1\\nRecovery", 2: "R2\\nNeutral",
    3: "R3\\nSlow\\nGrowth", 4: "R4\\nStress",
    5: "R5\\nLow &\\nContracting", 6: "R6\\nCrisis", 7: "R7\\nContraction",
}
SPARSE_THRESHOLD = 252
HEATMAP_STRATS = list(regime_conditional_sharpe.index)

ndays = {}
for k in range(8):
    mask = (regime_daily == float(k)).reindex(base.index, fill_value=False)
    ndays[k] = mask.sum()

reg_order = [0, 1, 2, 3, 4, 5, 6, 7]
Z = regime_conditional_sharpe[reg_order].values   # shape: 12×8

vmax = min(max(abs(np.nanmin(Z)), abs(np.nanmax(Z))), 3.0)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(Z, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, label="Annualized Sharpe", shrink=0.8)

for i, strat in enumerate(HEATMAP_STRATS):
    for j, k in enumerate(reg_order):
        val = Z[i, j]
        n   = ndays[k]
        txt = "—" if np.isnan(val) else (f"{val:.2f}*" if n < SPARSE_THRESHOLD else f"{val:.2f}")
        color = "white" if (not np.isnan(val) and abs(val) > vmax * 0.6) else ("dimgray" if np.isnan(val) else "black")
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
        if n < SPARSE_THRESHOLD:
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5), 1, 1, fill=False, hatch="////",
                                        edgecolor="white", linewidth=0, alpha=0.35, zorder=3))

# Gold borders for SWITCH(v2a) cells
highlight_cells = [(HEATMAP_STRATS.index("MSR(LW)"), reg_order.index(0)),
                   (HEATMAP_STRATS.index("MSR(sample)"), reg_order.index(5))]
for (ri, ci) in highlight_cells:
    ax.add_patch(FancyBboxPatch((ci-0.48, ri-0.48), 0.96, 0.96, boxstyle="square,pad=0",
                                 fill=False, edgecolor="gold", linewidth=2.5, zorder=5))

ax.set_xticks(range(8)); ax.set_xticklabels([REGIME_LABELS[k] for k in reg_order], fontsize=7.5)
ax.set_yticks(range(len(HEATMAP_STRATS))); ax.set_yticklabels(HEATMAP_STRATS, fontsize=8)
ax.set_title(
    "Regime-Conditional Annualized Sharpe — 12 Base Strategies × 8 Regimes\\n"
    "(*sparse: n < 252 days; gold border = SWITCH(v2a) selection rule)",
    fontsize=9.5, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── Figure 4 ───
md("""### Figure 4 — VMP Exposure Mechanism for MSR(LW)

*Paper reference: §2.4 VMP Overlay, Figure 4 caption.*
21-day realized vol (top) and exposure multiplier (bottom) for MSR(LW).
""")

code("""\
msr_lw = base["MSR(ledoit_wolf)"]
target_vol_msr = msr_lw.std() * np.sqrt(TRADING_DAYS)
roll_vol_msr   = msr_lw.rolling(21).std() * np.sqrt(TRADING_DAYS)
roll_vol_lag   = roll_vol_msr.shift(1)
exposure_msr   = (target_vol_msr / roll_vol_lag).clip(0.25, 1.5)
cap_low  = exposure_msr <= 0.251
cap_high = exposure_msr >= 1.499

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"hspace": 0.08})
ax_vol, ax_exp = axes

ax_vol.plot(roll_vol_msr.index, roll_vol_msr.values * 100,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="21-day realized vol")
ax_vol.axhline(target_vol_msr * 100, color="black", ls="--", lw=1.0,
               label=f"Long-run vol ({target_vol_msr*100:.1f}%)")
shade_crises(ax_vol)
ax_vol.set_ylabel("Realized Vol (%, ann.)")
ax_vol.set_title("MSR(LW): 21-day Realized Volatility and VMP Exposure Multiplier",
                  fontsize=11, fontweight="bold")
ax_vol.legend(fontsize=8, loc="upper right")

ax_exp.plot(exposure_msr.index, exposure_msr.values,
            color=FAMILY_COLORS["Classical MV"], lw=0.9, alpha=0.85, label="VMP exposure")
ax_exp.fill_between(exposure_msr.index, 0.25, exposure_msr.values,
                     where=cap_low, color="#d62728", alpha=0.35, label="Vol cap active (0.25×)")
ax_exp.fill_between(exposure_msr.index, exposure_msr.values, 1.5,
                     where=cap_high, color="#2ca02c", alpha=0.35, label="Max exposure (1.5×)")
ax_exp.axhline(0.25, color="#d62728", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.50, color="#2ca02c", ls="--", lw=0.9, alpha=0.7)
ax_exp.axhline(1.00, color="black",   ls=":", lw=0.7, alpha=0.5)
shade_crises(ax_exp)
fig.canvas.draw()
ylim_e = ax_exp.get_ylim()
for start, end, label in CRISES:
    ax_exp.text(start+(end-start)/2, ylim_e[0]+0.01*(ylim_e[1]-ylim_e[0]),
                label, fontsize=6.5, ha="center", va="bottom", color="dimgray", style="italic")
ax_exp.set_ylabel("Exposure multiplier"); ax_exp.set_xlabel("Date")
ax_exp.legend(fontsize=8, loc="upper right", ncol=2)
ax_exp.set_ylim(0.15, 1.65)
fig.tight_layout()
plt.show()
""")

# ─── Reproducibility ───
md("""### Reproducibility Check

Assert every number in the paper's main table matches the values computed from cache.
Tolerance: ±0.001 for Sharpe, ±0.001 for Ann Ret (%pt), ±0.001 for Max DD (%pt).
""")

code("""\
PAPER_TABLE = {
    # (col, is_vmp): (ann_ret, ann_vol, sharpe, max_dd, calmar, turnover, net_sharpe)
    ("EW", False):               (14.31, 14.84, 0.976, -37.86, 0.378, 0.00, 0.976),
    ("EW", True):                (18.13, 14.09, 1.253, -28.95, 0.626, 0.00, 1.253),
    ("GMV(sample)", False):      ( 1.80,  1.43, 1.260,  -5.94, 0.304, 0.15, 1.233),
    ("GMV(sample)", True):       ( 2.00,  1.30, 1.533,  -5.40, 0.371, 0.15, 1.503),
    ("GMV(ledoit_wolf)", False):  ( 2.88,  3.23, 0.896, -11.60, 0.248, 0.54, 0.853),
    ("GMV(ledoit_wolf)", True):   ( 3.86,  3.26, 1.178, -11.11, 0.348, 0.54, 1.136),
    ("GMV(oas)", False):          ( 2.27,  2.58, 0.883, -10.64, 0.213, 0.47, 0.837),
    ("GMV(oas)", True):           ( 3.13,  2.60, 1.200,  -9.11, 0.344, 0.47, 1.154),
    ("MSR(sample)", False):       ( 6.81,  7.80, 0.884, -21.47, 0.317, 5.19, 0.717),
    ("MSR(sample)", True):        ( 8.44,  5.89, 1.405, -11.45, 0.737, 5.19, 1.183),
    ("MSR(ledoit_wolf)", False):  (15.40, 11.91, 1.262, -21.43, 0.719, 4.65, 1.163),
    ("MSR(ledoit_wolf)", True):   (17.53, 11.80, 1.429, -22.66, 0.774, 4.65, 1.329),
    ("MDP(sample)", False):       ( 5.05,  4.63, 1.088, -18.40, 0.275, 2.60, 0.945),
    ("MDP(sample)", True):        ( 6.41,  4.32, 1.460, -12.03, 0.533, 2.60, 1.307),
    ("MDP(ledoit_wolf)", False):  ( 6.34,  5.32, 1.182, -15.73, 0.403, 0.79, 1.144),
    ("MDP(ledoit_wolf)", True):   ( 7.94,  5.42, 1.437, -13.16, 0.604, 0.79, 1.400),
    ("RP(sample)", False):        ( 5.36,  5.59, 0.961, -15.96, 0.336, 2.96, 0.829),
    ("RP(sample)", True):         ( 7.22,  5.35, 1.330, -12.20, 0.592, 2.96, 1.191),
    ("RP(ledoit_wolf)", False):   ( 7.25,  6.74, 1.073, -16.61, 0.437, 0.95, 1.037),
    ("RP(ledoit_wolf)", True):    ( 8.82,  6.64, 1.306, -13.68, 0.645, 0.95, 1.269),
    ("HRP(sample)", False):       ( 5.99,  6.70, 0.902, -16.57, 0.362, 3.92, 0.753),
    ("HRP(sample)", True):        ( 7.04,  6.57, 1.068, -15.51, 0.454, 3.92, 0.915),
    ("HRP(ledoit_wolf)", False):  ( 6.48,  7.60, 0.865, -15.65, 0.414, 3.63, 0.743),
    ("HRP(ledoit_wolf)", True):   ( 7.63,  7.42, 1.027, -15.06, 0.506, 3.63, 0.903),
    ("SWITCH(sample)", False):    ( 8.70,  8.09, 1.071, -20.79, 0.418, 3.37, 0.967),
    ("SWITCH(sample)", True):     (10.48,  7.01, 1.457, -13.91, 0.753, 3.37, 1.337),
    ("SWITCH(ledoit_wolf)", False): (11.02, 9.23, 1.179, -21.13, 0.521, 1.98, 1.125),
    ("SWITCH(ledoit_wolf)", True):  (12.91, 8.71, 1.438, -18.06, 0.715, 1.98, 1.381),
    ("TSMOM(12m)", False):         ( 4.05,  6.70, 0.626, -21.68, 0.187, 2.93, 0.514),
    ("TSMOM(12m)", True):          ( 6.13,  6.30, 0.976, -13.47, 0.455, 2.93, 0.857),
    ("TSMOM(6m)", False):          ( 6.48,  7.23, 0.904, -24.18, 0.268, 4.77, 0.738),
    ("TSMOM(6m)", True):           ( 7.27,  6.56, 1.102, -12.33, 0.589, 4.77, 0.918),
    ("BL-Eq(sample)", False):      (12.76, 14.77, 0.887, -37.86, 0.337, 0.00, 0.887),
    ("BL-Eq(sample)", True):       (16.24, 14.00, 1.145, -28.85, 0.563, 0.00, 1.145),
    ("BL-Eq(LW)", False):          (12.76, 14.77, 0.887, -37.86, 0.337, 0.00, 0.887),
    ("BL-Eq(LW)", True):           (16.24, 14.00, 1.145, -28.85, 0.563, 0.00, 1.145),
    ("BL-Mom(LW)", False):         (20.01, 19.12, 1.049, -50.85, 0.394, 4.91, 0.985),
    ("BL-Mom(LW)", True):          (24.97, 17.73, 1.346, -36.01, 0.693, 4.91, 1.276),
    ("BL-Rev(LW)", False):         (10.17, 22.27, 0.547, -48.33, 0.210, 10.05, 0.433),
    ("BL-Rev(LW)", True):          (12.18, 19.13, 0.697, -47.61, 0.256, 10.05, 0.565),
    ("FF3-Mom", False):            ( 9.60, 18.53, 0.588, -39.51, 0.243, 20.51, 0.310),
    ("FF3-Mom", True):             (11.61, 16.97, 0.733, -29.85, 0.389, 20.51, 0.430),
    ("FF3-LowVol", False):         ( 3.17,  3.39, 0.936, -10.68, 0.296, 0.41, 0.905),
    ("FF3-LowVol", True):          ( 3.77,  3.27, 1.146,  -9.53, 0.395, 0.41, 1.115),
    ("FF3-Quality", False):        ( 6.59,  9.41, 0.726, -25.98, 0.254, 3.62, 0.628),
    ("FF3-Quality", True):         ( 8.18,  8.06, 1.016, -16.72, 0.489, 3.62, 0.902),
    ("FF3-Multi", False):          ( 6.79,  8.86, 0.786, -19.54, 0.348, 7.95, 0.561),
    ("FF3-Multi", True):           ( 8.35,  8.42, 0.995, -15.98, 0.522, 7.95, 0.757),
}

TOL_SHARPE = 0.002
TOL_PCT    = 0.02   # percentage points (e.g. 0.02 = 2 bps on ann_ret in %)

failures = []
for (col, is_vmp), (exp_ret, exp_vol, exp_sharpe, exp_maxdd, exp_calmar, exp_to, exp_net) in PAPER_TABLE.items():
    actual_col = f"VMP({col})" if is_vmp else col
    s  = vmp[f"VMP({col})"] if is_vmp else base[col]
    act_sharpe  = ann_sharpe(s)
    act_ret     = ann_return(s) * 100
    act_vol     = ann_vol(s) * 100
    act_maxdd   = max_dd(s) * 100
    act_calmar  = calmar(s)
    act_to      = turnover_for(actual_col)
    act_net     = net_sharpe_for(actual_col)

    errs = []
    if abs(act_sharpe - exp_sharpe) > TOL_SHARPE:
        errs.append(f"Sharpe: paper={exp_sharpe:.3f} computed={act_sharpe:.3f}")
    if abs(act_ret - exp_ret) > TOL_PCT:
        errs.append(f"AnnRet: paper={exp_ret:.2f}% computed={act_ret:.2f}%")
    if abs(act_maxdd - exp_maxdd) > TOL_PCT:
        errs.append(f"MaxDD: paper={exp_maxdd:.2f}% computed={act_maxdd:.2f}%")
    if abs(act_net - exp_net) > TOL_SHARPE:
        errs.append(f"NetSharpe: paper={exp_net:.3f} computed={act_net:.3f}")
    if errs:
        failures.append(f"{actual_col}: {'; '.join(errs)}")

if failures:
    print("REPRODUCIBILITY FAILURES:")
    for f in failures:
        print(f"  ✗ {f}")
else:
    print(f"✓ All {len(PAPER_TABLE)} paper entries reproduced within tolerance "
          f"(Sharpe ±{TOL_SHARPE}, AnnRet/MaxDD ±{TOL_PCT}%pt)")
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 HEADER
# ─────────────────────────────────────────────────────────────────────────────
md("---\n## Part 2 — Practitioner-Standard Analytics\n\nExtensions not in the paper.")

# ─── 2.1 Rolling Sharpe ───
md("""### 2.1 — Rolling 12-Month Sharpe Time Series

Six representative strategies: EW, GMV(LW), MSR(LW), MDP(LW), SWITCH(v2a), VMP(BL-Mom(LW)).
Rolling window: 252 trading days.
""")

code("""\
REP6_COLS = [
    ("EW",                   "EW",             FAMILY_COLORS["EW (benchmark)"]),
    ("GMV(ledoit_wolf)",      "GMV(LW)",        FAMILY_COLORS["Classical MV"]),
    ("MSR(ledoit_wolf)",      "MSR(LW)",        FAMILY_COLORS["Classical MV"]),
    ("MDP(ledoit_wolf)",      "MDP(LW)",        FAMILY_COLORS["Diversification"]),
    (None,                    "SWITCH(v2a)",    FAMILY_COLORS["Regime Switch"]),
    ("VMP(BL-Mom(LW))",       "VMP(BL-Mom(LW))",FAMILY_COLORS["Black-Litterman"]),
]

WINDOW = TRADING_DAYS  # 252 days

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
axes = axes.flatten()

for idx, (col, label, color) in enumerate(REP6_COLS):
    ax = axes[idx]
    s = switch_v2a if col is None else strategy_returns(col)
    roll_sharpe = s.rolling(WINDOW).apply(lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS), raw=True)
    ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, lw=1.0, alpha=0.9)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axhline(1, color="black", lw=0.6, ls="--", alpha=0.5)
    shade_crises(ax, alpha=0.08)
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)
    ax.set_ylabel("Rolling 12m Sharpe")
    ax.set_xlabel("Date")
    full_mean = ann_sharpe(s)
    ax.text(0.02, 0.97, f"Full-period: {full_mean:.2f}", transform=ax.transAxes,
            fontsize=7.5, va="top", color=color)

fig.suptitle("Rolling 12-Month Sharpe — 6 Representative Strategies", fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── 2.2 Calendar-year returns heatmap ───
md("""### 2.2 — Calendar-Year Returns Heatmap

24 base strategies × 19 calendar years (2008–2026, with 2026 partial through end of data).
Diverging colormap centered at 0%.
""")

code("""\
years = sorted(base.index.year.unique())
cal_data = {}
for col in base.columns:
    s = base[col]
    yearly = {}
    for yr in years:
        mask = s.index.year == yr
        sub  = s[mask].dropna()
        if len(sub) > 0:
            yearly[yr] = ((1 + sub).prod() - 1) * 100
        else:
            yearly[yr] = np.nan
    cal_data[DISPLAY[col]] = yearly

df_cal = pd.DataFrame(cal_data).T   # strategies × years
df_cal.columns = [str(y) for y in df_cal.columns]

# Sort strategies by median annual return
strategy_order = df_cal.median(axis=1).sort_values(ascending=False).index

vmax_cal = max(abs(np.nanpercentile(df_cal.values, 2.5)),
               abs(np.nanpercentile(df_cal.values, 97.5)))
vmax_cal = min(vmax_cal, 60.0)

fig, ax = plt.subplots(figsize=(18, 9))
im = ax.imshow(df_cal.loc[strategy_order].values, cmap="RdBu", vmin=-vmax_cal, vmax=vmax_cal,
               aspect="auto")
plt.colorbar(im, ax=ax, label="Annual Return (%)", shrink=0.6)

for i, strat in enumerate(strategy_order):
    for j, yr in enumerate(df_cal.columns):
        val = df_cal.loc[strat, yr]
        if not np.isnan(val):
            color = "white" if abs(val) > vmax_cal * 0.5 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5, color=color)

ax.set_xticks(range(len(df_cal.columns)))
ax.set_xticklabels(df_cal.columns, fontsize=7.5, rotation=45, ha="right")
ax.set_yticks(range(len(strategy_order)))
ax.set_yticklabels(strategy_order, fontsize=7.5)
ax.set_title("Calendar-Year Returns (%) — 24 Base Strategies × 19 Years",
              fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── 2.3 Underwater drawdown ───
md("""### 2.3 — Underwater Drawdown Plot

Top 6 strategies by Sharpe (from all 48): cumulative wealth (top) and underwater drawdown (bottom).
Paired 2-panel layout, one column per strategy.
""")

code("""\
top6_cols = df_all.nlargest(6, "sharpe")["col"].tolist()
top6_labels = [c.replace("ledoit_wolf","LW").replace("(oas)","(OAS)") for c in top6_cols]
top6_rets = [vmp[c] if c in vmp.columns else base[c] for c in top6_cols]
top6_colors = [FAMILY_COLORS[FAMILY_MAP.get(c.replace("VMP(","").rstrip(")"), "EW (benchmark)")]
               for c in top6_cols]

fig, axes = plt.subplots(2, 6, figsize=(18, 7), sharex=True)

for idx in range(6):
    s = top6_rets[idx]
    cw = cum_wealth(s)
    dd = (cw - cw.cummax()) / cw.cummax() * 100
    color = top6_colors[idx]
    label = top6_labels[idx]
    sharpe_val = ann_sharpe(s)

    ax_top = axes[0, idx]
    ax_bot = axes[1, idx]

    ax_top.semilogy(cw.index, cw.values, color=color, lw=1.1)
    shade_crises(ax_top, alpha=0.07)
    ax_top.set_title(f"{label}\\nSharpe={sharpe_val:.3f}", fontsize=7.5, fontweight="bold", color=color)
    if idx == 0:
        ax_top.set_ylabel("Cumulative Wealth (log)")

    ax_bot.fill_between(dd.index, dd.values, 0, where=(dd.values < 0),
                         color=color, alpha=0.4)
    ax_bot.plot(dd.index, dd.values, color=color, lw=0.7)
    ax_bot.axhline(0, color="black", lw=0.5)
    shade_crises(ax_bot, alpha=0.07)
    if idx == 0:
        ax_bot.set_ylabel("Drawdown (%)")
    ax_bot.set_xlabel("")

fig.suptitle("Top-6 Strategies by Sharpe — Cumulative Wealth & Underwater Drawdown",
              fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
""")

# ─── 2.4 CVaR bar chart ───
md("""### 2.4 — CVaR(95%) Bar Chart

All 24 base strategies, sorted descending by CVaR (tail risk). Strategies with high CVaR relative
to their Sharpe ratio are flagged.
""")

code("""\
def cvar_95(s):
    \"\"\"Historical CVaR at 95% confidence (expected loss in worst 5% of days).\"\"\"
    cutoff = np.percentile(s, 5)
    tail   = s[s <= cutoff]
    return tail.mean() * 100  # in %

cvar_rows = []
for col in base.columns:
    s = base[col]
    cv = cvar_95(s)
    sh = ann_sharpe(s)
    cvar_rows.append({"col": col, "display": DISPLAY[col],
                       "cvar_95": cv, "sharpe": sh,
                       "family": FAMILY_MAP.get(col, "EW (benchmark)")})
df_cvar = pd.DataFrame(cvar_rows).sort_values("cvar_95")  # worst CVaR first (most negative)

# Flag if |CVaR| / Sharpe > threshold (high loss per unit of Sharpe)
df_cvar["flag"] = df_cvar["cvar_95"].abs() / df_cvar["sharpe"] > 2.5

fig, ax = plt.subplots(figsize=(13, 7))
colors_bar = [FAMILY_COLORS[f] for f in df_cvar["family"]]
bars = ax.barh(df_cvar["display"], df_cvar["cvar_95"], color=colors_bar, edgecolor="white", lw=0.3)

for bar, row in zip(bars, df_cvar.itertuples()):
    if row.flag:
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                "★", va="center", ha="right", fontsize=9, color="gold")

ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("CVaR(95%) — Daily Return (%)")
ax.set_title("CVaR(95%) by Base Strategy — Sorted by Tail Risk\\n★ = High loss-per-Sharpe (|CVaR|/Sharpe > 2.5)",
              fontsize=10, fontweight="bold")

legend_patches = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_COLORS]
ax.legend(handles=legend_patches, ncol=2, fontsize=7, loc="lower right")
fig.tight_layout()
plt.show()
""")

# ─── 2.5 Rolling SPY correlation ───
md("""### 2.5 — Rolling 252-Day Correlation to SPY

6 representative strategies vs SPY.US returns (from prices_30.parquet).
Higher correlation = more equity beta exposure.
""")

code("""\
spy_aligned = spy_ret.reindex(base.index).ffill()

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True, sharex=True)
axes = axes.flatten()

for idx, (col, label, color) in enumerate(REP6_COLS):
    ax = axes[idx]
    s = switch_v2a if col is None else strategy_returns(col)
    common_idx = s.index.intersection(spy_aligned.dropna().index)
    s_c   = s.reindex(common_idx)
    spy_c = spy_aligned.reindex(common_idx)
    # Vectorized rolling correlation via pandas
    combined = pd.DataFrame({"strat": s_c, "spy": spy_c})
    roll_corr = combined["strat"].rolling(WINDOW).corr(combined["spy"])
    ax.plot(roll_corr.index, roll_corr.values, color=color, lw=1.0, alpha=0.9)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.4)
    shade_crises(ax, alpha=0.08)
    ax.set_title(label, fontsize=9, fontweight="bold", color=color)
    ax.set_ylabel("Rolling Corr to SPY")
    ax.set_ylim(-0.6, 1.05)
    full_corr = s_c.corr(spy_c)
    ax.text(0.02, 0.03, f"Full-period: {full_corr:.2f}", transform=ax.transAxes,
            fontsize=7.5, color=color)

fig.suptitle("Rolling 252-Day Correlation to SPY — 6 Representative Strategies",
              fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── 2.6 Turnover analysis ───
md("""### 2.6 — Turnover Histogram + Boxplot

All 24 base strategies. Histogram of daily one-way turnover distribution; boxplot sorted by median.
""")

code("""\
turnover_series = {}
for col in base.columns:
    w = weights_cache.get(col)
    if w is not None:
        ts = compute_turnover(w).dropna() * 100  # in %
        turnover_series[DISPLAY[col]] = ts

# Sort by median
median_to = {k: v.median() for k, v in turnover_series.items()}
sorted_by_median = sorted(median_to, key=median_to.get)

fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(16, 8))

# Histogram — all strategies overlaid
for col, ts in turnover_series.items():
    fam = FAMILY_MAP.get([k for k, v in DISPLAY.items() if v == col][0], "EW (benchmark)") if col != "SWITCH(v2a)" else "Regime Switch"
    ax_hist.hist(ts.values, bins=50, alpha=0.25, color=FAMILY_COLORS.get(fam, "gray"), density=True)
ax_hist.set_xlabel("Daily One-Way Turnover (%)")
ax_hist.set_ylabel("Density")
ax_hist.set_title("Turnover Distribution — 24 Base Strategies (overlaid)", fontsize=9, fontweight="bold")
ax_hist.set_xlim(left=0)

# Boxplot — sorted by median
box_data  = [turnover_series[k].values for k in sorted_by_median]
box_cols  = []
for k in sorted_by_median:
    orig_col = [c for c, d in DISPLAY.items() if d == k]
    box_cols.append(FAMILY_COLORS.get(FAMILY_MAP.get(orig_col[0], "EW (benchmark)"), "gray") if orig_col else "gray")

bplot = ax_box.boxplot(box_data, vert=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3),
                        whis=[5, 95])
for patch, c in zip(bplot["boxes"], box_cols):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

ax_box.set_yticks(range(1, len(sorted_by_median)+1))
ax_box.set_yticklabels(sorted_by_median, fontsize=7.5)
ax_box.set_xlabel("Daily One-Way Turnover (%)")
ax_box.set_title("Turnover Boxplot — Sorted by Median (5th–95th pctile whiskers)",
                  fontsize=9, fontweight="bold")
ax_box.set_xlim(left=0)

fig.tight_layout()
plt.show()
""")

# ─── 2.7 Max single-position weight ───
md("""### 2.7 — Maximum Single-Position Weight Distribution

Boxplot of daily max-weight across the 30-asset portfolio, all 24 base strategies.
Sorted by median max weight.
""")

code("""\
max_weight_series = {}
for col in base.columns:
    w = weights_cache.get(col)
    if w is not None:
        max_weight_series[DISPLAY[col]] = w.max(axis=1).dropna() * 100  # in %

median_mw = {k: v.median() for k, v in max_weight_series.items()}
sorted_mw = sorted(median_mw, key=median_mw.get)

box_data_mw = [max_weight_series[k].values for k in sorted_mw]
box_cols_mw = []
for k in sorted_mw:
    orig = [c for c, d in DISPLAY.items() if d == k]
    box_cols_mw.append(FAMILY_COLORS.get(FAMILY_MAP.get(orig[0], "EW (benchmark)"), "gray") if orig else "gray")

fig, ax = plt.subplots(figsize=(12, 9))
bplot = ax.boxplot(box_data_mw, vert=False, patch_artist=True,
                   medianprops=dict(color="black", lw=1.5),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3),
                   whis=[5, 95])
for patch, c in zip(bplot["boxes"], box_cols_mw):
    patch.set_facecolor(c); patch.set_alpha(0.7)

ax.set_yticks(range(1, len(sorted_mw)+1))
ax.set_yticklabels(sorted_mw, fontsize=7.5)
ax.axvline(1/30*100, color="black", ls="--", lw=0.8, alpha=0.5)
ax.text(1/30*100 + 0.2, len(sorted_mw)*0.9, "EW (3.33%)", fontsize=7, color="dimgray")
ax.set_xlabel("Max Single-Position Weight (%)")
ax.set_title("Maximum Single-Position Weight Distribution — 24 Base Strategies\\n(5th–95th pctile whiskers; dashed = EW benchmark 3.33%)",
              fontsize=9, fontweight="bold")

legend_patches = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_COLORS]
ax.legend(handles=legend_patches, ncol=2, fontsize=7.5, loc="lower right")
fig.tight_layout()
plt.show()
""")

# ─── 2.8 Average weight pie charts ───
md("""### 2.8 — Average Weight Pie Charts

Time-averaged allocation for 6 representative strategies: EW, GMV(sample), GMV(LW), MSR(LW), MDP(LW), VMP(SWITCH(LW)).
Assets colored by asset class. 2×3 grid.
""")

code("""\
ASSET_CLASS = {
    "AAPL.US": "US Equity", "MSFT.US": "US Equity", "GOOGL.US": "US Equity",
    "NVDA.US": "US Equity", "JPM.US": "US Equity", "JNJ.US": "US Equity",
    "XOM.US": "US Equity", "WMT.US": "US Equity",
    "XLK.US": "Sector ETF", "XLF.US": "Sector ETF", "XLE.US": "Sector ETF",
    "XLV.US": "Sector ETF", "XLP.US": "Sector ETF", "XLU.US": "Sector ETF",
    "SPY.US": "Broad Equity", "IWM.US": "Broad Equity",
    "EFA.US": "Intl Equity", "EEM.US": "Intl Equity", "FXI.US": "Intl Equity",
    "SHY.US": "Fixed Income", "IEF.US": "Fixed Income", "TLT.US": "Fixed Income",
    "AGG.US": "Fixed Income", "HYG.US": "Fixed Income",
    "GLD.US": "Alternatives", "SLV.US": "Alternatives", "DBC.US": "Alternatives",
    "USO.US": "Alternatives", "EURUSD.FOREX": "Alternatives", "BTC-USD.CC": "Alternatives",
}
AC_COLORS = {
    "US Equity":    "#1f77b4",
    "Sector ETF":   "#aec7e8",
    "Broad Equity": "#2ca02c",
    "Intl Equity":  "#98df8a",
    "Fixed Income": "#ff7f0e",
    "Alternatives": "#d62728",
}

PIE6 = [
    ("EW",               "EW"),
    ("GMV(sample)",      "GMV(sample)"),
    ("GMV(ledoit_wolf)", "GMV(LW)"),
    ("MSR(ledoit_wolf)", "MSR(LW)"),
    ("MDP(ledoit_wolf)", "MDP(LW)"),
    ("SWITCH(ledoit_wolf)", "VMP(SWITCH(LW))"),  # use SWITCH(LW) weights
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (wt_col, label) in enumerate(PIE6):
    ax = axes[idx]
    w = weights_cache.get(wt_col)
    if w is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        continue
    avg_w = w.mean(axis=0).dropna()
    avg_w = avg_w[avg_w > 0.001]  # filter tiny weights

    # Group by asset class
    ac_weights = {}
    asset_labels = []
    asset_colors_pie = []
    for ticker, wgt in avg_w.items():
        ac = ASSET_CLASS.get(ticker, "Alternatives")
        ac_weights[ac] = ac_weights.get(ac, 0) + wgt
    acs = sorted(ac_weights, key=ac_weights.get, reverse=True)
    wgts = [ac_weights[ac] for ac in acs]
    colors_pie = [AC_COLORS.get(ac, "gray") for ac in acs]
    wedge_labels = [f"{ac}\\n{ac_weights[ac]*100:.1f}%" for ac in acs]
    ax.pie(wgts, labels=wedge_labels, colors=colors_pie, startangle=90,
           textprops={"fontsize": 6.5}, pctdistance=0.8,
           wedgeprops=dict(edgecolor="white", lw=0.5))
    ax.set_title(label, fontsize=9, fontweight="bold",
                 color=FAMILY_COLORS.get(FAMILY_MAP.get(wt_col, "EW (benchmark)"), "black"))

fig.suptitle("Average Asset-Class Allocation — 6 Representative Strategies",
              fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── 2.9 Asset-class allocation timeline ───
md("""### 2.9 — Asset-Class Allocation Timeline

6 asset classes stacked area chart for the same 6 representative strategies.
""")

code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
axes = axes.flatten()
AC_ORDER = ["US Equity","Sector ETF","Broad Equity","Intl Equity","Fixed Income","Alternatives"]
AC_COLORS_TIMELINE = {
    "US Equity":    "#1f77b4",
    "Sector ETF":   "#aec7e8",
    "Broad Equity": "#2ca02c",
    "Intl Equity":  "#98df8a",
    "Fixed Income": "#ff7f0e",
    "Alternatives": "#d62728",
}

for idx, (wt_col, label) in enumerate(PIE6):
    ax = axes[idx]
    w = weights_cache.get(wt_col)
    if w is None:
        continue
    # Resample to monthly for smoother timeline
    w_monthly = w.resample("ME").last().fillna(0)
    ac_ts = pd.DataFrame({ac: w_monthly[[c for c in w_monthly.columns if ASSET_CLASS.get(c) == ac]].sum(axis=1)
                           for ac in AC_ORDER})
    # Fill any NaN with 0 and renormalize
    ac_ts = ac_ts.fillna(0)
    total = ac_ts.sum(axis=1).replace(0, np.nan)
    ac_ts_norm = ac_ts.div(total, axis=0).fillna(0)

    bottom = np.zeros(len(ac_ts_norm))
    for ac in AC_ORDER:
        ax.fill_between(ac_ts_norm.index, bottom, bottom + ac_ts_norm[ac].values,
                         alpha=0.85, color=AC_COLORS_TIMELINE[ac], label=ac)
        bottom += ac_ts_norm[ac].values
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weight")
    fam_col = FAMILY_COLORS.get(FAMILY_MAP.get(wt_col, "EW (benchmark)"), "black")
    ax.set_title(label, fontsize=9, fontweight="bold", color=fam_col)
    shade_crises(ax, alpha=0.07)

axes[0].legend(loc="upper right", fontsize=6.5, ncol=2)
fig.suptitle("Asset-Class Allocation Timeline (Monthly) — 6 Representative Strategies",
              fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

# ─── 2.10 Crisis snapshots ───
md("""### 2.10 — Crisis-Date Allocation Snapshots

Horizontal bar charts of portfolio weights at 3 crisis dates: 2009-03, 2020-03, 2022-10.
Same 6 representative strategies. Assets colored by asset class.
""")

code("""\
CRISIS_DATES = [
    pd.Timestamp("2009-03-31"),
    pd.Timestamp("2020-03-31"),
    pd.Timestamp("2022-10-31"),
]
CRISIS_LABELS = ["Mar 2009 (GFC trough)", "Mar 2020 (COVID)", "Oct 2022 (Rate Shock peak)"]

# Use SWITCH(LW) weights for "VMP(SWITCH(LW))"
SNAPSHOT6 = [
    ("EW",               "EW"),
    ("GMV(sample)",      "GMV(sample)"),
    ("GMV(ledoit_wolf)", "GMV(LW)"),
    ("MSR(ledoit_wolf)", "MSR(LW)"),
    ("MDP(ledoit_wolf)", "MDP(LW)"),
    ("SWITCH(ledoit_wolf)", "SWITCH(LW)"),
]

fig, axes = plt.subplots(3, 6, figsize=(22, 11))

for date_idx, (crisis_date, crisis_label) in enumerate(zip(CRISIS_DATES, CRISIS_LABELS)):
    for strat_idx, (wt_col, strat_label) in enumerate(SNAPSHOT6):
        ax = axes[date_idx, strat_idx]
        w = weights_cache.get(wt_col)
        if w is None:
            ax.axis("off"); continue
        # Find nearest available date
        available = w.index[w.index <= crisis_date]
        if len(available) == 0:
            ax.axis("off"); continue
        snap_date = available[-1]
        snap_w = w.loc[snap_date].dropna()
        snap_w = snap_w[snap_w > 0.001].sort_values(ascending=True)
        ac_colors_snap = [AC_COLORS.get(ASSET_CLASS.get(t, "Alternatives"), "gray")
                           for t in snap_w.index]
        ax.barh(range(len(snap_w)), snap_w.values * 100, color=ac_colors_snap, edgecolor="white", lw=0.2)
        ax.set_yticks(range(len(snap_w)))
        ax.set_yticklabels([t.replace(".US","").replace(".FOREX","").replace("-USD.CC","") for t in snap_w.index],
                            fontsize=5.5)
        ax.set_xlabel("Weight (%)", fontsize=6)
        if date_idx == 0:
            fam_col = FAMILY_COLORS.get(FAMILY_MAP.get(wt_col, "EW (benchmark)"), "black")
            ax.set_title(strat_label, fontsize=7.5, fontweight="bold", color=fam_col)
        if strat_idx == 0:
            ax.set_ylabel(crisis_label, fontsize=7, fontweight="bold")

fig.suptitle("Portfolio Composition at Crisis Dates — 6 Strategies", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
""")

# ─── 2.11 Strategy correlation matrix ───
md("""### 2.11 — 48×48 Strategy Correlation Matrix (Hierarchically Clustered)

Daily-return correlation matrix for all 48 strategies (24 base + 24 VMP variants).
Hierarchical clustering (Ward linkage) reorders rows/columns by cluster.
""")

code("""\
# Combine all 48 strategies into one DataFrame
all_rets = pd.concat([base.rename(columns={c: DISPLAY[c] for c in base.columns}),
                       vmp.rename(columns={c: c.replace("ledoit_wolf","LW").replace("(oas)","(OAS)") for c in vmp.columns})],
                      axis=1)

# Column family for color stripe
def col_family(name):
    raw = name.replace("VMP(","").rstrip(")")
    raw_orig = {v: k for k, v in DISPLAY.items()}.get(raw, raw)
    return FAMILY_MAP.get(raw_orig, "EW (benchmark)")

corr_mat = all_rets.corr()

# Hierarchical clustering on distance = 1 - corr
dist = squareform(1 - corr_mat.values, checks=False)
dist = np.clip(dist, 0, None)
link = linkage(dist, method="ward")
order = leaves_list(link)

corr_ordered = corr_mat.values[np.ix_(order, order)]
names_ordered = corr_mat.columns[order].tolist()

fig, ax = plt.subplots(figsize=(15, 13))
im = ax.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson Correlation")
ax.set_xticks(range(len(names_ordered)))
ax.set_yticks(range(len(names_ordered)))
ax.set_xticklabels(names_ordered, rotation=90, fontsize=4.5)
ax.set_yticklabels(names_ordered, fontsize=4.5)
ax.set_title("48×48 Strategy Return Correlation Matrix (Ward-Clustered)",
              fontsize=11, fontweight="bold")

# Family color stripe on left
fam_stripe_colors = [FAMILY_COLORS.get(col_family(n), "gray") for n in names_ordered]
for i, c in enumerate(fam_stripe_colors):
    ax.add_patch(plt.Rectangle((-1.6, i-0.5), 1, 1, color=c, clip_on=False, transform=ax.transData, zorder=5))

legend_patches = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_COLORS]
ax.legend(handles=legend_patches, fontsize=7, loc="lower right",
          bbox_to_anchor=(1.15, 0), framealpha=0.85)
fig.tight_layout()
plt.show()
""")

# ─── 2.12 Bootstrap Sharpe CIs ───
md("""### 2.12 — Bootstrap Sharpe Confidence Intervals

Block bootstrap (block size = 252 days, 10,000 resamples) for the top-10 strategies by Sharpe.
Forest plot with point estimates and 95% confidence intervals.
""")

code("""\
np.random.seed(42)

N_BOOT   = 10_000
BLOCK_SZ = 252

def block_bootstrap_sharpe(s, n_boot=N_BOOT, block_size=BLOCK_SZ):
    arr = s.dropna().values
    n   = len(arr)
    n_blocks = int(np.ceil(n / block_size))
    sharpes  = np.empty(n_boot)
    for b in range(n_boot):
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s+block_size] for s in starts])[:n]
        mu, sigma = sample.mean(), sample.std()
        sharpes[b] = mu / sigma * np.sqrt(TRADING_DAYS) if sigma > 0 else np.nan
    return sharpes

top10_strats = df_all.nlargest(10, "sharpe")[["col","display","sharpe","is_vmp"]].reset_index(drop=True)

print("Computing block bootstrap Sharpe CIs for top-10 strategies (may take ~30 seconds)...")
boot_results = []
for _, row in top10_strats.iterrows():
    s = vmp[row.col] if row.is_vmp else base[row.col]
    dist_boots = block_bootstrap_sharpe(s)
    lo, hi = np.nanpercentile(dist_boots, [2.5, 97.5])
    boot_results.append({"display": row.display, "point": row.sharpe, "lo": lo, "hi": hi})
    print(f"  {row.display:35s}  {row.sharpe:.3f}  [{lo:.3f}, {hi:.3f}]")

df_boot = pd.DataFrame(boot_results).sort_values("point")

fig, ax = plt.subplots(figsize=(9, 7))
for i, row in enumerate(df_boot.itertuples()):
    is_vmp_strat = "VMP" in row.display
    color = "#1f77b4" if is_vmp_strat else "#7f7f7f"
    ax.plot([row.lo, row.hi], [i, i], color=color, lw=2.5, alpha=0.8, solid_capstyle="butt")
    ax.scatter([row.point], [i], color=color, s=60, zorder=5)
    ax.text(row.hi + 0.005, i, f"{row.point:.3f}", va="center", fontsize=8)

ax.set_yticks(range(len(df_boot)))
ax.set_yticklabels(df_boot["display"].tolist(), fontsize=8.5)
ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
ax.text(1.002, len(df_boot)-0.5, "Sharpe=1.0", fontsize=7, color="dimgray")
ax.set_xlabel("Annualized Sharpe Ratio")
ax.set_title(f"Block Bootstrap Sharpe 95% CIs — Top-10 Strategies\\n"
             f"(block size={BLOCK_SZ} days, {N_BOOT:,} resamples)",
             fontsize=10, fontweight="bold")

vmp_patch = mpatches.Patch(color="#1f77b4", label="VMP variant")
base_patch = mpatches.Patch(color="#7f7f7f", label="Base strategy")
ax.legend(handles=[vmp_patch, base_patch], fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()
""")

# ─── 2.13 Pairwise Sharpe significance ───
md("""### 2.13 — Pairwise Sharpe-Difference Significance (Memmel 2003)

Five key contrasts tested with the Memmel (2003) / Jobson-Korkie paired Sharpe test.
H₀: Sharpe(A) = Sharpe(B).  Two-sided z-test.
""")

code("""\
from scipy.stats import norm as scipy_norm

def memmel_test(r1, r2):
    \"\"\"Memmel (2003) test for equality of Sharpe ratios.
    Returns (z_stat, p_value, delta_sharpe).
    Uses daily (non-annualized) Sharpe to match the original formulation.
    \"\"\"
    n  = min(len(r1), len(r2))
    r1 = np.asarray(r1.dropna().iloc[:n])
    r2 = np.asarray(r2.dropna().iloc[:n])
    mu1, mu2     = r1.mean(), r2.mean()
    sigma1, sigma2 = r1.std(), r2.std()
    rho = np.corrcoef(r1, r2)[0, 1]
    sr1 = mu1 / sigma1
    sr2 = mu2 / sigma2
    var = 2 * (1 - rho) + 0.5 * (sr1**2 + sr2**2) - sr1 * sr2 * rho
    var = max(var, 1e-12)
    z   = np.sqrt(n) * (sr1 - sr2) / np.sqrt(var)
    p   = 2 * scipy_norm.sf(abs(z))
    delta = (sr1 - sr2) * np.sqrt(TRADING_DAYS)  # annualized delta
    return z, p, delta

CONTRASTS = [
    ("VMP(EW)",               "EW",               "VMP(EW) vs EW"),
    ("VMP(MSR(ledoit_wolf))", "MSR(ledoit_wolf)",  "VMP(MSR(LW)) vs MSR(LW)"),
    ("MSR(ledoit_wolf)",      "MSR(sample)",       "MSR(LW) vs MSR(sample)"),
    (None,                    "SWITCH(ledoit_wolf)", "SWITCH(v2a) vs SWITCH(LW)"),
    ("VMP(SWITCH(ledoit_wolf))", "VMP(MSR(ledoit_wolf))", "VMP(SWITCH(LW)) vs VMP(MSR(LW))"),
]

results_mem = []
for colA, colB, label in CONTRASTS:
    sA = switch_v2a if colA is None else strategy_returns(colA)
    sB = strategy_returns(colB)
    common = sA.index.intersection(sB.index)
    z, p, delta = memmel_test(sA.reindex(common), sB.reindex(common))
    sh_A = ann_sharpe(sA)
    sh_B = ann_sharpe(sB)
    results_mem.append({
        "Contrast": label,
        "Sharpe(A)": sh_A, "Sharpe(B)": sh_B,
        "ΔSharpe": delta, "z": z, "p-value": p,
        "Significant": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")),
    })

df_mem = pd.DataFrame(results_mem)
df_mem[["Sharpe(A)","Sharpe(B)","ΔSharpe","z"]] = df_mem[["Sharpe(A)","Sharpe(B)","ΔSharpe","z"]].round(3)
df_mem["p-value"] = df_mem["p-value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
display(df_mem)
print("\\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
""")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.14.0",
}

with open(NB_PATH, "w") as f:
    nbformat.write(nb, f)

print(f"Written {NB_PATH}  ({len(cells)} cells, "
      f"{sum(1 for c in cells if c.cell_type=='code')} code / "
      f"{sum(1 for c in cells if c.cell_type=='markdown')} markdown)")
