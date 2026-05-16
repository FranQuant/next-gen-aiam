"""Generate two new figures for revision pass 5.

1. asset_class_allocation_timeline.png — 2×2 grid of stacked-area charts
2. rolling_sharpe_small_multiples.png — 2×3 grid of rolling Sharpe panels
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

WEIGHTS_DIR = Path("/Users/frasagui/Projects/next-gen-aiam/data/cache/portfolio_weights")
RETURNS_DIR = Path("/Users/frasagui/Projects/next-gen-aiam/data/cache/portfolio_returns")
FIGS_DIR = Path("/Users/frasagui/Projects/next-gen-aiam/docs/figures")

# ── Asset-class mapping ────────────────────────────────────────────────────────
ASSET_CLASSES = {
    "US Large Cap": ["AAPL.US","MSFT.US","GOOGL.US","NVDA.US","JPM.US","JNJ.US","XOM.US","WMT.US"],
    "US Sectors":   ["XLK.US","XLF.US","XLE.US","XLV.US","XLP.US","XLU.US"],
    "US Broad Eq":  ["SPY.US","IWM.US"],
    "Intl Equity":  ["EFA.US","EEM.US","FXI.US"],
    "Fixed Income": ["SHY.US","IEF.US","TLT.US","AGG.US","HYG.US"],
    "Cmdty/FX":     ["GLD.US","SLV.US","DBC.US","USO.US","EURUSD.FOREX"],
}

AC_COLORS = {
    "US Large Cap": "#2166ac",
    "US Sectors":   "#4dac26",
    "US Broad Eq":  "#74add1",
    "Intl Equity":  "#f46d43",
    "Fixed Income": "#d1e5f0",
    "Cmdty/FX":     "#fdae61",
}

# ── FIGURE 1 — Asset-class allocation timeline ─────────────────────────────────
def load_weights(fname):
    return pd.read_parquet(WEIGHTS_DIR / fname).resample("ME").last().fillna(0)

strat_files = {
    "EW":        "EW_29assets_2003_2026.parquet",
    "MDP(LW)":   "MDP_ledoit_wolf_29assets_2003_2026.parquet",
    "MSR(LW)":   "MSR_ledoit_wolf_29assets_2003_2026.parquet",
    "SWITCH(LW)":"SWITCH_ledoit_wolf_29assets_2003_2026.parquet",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for ax, (title, fname) in zip(axes, strat_files.items()):
    w = load_weights(fname)
    # Build asset-class weight series
    ac_series = {}
    all_assigned = []
    for ac, tickers in ASSET_CLASSES.items():
        cols = [t for t in tickers if t in w.columns]
        if cols:
            ac_series[ac] = w[cols].sum(axis=1)
            all_assigned.extend(cols)
        else:
            ac_series[ac] = pd.Series(0.0, index=w.index)

    # Normalise so they sum to 1 (should already, but just in case)
    total = sum(ac_series.values())
    for ac in ac_series:
        ac_series[ac] = ac_series[ac] / total.replace(0, 1)

    labels = list(ac_series.keys())
    colors = [AC_COLORS[ac] for ac in labels]
    data = np.array([ac_series[ac].values for ac in labels])

    ax.stackplot(w.index, data, labels=labels, colors=colors, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Portfolio weight", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # shade crisis regions
    crises = [("2008-09-15","2009-03-31"), ("2020-02-19","2020-04-30"), ("2022-01-01","2022-12-31")]
    for s, e in crises:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15, linewidth=0)

# shared legend
handles = [mpatches.Patch(color=AC_COLORS[ac], label=ac) for ac in ASSET_CLASSES]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Asset-Class Allocation Over Time (2003–2026)\nMonthly rebalance, grey shading = GFC / COVID / 2022 rate shock",
             fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
out1 = FIGS_DIR / "asset_class_allocation_timeline.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── FIGURE 2 — Rolling 12-month Sharpe small multiples ──────────────────────────
returns_all = pd.read_parquet(
    RETURNS_DIR / "31strategies_29assets_2003_2026.parquet"
)
returns_vmp = pd.read_parquet(
    RETURNS_DIR / "31strategies_vmp_29assets_2003_2026.parquet"
)
rets = pd.concat([returns_all, returns_vmp], axis=1)

# Check available columns
rets.columns = rets.columns.str.strip()

# 6 focal strategies — use exact column names from the parquet files
matched = {
    "EW":           "EW",
    "MDP(LW)":      "MDP(ledoit_wolf)",
    "MSR(LW)":      "MSR(ledoit_wolf)",
    "SWITCH(LW)":   "SWITCH(ledoit_wolf)",
    "VMP(MDP(LW))": "VMP(MDP(ledoit_wolf))",
    "HRP(LW)":      "HRP(ledoit_wolf)",
}

print("Using columns:", list(matched.values()))

WINDOW = 252  # 12-month rolling window (trading days)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
axes = axes.flatten()

colors_focal = ["#1f77b4","#2ca02c","#d62728","#ff7f0e","#9467bd","#8c564b"]

for ax, (label, col), color in zip(axes, matched.items(), colors_focal):
    r = rets[col].dropna()
    # rolling annualised Sharpe (rf=0)
    roll_sharpe = (r.rolling(WINDOW).mean() / r.rolling(WINDOW).std()) * np.sqrt(252)
    ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=1.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Sharpe=1.0")
    ax.axhline(0.0, color="grey", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylim(-1, 3.5)
    ax.set_ylabel("12-m Rolling Sharpe", fontsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # shade crises
    for s, e in [("2008-09-15","2009-03-31"), ("2020-02-19","2020-04-30"), ("2022-01-01","2022-12-31")]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15, linewidth=0)
    ax.tick_params(axis="x", labelsize=7, rotation=30)

# shared note
fig.suptitle(
    "12-Month Rolling Sharpe Ratio — Selected Strategies (2003–2026)\n"
    "Dashed line = Sharpe 1.0; grey shading = GFC / COVID / 2022 rate shock",
    fontsize=11
)
plt.tight_layout()
out2 = FIGS_DIR / "rolling_sharpe_small_multiples.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
