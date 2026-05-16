#!/usr/bin/env python3
"""Pass-4 Part 1: Compute canonical metrics for all 62 strategies.

Outputs:
  data/cache/appendix_a_canonical.csv
  data/cache/regime_conditional_sharpe_29.parquet
  data/cache/memmel_hrp.json
  data/cache/test_period_top5.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = pd.read_parquet("data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet")
VMP  = pd.read_parquet("data/cache/portfolio_returns/31strategies_vmp_29assets_2003_2026.parquet")
SW   = pd.read_parquet("data/cache/portfolio_returns/switch_v2a_oos_29assets.parquet")

# Regime signal — fill forward from monthly to daily
REGIME_RAW = pd.read_parquet("data/cache/regime_signals_2003_2026.parquet")["dominant_regime"].dropna()
REGIME = REGIME_RAW.resample("B").ffill().reindex(BASE.index, method="ffill")

WEIGHT_DIR = Path("data/cache/portfolio_weights")

# Asset-class cost map (one-way bps)
COST_MAP = {
    # Fixed income ETFs — 2 bps
    "SHY.US": 2, "IEF.US": 2, "TLT.US": 2, "AGG.US": 2, "HYG.US": 2,
    # Broad equity & sector ETFs — 3 bps
    "SPY.US": 3, "IWM.US": 3,
    "XLK.US": 3, "XLF.US": 3, "XLE.US": 3, "XLV.US": 3, "XLP.US": 3, "XLU.US": 3,
    # Single stocks, intl equity, commodity/FX — 5 bps
    "AAPL.US": 5, "MSFT.US": 5, "GOOGL.US": 5, "NVDA.US": 5,
    "JPM.US": 5, "JNJ.US": 5, "XOM.US": 5, "WMT.US": 5,
    "EFA.US": 5, "EEM.US": 5, "FXI.US": 5,
    "GLD.US": 5, "SLV.US": 5, "DBC.US": 5, "USO.US": 5, "EURUSD": 5,
}

# Mapping: parquet column name → weight file name (without _29assets_2003_2026.parquet suffix)
WEIGHT_FILE = {
    "EW":                 "EW",
    "GMV(sample)":        "GMV_sample",
    "GMV(ledoit_wolf)":   "GMV_ledoit_wolf",
    "GMV(oas)":           "GMV_oas",
    "MSR(sample)":        "MSR_sample",
    "MSR(ledoit_wolf)":   "MSR_ledoit_wolf",
    "MDP(sample)":        "MDP_sample",
    "MDP(ledoit_wolf)":   "MDP_ledoit_wolf",
    "RP(sample)":         "RP_sample",
    "RP(ledoit_wolf)":    "RP_ledoit_wolf",
    "HRP(sample)":        "HRP_sample",
    "HRP(ledoit_wolf)":   "HRP_ledoit_wolf",
    "SWITCH(sample)":     "SWITCH_sample",
    "SWITCH(ledoit_wolf)":"SWITCH_ledoit_wolf",
    "TSMOM(12m)":         "TSMOM_12m",
    "TSMOM(6m)":          "TSMOM_6m",
    "BL-Eq(sample)":      "BL-Eq_sample",
    "BL-Eq(LW)":          "BL-Eq_LW",
    "BL-Mom(LW)":         "BL-Mom_LW",
    "BL-Rev(LW)":         "BL-Rev_LW",
    "FF3-Mom":            "FF3-Mom",
    "FF3-LowVol":         "FF3-LowVol",
    "FF3-Quality":        "FF3-Quality",
    "FF3-Multi":          "FF3-Multi",
    "MSR_C(ledoit_wolf)": "MSR_C_ledoit_wolf",
    "MSR_C(sample)":      "MSR_C_sample",
    "MVO_C(ledoit_wolf)": "MVO_C_ledoit_wolf",
    "MVO_C(sample)":      "MVO_C_sample",
    "TSMOM-LS(12m)":      "TSMOM-LS_12m",
    "BL-Mom-LS(LW)":      "BL-Mom-LS_LW",
    "FF3-Mom-LS":         "FF3-Mom-LS",
}

FAMILY_MAP = {
    "EW":                 "Classical MV",
    "GMV(sample)":        "Classical MV",
    "GMV(ledoit_wolf)":   "Classical MV",
    "GMV(oas)":           "Classical MV",
    "MSR(sample)":        "Classical MV",
    "MSR(ledoit_wolf)":   "Classical MV",
    "MDP(sample)":        "Diversification",
    "MDP(ledoit_wolf)":   "Diversification",
    "RP(sample)":         "Diversification",
    "RP(ledoit_wolf)":    "Diversification",
    "HRP(sample)":        "Diversification",
    "HRP(ledoit_wolf)":   "Diversification",
    "SWITCH(sample)":     "Regime Switch",
    "SWITCH(ledoit_wolf)":"Regime Switch",
    "TSMOM(12m)":         "TS Momentum",
    "TSMOM(6m)":          "TS Momentum",
    "BL-Eq(sample)":      "Black-Litterman",
    "BL-Eq(LW)":          "Black-Litterman",
    "BL-Mom(LW)":         "Black-Litterman",
    "BL-Rev(LW)":         "Black-Litterman",
    "FF3-Mom":            "Factor",
    "FF3-LowVol":         "Factor",
    "FF3-Quality":        "Factor",
    "FF3-Multi":          "Factor",
    "MSR_C(ledoit_wolf)": "Constrained MV",
    "MSR_C(sample)":      "Constrained MV",
    "MVO_C(ledoit_wolf)": "Constrained MV",
    "MVO_C(sample)":      "Constrained MV",
    "TSMOM-LS(12m)":      "Long-Short",
    "BL-Mom-LS(LW)":      "Long-Short",
    "FF3-Mom-LS":         "Long-Short",
}

# ── Helper functions ───────────────────────────────────────────────────────────
def ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252)

def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1

def ann_vol(r: pd.Series) -> float:
    return r.dropna().std() * np.sqrt(252)

def max_drawdown(r: pd.Series) -> float:
    cum = (1 + r.dropna()).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    if mdd == 0:
        return np.nan
    return ann_return(r) / abs(mdd)

def hit_ratio(r: pd.Series) -> float:
    """Fraction of calendar months with positive return."""
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return (monthly > 0).mean() * 100

def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Average daily one-way turnover (fraction, not %)."""
    dw = weights.fillna(0).diff().abs()
    return dw.sum(axis=1) / 2  # one-way = half of L1 weight change

def load_weights(col: str) -> pd.DataFrame | None:
    stem = WEIGHT_FILE.get(col)
    if stem is None:
        return None
    path = WEIGHT_DIR / f"{stem}_29assets_2003_2026.parquet"
    if not path.exists():
        print(f"  WARNING: weight file missing: {path.name}")
        return None
    return pd.read_parquet(path)

def net_sharpe_flat(r: pd.Series, weights: pd.DataFrame, cost_bps: float = 10.0) -> float:
    """Sharpe after flat bps cost per unit one-way turnover."""
    r = r.dropna()
    w = weights.fillna(0).reindex(r.index, method="ffill")
    to = compute_turnover(w)
    net = r - to * (cost_bps / 10000)
    return ann_sharpe(net)

def net_sharpe_stratified(r: pd.Series, weights: pd.DataFrame) -> float:
    """Sharpe after asset-class-stratified costs."""
    r = r.dropna()
    w = weights.fillna(0).reindex(r.index, method="ffill")
    dw = w.diff().abs() / 2  # one-way
    # Cost per day: sum over assets of |Δw_i| * cost_i
    cost_vec = pd.Series(COST_MAP)
    # align tickers
    common = w.columns.intersection(cost_vec.index)
    cost_series = (dw[common] * cost_vec[common]).sum(axis=1) / 10000
    net = r - cost_series
    return ann_sharpe(net)

# ── Compute metrics for all 62 strategies ─────────────────────────────────────
print("Computing canonical metrics for all 62 strategies...")
records = []

for col in BASE.columns:
    r = BASE[col]
    weights = load_weights(col)

    to_pct = np.nan
    net10  = np.nan
    netstrat = np.nan

    if weights is not None:
        to = compute_turnover(weights)
        to_pct = to.reindex(r.index).dropna().mean() * 100
        net10    = net_sharpe_flat(r, weights, 10.0)
        netstrat = net_sharpe_stratified(r, weights)

    row = {
        "strategy":   col,
        "display":    col.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)"),
        "family":     FAMILY_MAP.get(col, "Unknown"),
        "is_vmp":     False,
        "ann_ret":    ann_return(r),
        "ann_vol":    ann_vol(r),
        "sharpe":     ann_sharpe(r),
        "hit_pct":    hit_ratio(r),
        "max_dd":     max_drawdown(r),
        "calmar":     calmar(r),
        "turnover":   to_pct,
        "net_10bps":  net10,
        "net_strat":  netstrat,
    }
    records.append(row)

    # VMP variant
    vcol = f"VMP({col})"
    if vcol in VMP.columns:
        vr = VMP[vcol]
        vnet10   = net_sharpe_flat(vr, weights, 10.0) if weights is not None else np.nan
        vnetstrat = net_sharpe_stratified(vr, weights) if weights is not None else np.nan
        vrow = {
            "strategy":   vcol,
            "display":    vcol.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)"),
            "family":     FAMILY_MAP.get(col, "Unknown"),
            "is_vmp":     True,
            "ann_ret":    ann_return(vr),
            "ann_vol":    ann_vol(vr),
            "sharpe":     ann_sharpe(vr),
            "hit_pct":    "—",  # VMP hit% not applicable
            "max_dd":     max_drawdown(vr),
            "calmar":     calmar(vr),
            "turnover":   to_pct,
            "net_10bps":  vnet10,
            "net_strat":  vnetstrat,
        }
        records.append(vrow)

df = pd.DataFrame(records)
df.to_csv("data/cache/appendix_a_canonical.csv", index=False)
print(f"  Saved appendix_a_canonical.csv — {len(df)} rows")

# ── Summary stats ──────────────────────────────────────────────────────────────
print("\nTop 10 by gross Sharpe (all 62):")
top10 = df.nlargest(10, "sharpe")[["display", "sharpe", "ann_ret", "max_dd", "net_10bps"]]
print(top10.to_string(index=False))

print("\nTop 10 by net Sharpe 10bps (artifact excluded):")
non_art = df[df.strategy != "VMP(GMV(sample))"]
top10net = non_art.nlargest(10, "net_10bps")[["display", "sharpe", "net_10bps", "turnover"]]
print(top10net.to_string(index=False))

print("\nBottom 5 by Sharpe (base strategies only):")
base_only = df[~df.is_vmp]
bot5 = base_only.nsmallest(5, "sharpe")[["display", "sharpe", "ann_ret"]]
print(bot5.to_string(index=False))

print("\nTop 5 by degradation (base strategies only):")
top5deg = base_only.nlargest(5, "degradation" if "degradation" in base_only else "sharpe")
deg = (base_only.assign(deg=lambda x: x.sharpe - x.net_10bps)
       .nlargest(5, "deg")[["display", "sharpe", "net_10bps", "turnover"]])
deg["degradation"] = deg.sharpe - deg.net_10bps
print(deg.to_string(index=False))

# ── Test-period top-5 ──────────────────────────────────────────────────────────
print("\n=== Part 1e: Test-period (2023+) top-5 Sharpe ===")
test_start = "2023-01-01"
all_rets = pd.concat([BASE, VMP], axis=1)
test_rets = all_rets[all_rets.index >= test_start]
test_sharpes = test_rets.apply(ann_sharpe).sort_values(ascending=False)
top5_test = test_sharpes.head(7)
print(top5_test)

test_df = pd.DataFrame({
    "strategy": top5_test.index,
    "test_sharpe": top5_test.values,
    "display": [s.replace("ledoit_wolf", "LW").replace("(oas)", "(OAS)") for s in top5_test.index],
})
test_df.to_csv("data/cache/test_period_top5.csv", index=False)
print("  Saved test_period_top5.csv")

# ── Part 1b: Regime-conditional Sharpe for 12 baselines ──────────────────────
print("\n=== Part 1b: Regime-conditional Sharpe (12 baselines × 8 regimes) ===")

HEATMAP_STRATS = [
    "EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)",
    "MSR(sample)", "MSR(ledoit_wolf)",
    "MDP(sample)", "MDP(ledoit_wolf)",
    "RP(sample)", "RP(ledoit_wolf)",
    "HRP(sample)", "HRP(ledoit_wolf)",
]

regimes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
hm_rows = {}
ndays_per_regime = {}

for k in regimes:
    mask = (REGIME == k).reindex(BASE.index, fill_value=False)
    ndays_per_regime[int(k)] = int(mask.sum())

for col in HEATMAP_STRATS:
    r = BASE[col]
    row = {}
    for k in regimes:
        mask = (REGIME == k).reindex(BASE.index, fill_value=False)
        sub = r[mask].dropna()
        row[int(k)] = ann_sharpe(sub) if len(sub) >= 21 else np.nan
    hm_rows[col.replace("ledoit_wolf", "LW")] = row

df_hm = pd.DataFrame(hm_rows).T
df_hm.to_parquet("data/cache/regime_conditional_sharpe_29.parquet")
print(f"  Regime-conditional Sharpe shape: {df_hm.shape}")
print(f"  Days per regime: {ndays_per_regime}")

# Best strategy per regime
print("\nR0 (Expansion) conditional Sharpes:")
r0_sharpes = {col: hm_rows[col.replace("ledoit_wolf","LW")][0] for col in HEATMAP_STRATS}
r0_best = max(r0_sharpes, key=lambda k: r0_sharpes[k] if not np.isnan(r0_sharpes[k]) else -999)
print(f"  Best: {r0_best} = {r0_sharpes[r0_best]:.4f}")

print("R5 (Low & Contracting) conditional Sharpes:")
r5_sharpes = {col: hm_rows[col.replace("ledoit_wolf","LW")][5] for col in HEATMAP_STRATS}
r5_best = max(r5_sharpes, key=lambda k: r5_sharpes[k] if not np.isnan(r5_sharpes[k]) else -999)
print(f"  Best: {r5_best} = {r5_sharpes[r5_best]:.4f}")

print("\nFull R0 conditional Sharpes (sorted):")
r0_sorted = sorted(r0_sharpes.items(), key=lambda x: -x[1])
for name, v in r0_sorted:
    print(f"  {name:25s}: {v:.4f}")

print("\nFull R5 conditional Sharpes (sorted):")
r5_sorted = sorted(r5_sharpes.items(), key=lambda x: -x[1])
for name, v in r5_sorted:
    print(f"  {name:25s}: {v:.4f}")

# ── Part 1c: Memmel test for HRP(sample) vs HRP(LW) ──────────────────────────
print("\n=== Part 1c: Memmel (2003) paired test: HRP(sample) vs HRP(LW) ===")

hrp_sample = BASE["HRP(sample)"].dropna()
hrp_lw     = BASE["HRP(ledoit_wolf)"].reindex(hrp_sample.index).dropna()
common_idx = hrp_sample.index.intersection(hrp_lw.index)
r1 = hrp_sample[common_idx]
r2 = hrp_lw[common_idx]

T = len(r1)
s1 = ann_sharpe(r1)
s2 = ann_sharpe(r2)
delta_s = s1 - s2

# Memmel (2003) test statistic
# Based on Jobson-Korkie (1981) with Memmel correction
mu1, mu2 = r1.mean(), r2.mean()
sig1, sig2 = r1.std(), r2.std()
sig12 = np.cov(r1, r2)[0, 1]
rho = np.corrcoef(r1, r2)[0, 1]

# JK numerator: sr1 - sr2 (unannualized)
sr1_unann = mu1 / sig1
sr2_unann = mu2 / sig2
diff = sr1_unann - sr2_unann

# JK variance (Memmel 2003 eq. 8)
v = (1 / T) * (2 * (1 - rho**2) +
               (sr1_unann**2 + sr2_unann**2 - 2 * sr1_unann * sr2_unann * rho) / 2
              )
se = np.sqrt(max(v, 1e-12))
z = diff / se
p = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"  HRP(sample) Sharpe = {s1:.4f}")
print(f"  HRP(LW)     Sharpe = {s2:.4f}")
print(f"  Delta (sample - LW) = {delta_s:.4f}")
print(f"  Memmel z = {z:.4f}")
print(f"  p-value  = {p:.4f}")
print(f"  T (observations) = {T}")

memmel_result = {
    "hrp_sample_sharpe": float(s1),
    "hrp_lw_sharpe":     float(s2),
    "delta":             float(delta_s),
    "z_stat":            float(z),
    "p_value":           float(p),
    "n_obs":             int(T),
    "branch":            "2b" if p >= 0.10 else ("2a" if delta_s > 0 else "2b"),
}

with open("data/cache/memmel_hrp.json", "w") as f:
    json.dump(memmel_result, f, indent=2)
print(f"  Saved memmel_hrp.json")
print(f"  Decision branch: {'2a (significant)' if p < 0.10 else '2b (near-invariance)'}")

# ── Part 1d: Verify long-short strategies ─────────────────────────────────────
print("\n=== Part 1d: Long-short strategies check ===")
ls_strats = ["TSMOM-LS(12m)", "BL-Mom-LS(LW)", "FF3-Mom-LS"]
ls_vmp = [f"VMP({s})" for s in ls_strats]
for col in ls_strats + ls_vmp:
    src = BASE if not col.startswith("VMP") else VMP
    if col in src.columns:
        r = src[col]
        print(f"  {col:30s}: Sharpe={ann_sharpe(r):.4f}, AnnRet={ann_return(r):.2%}")
    else:
        print(f"  MISSING: {col}")

# ── Additional Memmel: MSR(LW) vs MSR(sample) ─────────────────────────────────
print("\n=== Memmel: MSR(LW) vs MSR(sample) ===")
msr_s = BASE["MSR(sample)"].dropna()
msr_lw = BASE["MSR(ledoit_wolf)"].reindex(msr_s.index).dropna()
common_idx2 = msr_s.index.intersection(msr_lw.index)
r1m, r2m = msr_s[common_idx2], msr_lw[common_idx2]
T2 = len(r1m)
mu1m, mu2m = r1m.mean(), r2m.mean()
sig1m, sig2m = r1m.std(), r2m.std()
rho_m = np.corrcoef(r1m, r2m)[0, 1]
sr1m = mu1m / sig1m
sr2m = mu2m / sig2m
vm = (1/T2) * (2*(1-rho_m**2) + (sr1m**2 + sr2m**2 - 2*sr1m*sr2m*rho_m)/2)
sem = np.sqrt(max(vm, 1e-12))
# z for MSR(LW) > MSR(sample)
z_msr = (sr2m - sr1m) / sem  # LW - sample (testing LW is better)
p_msr = 2 * (1 - stats.norm.cdf(abs(z_msr)))
print(f"  MSR(sample) Sharpe = {ann_sharpe(msr_s):.4f}")
print(f"  MSR(LW)     Sharpe = {ann_sharpe(msr_lw):.4f}")
print(f"  Delta (LW - sample) = {ann_sharpe(msr_lw) - ann_sharpe(msr_s):.4f}")
print(f"  Memmel z = {z_msr:.4f}, p = {p_msr:.4f}")

# ── SWITCH(v2a) full-sample Sharpe ────────────────────────────────────────────
print("\n=== SWITCH(v2a) Sharpe ===")
sw_orig = SW["SWITCH_v2a_original"].reindex(BASE.index).dropna()
sw_train = SW["SWITCH_v2a_train_only"].reindex(BASE.index).dropna()
switch_lw = BASE["SWITCH(ledoit_wolf)"]
print(f"  SWITCH_v2a_original Sharpe  = {ann_sharpe(sw_orig):.4f}")
print(f"  SWITCH_v2a_train_only Sharpe = {ann_sharpe(sw_train):.4f}")
print(f"  SWITCH(LW) Sharpe           = {ann_sharpe(switch_lw):.4f}")
print(f"  SWITCH(sample) Sharpe       = {ann_sharpe(BASE['SWITCH(sample)']):.4f}")

# VMP(SWITCH(LW)) and VMP(SWITCH(sample))
print(f"  VMP(SWITCH(LW)) Sharpe      = {ann_sharpe(VMP['VMP(SWITCH(ledoit_wolf))']):.4f}")
print(f"  VMP(SWITCH(sample)) Sharpe  = {ann_sharpe(VMP['VMP(SWITCH(sample))']):.4f}")

# ── Full canonical stats summary for key strategies ───────────────────────────
print("\n=== Full stats for key strategies ===")
key_strats = [
    ("GMV(sample)", BASE), ("GMV(ledoit_wolf)", BASE), ("GMV(oas)", BASE),
    ("MSR(sample)", BASE), ("MSR(ledoit_wolf)", BASE),
    ("MDP(sample)", BASE), ("MDP(ledoit_wolf)", BASE),
    ("HRP(sample)", BASE), ("HRP(ledoit_wolf)", BASE),
    ("TSMOM(12m)", BASE), ("TSMOM(6m)", BASE),
    ("BL-Mom(LW)", BASE), ("VMP(BL-Mom(LW))", VMP),
    ("BL-Eq(sample)", BASE), ("BL-Eq(LW)", BASE),
    ("BL-Rev(LW)", BASE),
    ("FF3-LowVol", BASE), ("VMP(FF3-LowVol)", VMP),
    ("SWITCH(sample)", BASE), ("SWITCH(ledoit_wolf)", BASE),
    ("VMP(MSR(sample))", VMP), ("VMP(MSR(ledoit_wolf))", VMP),
    ("VMP(MDP(sample))", VMP), ("VMP(MDP(ledoit_wolf))", VMP),
    ("VMP(SWITCH(sample))", VMP), ("VMP(SWITCH(ledoit_wolf))", VMP),
    ("TSMOM-LS(12m)", BASE), ("BL-Mom-LS(LW)", BASE), ("FF3-Mom-LS", BASE),
    ("VMP(TSMOM-LS(12m))", VMP), ("VMP(BL-Mom-LS(LW))", VMP), ("VMP(FF3-Mom-LS)", VMP),
]
print(f"{'Strategy':35s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>8s} {'MaxDD':>9s} {'Calmar':>8s}")
for col, src in key_strats:
    if col not in src.columns:
        continue
    r = src[col]
    print(f"{col:35s} {ann_return(r):>8.2%} {ann_vol(r):>8.2%} {ann_sharpe(r):>8.4f} {max_drawdown(r):>9.2%} {calmar(r):>8.4f}")

print("\nDone. All canonical metrics computed.")
