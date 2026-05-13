"""
Black-Litterman experiment — adds 4 BL variants to the 16-strategy cache,
applies VMP overlay to all 20 strategies, and prints a 40-row paired table.

Four BL variants:
  BL-Eq(sample)  — equilibrium_only + sample_cov     (sanity check vs MSR)
  BL-Eq(LW)      — equilibrium_only + ledoit_wolf_cov (shrinkage equilibrium)
  BL-Mom(LW)     — momentum_views   + ledoit_wolf_cov (momentum-tilted)
  BL-Rev(LW)     — mean_reversion_views + ledoit_wolf_cov (reversion-tilted)

Saves:
    data/cache/portfolio_returns/20strategies_2008_2026.parquet
    data/cache/portfolio_returns/20strategies_vmp_2008_2026.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, sample_cov
from aiam.estimators.views import equilibrium_only, mean_reversion_views, momentum_views
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.black_litterman import BlackLitterman

PRICES_CACHE = "data/cache/prices_30.parquet"
BASE_CACHE = "data/cache/portfolio_returns/16strategies_2008_2026.parquet"
OUT_20 = "data/cache/portfolio_returns/20strategies_2008_2026.parquet"
VMP_OUT = "data/cache/portfolio_returns/20strategies_vmp_2008_2026.parquet"

START = "2008-01-01"
END = "2026-04-30"

BL_VARIANTS: dict[str, BlackLitterman] = {
    "BL-Eq(sample)": BlackLitterman(
        view_generator=equilibrium_only,
        cov_estimator=sample_cov,
    ),
    "BL-Eq(LW)": BlackLitterman(
        view_generator=equilibrium_only,
        cov_estimator=ledoit_wolf_cov,
    ),
    "BL-Mom(LW)": BlackLitterman(
        view_generator=momentum_views,
        cov_estimator=ledoit_wolf_cov,
    ),
    "BL-Rev(LW)": BlackLitterman(
        view_generator=mean_reversion_views,
        cov_estimator=ledoit_wolf_cov,
    ),
}


def _fmt_row(name: str, s: dict) -> dict:
    return {
        "strategy": name,
        "ann_return": f"{s['annualized_return']:.2%}",
        "ann_vol": f"{s['annualized_volatility']:.2%}",
        "sharpe": f"{s['sharpe_ratio']:.3f}",
        "max_dd": f"{s['max_drawdown']:.2%}",
        "calmar": f"{s['calmar_ratio']:.3f}",
    }


def _sep() -> dict:
    return {k: "" for k in ("strategy", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar")}


def main() -> None:
    # ── 1. Load existing 16-strategy cache ──────────────────────────────────
    print("Loading 16-strategy cache...")
    wide16 = pd.read_parquet(BASE_CACHE)

    # ── 2. Run horse race for BL variants ───────────────────────────────────
    print("Loading panel data...")
    prices = pd.read_parquet(PRICES_CACHE)
    panel = Panel({"prices": prices})

    bl_returns: dict[str, pd.Series] = {}
    for name, strategy in BL_VARIANTS.items():
        print(f"  Running {name}...", end="", flush=True)
        result = run_horse_race(panel, strategy, start=START, end=END)
        bl_returns[name] = result["portfolio_returns"].dropna()
        print(f"  Sharpe={result['stats']['sharpe_ratio']:.3f}")

    # ── 3. Merge and save 20-strategy cache ─────────────────────────────────
    wide20 = wide16.copy()
    for name, series in bl_returns.items():
        wide20[name] = series

    wide20.to_parquet(OUT_20)
    print(f"\nSaved 20-strategy cache → {OUT_20}")

    # ── 4. VMP overlay for all 20 strategies ────────────────────────────────
    all_strategies = wide20.columns.tolist()
    vmp_series: dict[str, pd.Series] = {}
    for strat in all_strategies:
        base = wide20[strat].dropna()
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(base)

    pd.DataFrame(vmp_series).to_parquet(VMP_OUT)
    print(f"Saved 20-strategy VMP cache → {VMP_OUT}")

    # ── 5. Sanity checks — BL vol/VMP ratio ─────────────────────────────────
    print("\nSanity checks — BL variants VMP vol ratio:")
    for strat in BL_VARIANTS:
        base = wide20[strat].dropna()
        base_lrvol = base.std() * np.sqrt(252)
        vmp = vmp_series[f"VMP({strat})"]
        vmp_rvol = vmp.std() * np.sqrt(252)
        ratio = vmp_rvol / base_lrvol
        status = "PASS" if abs(ratio - 1.0) <= 0.10 else "FAIL"
        print(f"  {strat:20s}  base={base_lrvol:.2%}  vmp={vmp_rvol:.2%}  ratio={ratio:.3f}  {status}")

    # ── 6. 40-row paired table ───────────────────────────────────────────────
    rows = []
    for strat in all_strategies:
        base = wide20[strat].dropna()
        vmp = vmp_series[f"VMP({strat})"]
        rows.append(_fmt_row(strat, performance_stats(base)))
        rows.append(_fmt_row(f"VMP({strat})", performance_stats(vmp)))
        rows.append(_sep())

    df_table = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 88)
    print("40-row table — 20 strategies × 2 (original + VMP), 2008-01 to 2026-04")
    print("BL params: tau=0.05, delta=2.5, lookback=252, prior=equal-weight")
    print("VMP params: lookback=21d, lag=1d, clip=(0.25,1.5), target_vol=strategy LR vol")
    print("=" * 88)
    print(df_table.to_string())
    print()


if __name__ == "__main__":
    main()
