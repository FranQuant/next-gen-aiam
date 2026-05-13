"""
TSMOM experiment — runs Time-Series Momentum (Moskowitz-Ooi-Pedersen 2012) horse race,
appends to the 14-strategy cache, and applies the VMP overlay to all 16 strategies.

Two variants:
  TSMOM(12m) — signal_lookback=252, vol_lookback=63  (MOP-2012 standard)
  TSMOM(6m)  — signal_lookback=126, vol_lookback=63  (more responsive)

Saves:
    data/cache/portfolio_returns/16strategies_2008_2026.parquet
    data/cache/portfolio_returns/16strategies_vmp_2008_2026.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.tsmom import TSMOM

PRICES_CACHE = "data/cache/prices_30.parquet"
BASE_CACHE = "data/cache/portfolio_returns/14strategies_2008_2026.parquet"
OUT_16 = "data/cache/portfolio_returns/16strategies_2008_2026.parquet"
VMP_OUT = "data/cache/portfolio_returns/16strategies_vmp_2008_2026.parquet"

START = "2008-01-01"
END = "2026-04-30"

TSMOM_VARIANTS: dict[str, TSMOM] = {
    "TSMOM(12m)": TSMOM(signal_lookback=252, vol_lookback=63),
    "TSMOM(6m)":  TSMOM(signal_lookback=126, vol_lookback=63),
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
    return {"strategy": "", "ann_return": "", "ann_vol": "", "sharpe": "", "max_dd": "", "calmar": ""}


def main() -> None:
    # ── 1. Load existing 14-strategy cache ──────────────────────────────────
    print("Loading 14-strategy cache...")
    wide14 = pd.read_parquet(BASE_CACHE)

    # ── 2. Run horse race for TSMOM variants ────────────────────────────────
    print("Loading panel data...")
    prices = pd.read_parquet(PRICES_CACHE)
    panel = Panel({"prices": prices})

    tsmom_returns: dict[str, pd.Series] = {}
    for name, strategy in TSMOM_VARIANTS.items():
        print(f"  Running {name}...", end="", flush=True)
        result = run_horse_race(panel, strategy, start=START, end=END)
        tsmom_returns[name] = result["portfolio_returns"].dropna()
        print(f"  Sharpe={result['stats']['sharpe_ratio']:.3f}")

    # ── 3. Merge and save 16-strategy cache ─────────────────────────────────
    wide16 = wide14.copy()
    for name, series in tsmom_returns.items():
        wide16[name] = series

    wide16.to_parquet(OUT_16)
    print(f"\nSaved 16-strategy cache → {OUT_16}")

    # ── 4. VMP overlay for all 16 strategies ────────────────────────────────
    all_strategies = wide16.columns.tolist()
    vmp_series: dict[str, pd.Series] = {}
    for strat in all_strategies:
        base = wide16[strat].dropna()
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(base)

    pd.DataFrame(vmp_series).to_parquet(VMP_OUT)
    print(f"Saved 16-strategy VMP cache → {VMP_OUT}")

    # ── 5. Sanity checks ────────────────────────────────────────────────────
    print("\nSanity checks — TSMOM variants:")
    for strat in TSMOM_VARIANTS:
        base = wide16[strat].dropna()
        base_lrvol = base.std() * np.sqrt(252)
        vmp = vmp_series[f"VMP({strat})"]
        vmp_rvol = vmp.std() * np.sqrt(252)
        ratio = vmp_rvol / base_lrvol
        status = "PASS" if abs(ratio - 1.0) <= 0.10 else "FAIL"
        print(f"  {strat:20s}  base={base_lrvol:.2%}  vmp={vmp_rvol:.2%}  ratio={ratio:.3f}  {status}")

    # ── 6. 32-row paired table ───────────────────────────────────────────────
    rows = []
    for strat in all_strategies:
        base = wide16[strat].dropna()
        vmp  = vmp_series[f"VMP({strat})"]
        rows.append(_fmt_row(strat, performance_stats(base)))
        rows.append(_fmt_row(f"VMP({strat})", performance_stats(vmp)))
        rows.append(_sep())

    df_table = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 84)
    print("32-row table — 16 strategies × 2 (original + VMP), 2008-01 to 2026-04")
    print("VMP params: lookback=21d, lag=1d, clip=(0.25,1.5), target_vol=strategy LR vol")
    print("=" * 84)
    print(df_table.to_string())
    print()


if __name__ == "__main__":
    main()
