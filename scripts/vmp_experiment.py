"""
VMP experiment — apply Moreira-Muir 2017 volatility management to all 14
cached strategies using target_vol=None (vol-stabilized to each strategy's
own long-run realized vol).

Produces a 28-row paired table and saves VMP returns to:
    data/cache/portfolio_returns/14strategies_vmp_2008_2026.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns

PORTFOLIO_CACHE = "data/cache/portfolio_returns/14strategies_2008_2026.parquet"
VMP_OUT = "data/cache/portfolio_returns/14strategies_vmp_2008_2026.parquet"


def _fmt_row(name: str, s: dict) -> dict:
    return {
        "strategy": name,
        "ann_return": f"{s['annualized_return']:.2%}",
        "ann_vol": f"{s['annualized_volatility']:.2%}",
        "sharpe": f"{s['sharpe_ratio']:.3f}",
        "max_dd": f"{s['max_drawdown']:.2%}",
        "calmar": f"{s['calmar_ratio']:.3f}",
    }


def main() -> None:
    print("Loading cache...")
    wide = pd.read_parquet(PORTFOLIO_CACHE)
    strategies = wide.columns.tolist()

    # ── Build VMP variants ───────────────────────────────────────────────────
    vmp_series: dict[str, pd.Series] = {}
    for strat in strategies:
        base = wide[strat].dropna()
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(base)

    # ── Sanity check 1: realized vol of VMP ≈ base long-run vol (±10%) ──────
    print("\nSanity check 1 — VMP realized vol within ±10% of base long-run vol:")
    all_pass = True
    for strat in strategies:
        base = wide[strat].dropna()
        base_lrvol = base.std() * np.sqrt(252)
        vmp = vmp_series[f"VMP({strat})"]
        vmp_rvol = vmp.std() * np.sqrt(252)
        ratio = vmp_rvol / base_lrvol
        status = "PASS" if abs(ratio - 1.0) <= 0.10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {strat:30s}  base={base_lrvol:.2%}  vmp={vmp_rvol:.2%}  ratio={ratio:.3f}  {status}")
    print(f"  {'Overall':30s}  {'PASS' if all_pass else 'FAIL'}")

    # ── Sanity check 2: mean exposure ≈ 1.0 ─────────────────────────────────
    print("\nSanity check 2 — mean VMP exposure ≈ 1.0 (no systematic over/under):")
    for strat in strategies:
        base = wide[strat].dropna()
        base_lrvol = base.std() * np.sqrt(252)
        realized_vol = base.rolling(21).std() * np.sqrt(252)
        exposure = (base_lrvol / realized_vol).shift(1).clip(0.25, 1.5).fillna(1.0)
        mean_exp = exposure.mean()
        status = "ok" if abs(mean_exp - 1.0) < 0.15 else "CHECK"
        print(f"  {strat:30s}  mean_exposure={mean_exp:.4f}  {status}")

    # ── Build 28-row paired table ────────────────────────────────────────────
    rows = []
    for strat in strategies:
        base = wide[strat].dropna()
        vmp = vmp_series[f"VMP({strat})"]
        rows.append(_fmt_row(strat, performance_stats(base)))
        rows.append(_fmt_row(f"VMP({strat})", performance_stats(vmp)))
        rows.append({"strategy": "", "ann_return": "", "ann_vol": "", "sharpe": "", "max_dd": "", "calmar": ""})

    df_table = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 80)
    print("VMP experiment — 14 strategies × 2 (original + VMP), 2008-01 to 2026-05")
    print("VMP params: lookback=21d, lag=1d, clip=(0.25, 1.5), target_vol=strategy's own LR vol")
    print("=" * 80)
    print(df_table.to_string())
    print()

    # ── Save VMP returns ─────────────────────────────────────────────────────
    vmp_wide = pd.DataFrame(vmp_series)
    vmp_wide.to_parquet(VMP_OUT)
    print(f"Saved {vmp_wide.shape[1]} VMP series → {VMP_OUT}")


if __name__ == "__main__":
    main()
