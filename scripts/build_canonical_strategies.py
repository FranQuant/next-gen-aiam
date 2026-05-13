"""
Canonical strategy builder — incremental session runner.

Session 4 (BL): 16 → 20 strategies (Black-Litterman variants).
Session 5 (FF3): 20 → 24 strategies (cross-sectional factor portfolios).

Saves per session:
    data/cache/portfolio_returns/20strategies_2008_2026.parquet
    data/cache/portfolio_returns/20strategies_vmp_2008_2026.parquet
    data/cache/portfolio_returns/24strategies_2008_2026.parquet
    data/cache/portfolio_returns/24strategies_vmp_2008_2026.parquet

Run without arguments to execute all sessions sequentially (skips a session if
the output cache already exists). Pass --session=4 or --session=5 to run only
that session.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, sample_cov
from aiam.estimators.factor_signals import low_vol_signal, momentum_signal, quality_signal
from aiam.estimators.views import equilibrium_only, mean_reversion_views, momentum_views
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.black_litterman import BlackLitterman
from aiam.strategy.factor_portfolio import FactorPortfolio, MultiFactorPortfolio

PRICES_CACHE = "data/cache/prices_30.parquet"
START = "2008-01-01"
END = "2026-04-30"

# ── Session caches ───────────────────────────────────────────────────────────
S4_IN = "data/cache/portfolio_returns/16strategies_2008_2026.parquet"
S4_OUT = "data/cache/portfolio_returns/20strategies_2008_2026.parquet"
S4_VMP = "data/cache/portfolio_returns/20strategies_vmp_2008_2026.parquet"

S5_IN = S4_OUT
S5_OUT = "data/cache/portfolio_returns/24strategies_2008_2026.parquet"
S5_VMP = "data/cache/portfolio_returns/24strategies_vmp_2008_2026.parquet"

# ── Strategy definitions ─────────────────────────────────────────────────────
_BL_VARIANTS: dict[str, BlackLitterman] = {
    "BL-Eq(sample)": BlackLitterman(view_generator=equilibrium_only, cov_estimator=sample_cov),
    "BL-Eq(LW)":     BlackLitterman(view_generator=equilibrium_only, cov_estimator=ledoit_wolf_cov),
    "BL-Mom(LW)":    BlackLitterman(view_generator=momentum_views,   cov_estimator=ledoit_wolf_cov),
    "BL-Rev(LW)":    BlackLitterman(view_generator=mean_reversion_views, cov_estimator=ledoit_wolf_cov),
}

_MOM_STRAT  = FactorPortfolio(signal_fn=momentum_signal, lookback=756, weighting="inverse_vol")
_LV_STRAT   = FactorPortfolio(signal_fn=low_vol_signal,  lookback=756, weighting="inverse_vol")
_QUAL_STRAT = FactorPortfolio(signal_fn=quality_signal,  lookback=756, weighting="inverse_vol")

_FF3_VARIANTS: dict = {
    "FF3-Mom":     _MOM_STRAT,
    "FF3-LowVol":  _LV_STRAT,
    "FF3-Quality": _QUAL_STRAT,
    "FF3-Multi":   MultiFactorPortfolio([_MOM_STRAT, _LV_STRAT, _QUAL_STRAT]),
}


def _fmt_row(name: str, s: dict) -> dict:
    return {
        "strategy": name,
        "ann_return": f"{s['annualized_return']:.2%}",
        "ann_vol":    f"{s['annualized_volatility']:.2%}",
        "sharpe":     f"{s['sharpe_ratio']:.3f}",
        "max_dd":     f"{s['max_drawdown']:.2%}",
        "calmar":     f"{s['calmar_ratio']:.3f}",
    }


def _sep() -> dict:
    return {k: "" for k in ("strategy", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar")}


def _run_session(
    label: str,
    base_cache: str,
    new_variants: dict,
    out_cache: str,
    vmp_cache: str,
    panel: Panel,
    table_header: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"Session {label}")
    print(f"{'='*60}")
    print(f"Loading base cache: {base_cache}")
    base_wide = pd.read_parquet(base_cache)

    new_returns: dict[str, pd.Series] = {}
    for name, strategy in new_variants.items():
        print(f"  Running {name}...", end="", flush=True)
        result = run_horse_race(panel, strategy, start=START, end=END)
        new_returns[name] = result["portfolio_returns"].dropna()
        print(f"  Sharpe={result['stats']['sharpe_ratio']:.3f}")

    wide = base_wide.copy()
    for name, series in new_returns.items():
        wide[name] = series
    wide.to_parquet(out_cache)
    print(f"Saved {wide.shape[1]}-strategy cache → {out_cache}")

    all_strategies = wide.columns.tolist()
    vmp_series: dict[str, pd.Series] = {}
    for strat in all_strategies:
        base = wide[strat].dropna()
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(base)
    pd.DataFrame(vmp_series).to_parquet(vmp_cache)
    print(f"Saved VMP cache → {vmp_cache}")

    print(f"\nSanity checks — new variants VMP vol ratio:")
    for strat in new_variants:
        base = wide[strat].dropna()
        base_lrvol = base.std() * np.sqrt(252)
        vmp = vmp_series[f"VMP({strat})"]
        vmp_rvol = vmp.std() * np.sqrt(252)
        ratio = vmp_rvol / base_lrvol
        status = "PASS" if abs(ratio - 1.0) <= 0.10 else "FAIL"
        print(f"  {strat:22s}  base={base_lrvol:.2%}  vmp={vmp_rvol:.2%}  ratio={ratio:.3f}  {status}")

    rows = []
    for strat in all_strategies:
        base = wide[strat].dropna()
        vmp = vmp_series[f"VMP({strat})"]
        rows.append(_fmt_row(strat, performance_stats(base)))
        rows.append(_fmt_row(f"VMP({strat})", performance_stats(vmp)))
        rows.append(_sep())

    df_table = pd.DataFrame(rows).set_index("strategy")
    print(f"\n{'='*92}")
    print(table_header)
    print(f"VMP params: lookback=21d, lag=1d, clip=(0.25,1.5), target_vol=strategy LR vol")
    print(f"{'='*92}")
    print(df_table.to_string())
    print()


def main(sessions: list[int] | None = None) -> None:
    if sessions is None:
        sessions = [4, 5]

    print("Loading panel data...")
    prices = pd.read_parquet(PRICES_CACHE)
    panel = Panel({"prices": prices})

    if 4 in sessions:
        _run_session(
            label="4 — Black-Litterman (16 → 20 strategies)",
            base_cache=S4_IN,
            new_variants=_BL_VARIANTS,
            out_cache=S4_OUT,
            vmp_cache=S4_VMP,
            panel=panel,
            table_header=(
                "40-row table — 20 strategies × 2 (original + VMP), 2008-01 to 2026-04\n"
                "BL params: tau=0.05, delta=2.5, lookback=252, prior=equal-weight"
            ),
        )

    if 5 in sessions:
        _run_session(
            label="5 — Cross-sectional factor portfolios (20 → 24 strategies)",
            base_cache=S5_IN,
            new_variants=_FF3_VARIANTS,
            out_cache=S5_OUT,
            vmp_cache=S5_VMP,
            panel=panel,
            table_header=(
                "48-row table — 24 strategies × 2 (original + VMP), 2008-01 to 2026-04\n"
                "FF3 params: top_fraction=1/3, weighting=inverse_vol, lookback=756"
            ),
        )


if __name__ == "__main__":
    session_args = [int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--session=")]
    main(sessions=session_args if session_args else None)
