"""Build constrained MSR and MVO strategies — 24 → 28 strategies.

Adds MSR_C(ledoit_wolf), MSR_C(sample), MVO_C(ledoit_wolf), MVO_C(sample)
to the 24-strategy cache.  Saves weights and updates portfolio_returns caches.
Following JPM (2022) §3 practice: per-asset bounds [5%, 40%].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, sample_cov
from aiam.estimators.mean import sample_mean
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.max_sharpe_constrained import MaximumSharpeConstrained
from aiam.strategy.mvo_constrained import MVOConstrained

PRICES_CACHE = "data/cache/prices_29.parquet"
S24_CACHE = "data/cache/portfolio_returns/24strategies_2008_2026.parquet"
S28_CACHE = "data/cache/portfolio_returns/28strategies_2008_2026.parquet"
S28_VMP  = "data/cache/portfolio_returns/28strategies_vmp_2008_2026.parquet"
START = "2008-01-01"
END = "2026-04-30"
BOUNDS = (0.05, 0.40)

STRATEGIES = {
    "MSR_C(ledoit_wolf)": MaximumSharpeConstrained(
        cov_estimator=ledoit_wolf_cov,
        mean_estimator=sample_mean,
        bounds=BOUNDS,
    ),
    "MSR_C(sample)": MaximumSharpeConstrained(
        cov_estimator=sample_cov,
        mean_estimator=sample_mean,
        bounds=BOUNDS,
    ),
    "MVO_C(ledoit_wolf)": MVOConstrained(
        cov_estimator=ledoit_wolf_cov,
        bounds=BOUNDS,
    ),
    "MVO_C(sample)": MVOConstrained(
        cov_estimator=sample_cov,
        bounds=BOUNDS,
    ),
}


def main() -> None:
    print("Loading panel data...")
    prices = pd.read_parquet(PRICES_CACHE)
    panel = Panel({"prices": prices})

    print("Loading 24-strategy base cache...")
    wide24 = pd.read_parquet(S24_CACHE)

    new_returns: dict[str, pd.Series] = {}
    for name, strategy in STRATEGIES.items():
        print(f"  Running {name}...", end="", flush=True)
        result = run_horse_race(
            panel, strategy, start=START, end=END,
            save_weights=True, strategy_name=name,
        )
        sharpe = result["stats"]["sharpe_ratio"]
        new_returns[name] = result["portfolio_returns"].dropna()
        print(f"  Sharpe={sharpe:.3f}")

    wide28 = wide24.copy()
    for name, series in new_returns.items():
        wide28[name] = series
    wide28.to_parquet(S28_CACHE)
    print(f"\nSaved 28-strategy cache → {S28_CACHE}")

    vmp_series: dict[str, pd.Series] = {}
    for strat in wide28.columns:
        base_s = wide28[strat].dropna()
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(base_s)
    pd.DataFrame(vmp_series).to_parquet(S28_VMP)
    print(f"Saved VMP cache → {S28_VMP}")

    print("\n── Summary table ──")
    for name in STRATEGIES:
        s = new_returns[name]
        v = vmp_series[f"VMP({name})"]
        st = performance_stats(s)
        sv = performance_stats(v)
        print(
            f"  {name:24s}  base Sharpe={st['sharpe_ratio']:.3f}"
            f"  vmp Sharpe={sv['sharpe_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
