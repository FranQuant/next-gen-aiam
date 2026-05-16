"""Build long-short variants — 28 → 31 strategies.

Adds TSMOM-LS(12m), BL-Mom-LS(LW), FF3-Mom-LS.
Assumptions: zero borrow cost, unlimited short availability, no short rebate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov
from aiam.estimators.views import momentum_views
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.transaction_costs import compute_turnover
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.black_litterman import BlackLitterman
from aiam.strategy.factor_portfolio import FF3MomLongShort
from aiam.strategy.tsmom import TSMOM

PRICES_CACHE = "data/cache/prices_29.parquet"
S28_CACHE = "data/cache/portfolio_returns/28strategies_2008_2026.parquet"
S31_CACHE = "data/cache/portfolio_returns/31strategies_2008_2026.parquet"
S31_VMP  = "data/cache/portfolio_returns/31strategies_vmp_2008_2026.parquet"
START = "2008-01-01"
END   = "2026-04-30"

STRATEGIES = {
    "TSMOM-LS(12m)": TSMOM(signal_lookback=252, long_only=False),
    "BL-Mom-LS(LW)": BlackLitterman(
        view_generator=momentum_views,
        cov_estimator=ledoit_wolf_cov,
        long_only=False,
    ),
    "FF3-Mom-LS": FF3MomLongShort(),
}


def _row(name: str, s: dict, turnover: float) -> None:
    print(
        f"  {name:22s}  ret={s['annualized_return']:.2%}  vol={s['annualized_volatility']:.2%}"
        f"  sharpe={s['sharpe_ratio']:.3f}  dd={s['max_drawdown']:.2%}  to={turnover:.2f}%"
    )


def main() -> None:
    print("Loading panel data...")
    prices = pd.read_parquet(PRICES_CACHE)
    panel = Panel({"prices": prices})

    print("Loading 28-strategy base cache...")
    wide28 = pd.read_parquet(S28_CACHE)

    new_returns: dict[str, pd.Series] = {}
    for name, strategy in STRATEGIES.items():
        print(f"  Running {name}...", end="", flush=True)
        result = run_horse_race(
            panel, strategy, start=START, end=END,
            save_weights=True, strategy_name=name,
        )
        new_returns[name] = result["portfolio_returns"].dropna()
        st = result["stats"]
        print(f"  Sharpe={st['sharpe_ratio']:.3f}")

    wide31 = wide28.copy()
    for name, series in new_returns.items():
        wide31[name] = series
    wide31.to_parquet(S31_CACHE)
    print(f"\nSaved 31-strategy cache → {S31_CACHE}")

    vmp_series: dict[str, pd.Series] = {}
    for strat in wide31.columns:
        vmp_series[f"VMP({strat})"] = assemble_vmp_returns(wide31[strat].dropna())
    pd.DataFrame(vmp_series).to_parquet(S31_VMP)
    print(f"Saved VMP cache → {S31_VMP}")

    print("\n── Long-Short Summary ──")
    from pathlib import Path
    for name in STRATEGIES:
        s = new_returns[name]
        v = vmp_series[f"VMP({name})"]
        st = performance_stats(s)
        sv = performance_stats(v)
        wp = Path(f"data/cache/portfolio_weights/{name.replace('(','_').replace(')','').replace('/','_')}_2008_2026.parquet")
        to = compute_turnover(pd.read_parquet(wp)).dropna().mean() * 100 if wp.exists() else float("nan")
        net = s - compute_turnover(pd.read_parquet(wp)).shift(1).reindex(s.index, fill_value=0.0) * 10 / 10_000 if wp.exists() else s
        from aiam.evaluation.transaction_costs import apply_costs
        net = apply_costs(s, pd.read_parquet(wp)) if wp.exists() else s
        ns = net.mean() / net.std() * np.sqrt(252)
        print(
            f"  base  {name:22s}  sharpe={st['sharpe_ratio']:.3f}  ret={st['annualized_return']:.2%}"
            f"  vol={st['annualized_volatility']:.2%}  dd={st['max_drawdown']:.2%}"
            f"  calmar={st['calmar_ratio']:.3f}  to={to:.2f}%  hit={st['hit_ratio']:.3f}  net_s={ns:.3f}"
        )
        print(
            f"  VMP   {name:22s}  sharpe={sv['sharpe_ratio']:.3f}  ret={sv['annualized_return']:.2%}"
            f"  vol={sv['annualized_volatility']:.2%}  dd={sv['max_drawdown']:.2%}"
            f"  calmar={sv['calmar_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
