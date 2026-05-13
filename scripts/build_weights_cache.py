"""Populate data/cache/portfolio_weights/ for all 24 base strategies.

Skips any strategy whose weight file already exists and is newer than
data/cache/prices_30.parquet (mtime-check invalidation).

Run time: ~10-20 minutes for all 24 strategies on first execution.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.estimators.factor_signals import low_vol_signal, momentum_signal, quality_signal
from aiam.estimators.mean import sample_mean
from aiam.estimators.views import equilibrium_only, mean_reversion_views, momentum_views
from aiam.harness.horse_race import _load_or_run_weights, _weights_path
from aiam.strategy.black_litterman import BlackLitterman
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.factor_portfolio import FactorPortfolio, MultiFactorPortfolio
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.hierarchical_risk_parity import HierarchicalRiskParity
from aiam.strategy.max_sharpe import MaximumSharpe
from aiam.strategy.most_diversified import MostDiversified
from aiam.strategy.risk_parity import RiskParity
from aiam.strategy.switching import SwitchingStrategy
from aiam.strategy.tsmom import TSMOM

PRICES_CACHE = "data/cache/prices_30.parquet"
REGIME_CACHE = "data/cache/regime_signals.parquet"
START = "2008-01-01"
END = "2026-04-30"

_mom_strat  = FactorPortfolio(signal_fn=momentum_signal, lookback=756, weighting="inverse_vol")
_lv_strat   = FactorPortfolio(signal_fn=low_vol_signal,  lookback=756, weighting="inverse_vol")
_qual_strat = FactorPortfolio(signal_fn=quality_signal,  lookback=756, weighting="inverse_vol")

ALL_STRATEGIES: dict = {
    "EW":                EqualWeight(),
    "GMV(sample)":       GlobalMinVariance(sample_cov),
    "GMV(ledoit_wolf)":  GlobalMinVariance(ledoit_wolf_cov),
    "GMV(oas)":          GlobalMinVariance(oas_cov),
    "MSR(sample)":       MaximumSharpe(sample_cov, sample_mean),
    "MSR(ledoit_wolf)":  MaximumSharpe(ledoit_wolf_cov, sample_mean),
    "MDP(sample)":       MostDiversified(sample_cov),
    "MDP(ledoit_wolf)":  MostDiversified(ledoit_wolf_cov),
    "RP(sample)":        RiskParity(sample_cov),
    "RP(ledoit_wolf)":   RiskParity(ledoit_wolf_cov),
    "HRP(sample)":       HierarchicalRiskParity(sample_cov),
    "HRP(ledoit_wolf)":  HierarchicalRiskParity(ledoit_wolf_cov),
    "SWITCH(sample)": SwitchingStrategy(
        switching_rule={
            0: EqualWeight(),
            1: MostDiversified(sample_cov), 2: MostDiversified(sample_cov),
            3: MostDiversified(sample_cov), 4: MostDiversified(sample_cov),
            5: MaximumSharpe(sample_cov, sample_mean),
            6: MostDiversified(sample_cov), 7: MostDiversified(sample_cov),
        },
        default_strategy=MostDiversified(sample_cov),
    ),
    "SWITCH(ledoit_wolf)": SwitchingStrategy(
        switching_rule={
            0: EqualWeight(),
            1: MostDiversified(ledoit_wolf_cov), 2: MostDiversified(ledoit_wolf_cov),
            3: MostDiversified(ledoit_wolf_cov), 4: MostDiversified(ledoit_wolf_cov),
            5: MaximumSharpe(ledoit_wolf_cov, sample_mean),
            6: MostDiversified(ledoit_wolf_cov), 7: MostDiversified(ledoit_wolf_cov),
        },
        default_strategy=MostDiversified(ledoit_wolf_cov),
    ),
    "TSMOM(12m)":  TSMOM(signal_lookback=252),
    "TSMOM(6m)":   TSMOM(signal_lookback=126),
    "BL-Eq(sample)": BlackLitterman(view_generator=equilibrium_only, cov_estimator=sample_cov),
    "BL-Eq(LW)":     BlackLitterman(view_generator=equilibrium_only, cov_estimator=ledoit_wolf_cov),
    "BL-Mom(LW)":    BlackLitterman(view_generator=momentum_views,   cov_estimator=ledoit_wolf_cov),
    "BL-Rev(LW)":    BlackLitterman(view_generator=mean_reversion_views, cov_estimator=ledoit_wolf_cov),
    "FF3-Mom":     _mom_strat,
    "FF3-LowVol":  _lv_strat,
    "FF3-Quality": _qual_strat,
    "FF3-Multi":   MultiFactorPortfolio([_mom_strat, _lv_strat, _qual_strat]),
}


def main() -> None:
    prices_mtime = Path(PRICES_CACHE).stat().st_mtime

    print("Loading panel data...")
    prices  = pd.read_parquet(PRICES_CACHE)
    regimes = pd.read_parquet(REGIME_CACHE)
    panel   = Panel({"prices": prices, "regimes": regimes})

    Path("data/cache/portfolio_weights").mkdir(parents=True, exist_ok=True)

    total = len(ALL_STRATEGIES)
    for idx, (name, strategy) in enumerate(ALL_STRATEGIES.items(), 1):
        path = _weights_path(name)
        if path.exists() and path.stat().st_mtime > prices_mtime:
            print(f"[{idx:2d}/{total}] {name}: cached ✓ ({path.name})")
            continue

        t0 = time.time()
        print(f"[{idx:2d}/{total}] {name}: running ...", end="", flush=True)
        _load_or_run_weights(panel, strategy, name, START, END)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s → {path.name}")

    print(f"\nWeights cache complete. {total} files in data/cache/portfolio_weights/")


if __name__ == "__main__":
    main()
