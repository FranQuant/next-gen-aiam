"""
Master harness — run all 31 strategies on the 29-asset, 2003-2026 universe.

Produces:
  data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet   (31 base)
  data/cache/portfolio_returns/31strategies_vmp_29assets_2003_2026.parquet (31 VMP)
  data/cache/portfolio_weights/<strategy>_29assets_2003_2026.parquet     (per strategy)

First valid portfolio date ≈ 2004-01-01 (252-day lookback for MV families).
Strategies hold zero weight on assets not yet live (N(t) variable universe).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.estimators.factor_signals import low_vol_signal, momentum_signal, quality_signal
from aiam.estimators.mean import sample_mean
from aiam.estimators.views import equilibrium_only, mean_reversion_views, momentum_views
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race, _weights_path
from aiam.strategy.black_litterman import BlackLitterman
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.factor_portfolio import FactorPortfolio, FF3MomLongShort, MultiFactorPortfolio
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.hierarchical_risk_parity import HierarchicalRiskParity
from aiam.strategy.max_sharpe import MaximumSharpe
from aiam.strategy.max_sharpe_constrained import MaximumSharpeConstrained
from aiam.strategy.most_diversified import MostDiversified
from aiam.strategy.mvo_constrained import MVOConstrained
from aiam.strategy.risk_parity import RiskParity
from aiam.strategy.switching import SwitchingStrategy
from aiam.strategy.tsmom import TSMOM

PRICES_CACHE = Path("data/cache/prices_29.parquet")
REGIME_CACHE = Path("data/cache/regime_signals_2003_2026.parquet")
OUT_BASE = Path("data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet")
OUT_VMP = Path("data/cache/portfolio_returns/31strategies_vmp_29assets_2003_2026.parquet")
WEIGHTS_SUFFIX = "29assets_2003_2026"
START = "2003-01-02"  # first NYSE trading day; bdate_range includes Jan 1 (holiday) otherwise
END = "2026-04-30"
BOUNDS = (0.05, 0.40)

_mom_strat  = FactorPortfolio(signal_fn=momentum_signal, lookback=756, weighting="inverse_vol")
_lv_strat   = FactorPortfolio(signal_fn=low_vol_signal,  lookback=756, weighting="inverse_vol")
_qual_strat = FactorPortfolio(signal_fn=quality_signal,  lookback=756, weighting="inverse_vol")

_switch_sample_rule = {
    0: EqualWeight(),
    1: MostDiversified(sample_cov), 2: MostDiversified(sample_cov),
    3: MostDiversified(sample_cov), 4: MostDiversified(sample_cov),
    5: MaximumSharpe(sample_cov, sample_mean),
    6: MostDiversified(sample_cov), 7: MostDiversified(sample_cov),
}
_switch_lw_rule = {
    0: EqualWeight(),
    1: MostDiversified(ledoit_wolf_cov), 2: MostDiversified(ledoit_wolf_cov),
    3: MostDiversified(ledoit_wolf_cov), 4: MostDiversified(ledoit_wolf_cov),
    5: MaximumSharpe(ledoit_wolf_cov, sample_mean),
    6: MostDiversified(ledoit_wolf_cov), 7: MostDiversified(ledoit_wolf_cov),
}

ALL_STRATEGIES: dict = {
    # ── 14 core ──
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
    "SWITCH(sample)":    SwitchingStrategy(_switch_sample_rule, MostDiversified(sample_cov)),
    "SWITCH(ledoit_wolf)": SwitchingStrategy(_switch_lw_rule, MostDiversified(ledoit_wolf_cov)),
    # ── TSMOM ──
    "TSMOM(12m)":        TSMOM(signal_lookback=252),
    "TSMOM(6m)":         TSMOM(signal_lookback=126),
    # ── Black-Litterman ──
    "BL-Eq(sample)":     BlackLitterman(view_generator=equilibrium_only, cov_estimator=sample_cov),
    "BL-Eq(LW)":         BlackLitterman(view_generator=equilibrium_only, cov_estimator=ledoit_wolf_cov),
    "BL-Mom(LW)":        BlackLitterman(view_generator=momentum_views,   cov_estimator=ledoit_wolf_cov),
    "BL-Rev(LW)":        BlackLitterman(view_generator=mean_reversion_views, cov_estimator=ledoit_wolf_cov),
    # ── FF3 factor ──
    "FF3-Mom":           _mom_strat,
    "FF3-LowVol":        _lv_strat,
    "FF3-Quality":       _qual_strat,
    "FF3-Multi":         MultiFactorPortfolio([_mom_strat, _lv_strat, _qual_strat]),
    # ── Constrained MV ──
    "MSR_C(ledoit_wolf)": MaximumSharpeConstrained(ledoit_wolf_cov, sample_mean, bounds=BOUNDS),
    "MSR_C(sample)":      MaximumSharpeConstrained(sample_cov, sample_mean, bounds=BOUNDS),
    "MVO_C(ledoit_wolf)": MVOConstrained(ledoit_wolf_cov, bounds=BOUNDS),
    "MVO_C(sample)":      MVOConstrained(sample_cov, bounds=BOUNDS),
    # ── Long-short ──
    "TSMOM-LS(12m)":     TSMOM(signal_lookback=252, long_only=False),
    "BL-Mom-LS(LW)":     BlackLitterman(view_generator=momentum_views, cov_estimator=ledoit_wolf_cov, long_only=False),
    "FF3-Mom-LS":        FF3MomLongShort(),
}


def _fmt(name: str, s: dict) -> str:
    return (f"{name:28s}  ret={s['annualized_return']:.2%}  vol={s['annualized_volatility']:.2%}"
            f"  sharpe={s['sharpe_ratio']:.3f}  dd={s['max_drawdown']:.2%}")


def main() -> None:
    print("Loading panel data (29 assets, 2003-2026)...")
    prices  = pd.read_parquet(PRICES_CACHE)
    regimes = pd.read_parquet(REGIME_CACHE)
    panel   = Panel({"prices": prices, "regimes": regimes})
    print(f"  prices shape: {prices.shape}")

    Path("data/cache/portfolio_returns").mkdir(parents=True, exist_ok=True)
    Path("data/cache/portfolio_weights").mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.Series] = {}
    total = len(ALL_STRATEGIES)

    for idx, (name, strategy) in enumerate(ALL_STRATEGIES.items(), 1):
        t0 = time.time()
        print(f"[{idx:2d}/{total}] {name}: running...", end="", flush=True)
        result = run_horse_race(
            panel, strategy, start=START, end=END,
            save_weights=True, strategy_name=name,
            weights_suffix=WEIGHTS_SUFFIX,
        )
        elapsed = time.time() - t0
        s = result["portfolio_returns"].dropna()
        results[name] = s
        print(f"  {elapsed:.1f}s  Sharpe={result['stats']['sharpe_ratio']:.3f}")

    print("\nSaving base returns...")
    wide = pd.DataFrame(results)
    wide.to_parquet(OUT_BASE)
    print(f"Saved → {OUT_BASE}  shape={wide.shape}")

    print("Building VMP series...")
    vmp_series: dict[str, pd.Series] = {}
    for name in wide.columns:
        vmp_series[f"VMP({name})"] = assemble_vmp_returns(wide[name].dropna())
    vmp_df = pd.DataFrame(vmp_series)
    vmp_df.to_parquet(OUT_VMP)
    print(f"Saved → {OUT_VMP}  shape={vmp_df.shape}")

    print("\n── Full-sample Sharpe table (29 assets, 2003-2026) ──")
    print(f"{'Strategy':<28}  {'Ann Ret':>8}  {'Ann Vol':>8}  {'Sharpe':>8}  {'Max DD':>8}")
    print("-" * 72)
    for name in wide.columns:
        s = performance_stats(wide[name].dropna())
        vmp_name = f"VMP({name})"
        sv = performance_stats(vmp_df[vmp_name].dropna()) if vmp_name in vmp_df else None
        print(_fmt(name, s))
        if sv:
            print(_fmt(vmp_name, sv))

    print(f"\nFirst valid return date: {wide.dropna(how='all').index.min().date()}")
    print(f"MV lookback note: strategies with 252-day lookback start ≈ 2004-01-01")


if __name__ == "__main__":
    main()
