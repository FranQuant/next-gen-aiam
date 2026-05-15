"""Build 8-strategy no-BTC comparison (29-asset panel, BTC-USD.CC dropped).

Output: data/cache/portfolio_returns/8strategies_no_btc_2008_2026.parquet
Columns: EW, GMV(ledoit_wolf), MSR(sample), MSR(ledoit_wolf), MDP(ledoit_wolf),
         HRP(ledoit_wolf), SWITCH(ledoit_wolf), SWITCH(v2a), VMP(MSR(ledoit_wolf))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, sample_cov
from aiam.estimators.mean import sample_mean
from aiam.evaluation.switch_assembly import assemble_switch_returns
from aiam.evaluation.vmp_assembly import assemble_vmp_returns
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.hierarchical_risk_parity import HierarchicalRiskParity
from aiam.strategy.max_sharpe import MaximumSharpe
from aiam.strategy.most_diversified import MostDiversified
from aiam.strategy.switching import SwitchingStrategy

PRICES_CACHE = "data/cache/prices_30.parquet"
REGIME_CACHE = "data/cache/regime_signals.parquet"
OUT_PATH = "data/cache/portfolio_returns/8strategies_no_btc_2008_2026.parquet"
START, END = "2008-01-01", "2026-04-30"
TRADING_DAYS = 252

SWITCH_V1_RULE = {
    0: EqualWeight(),
    1: MostDiversified(ledoit_wolf_cov), 2: MostDiversified(ledoit_wolf_cov),
    3: MostDiversified(ledoit_wolf_cov), 4: MostDiversified(ledoit_wolf_cov),
    5: MaximumSharpe(ledoit_wolf_cov, sample_mean),
    6: MostDiversified(ledoit_wolf_cov), 7: MostDiversified(ledoit_wolf_cov),
}
SWITCH_V2A_RULE = {0: "MSR(ledoit_wolf)", 5: "MSR(sample)"}

STRATEGIES = {
    "EW":                EqualWeight(),
    "GMV(ledoit_wolf)":  GlobalMinVariance(ledoit_wolf_cov),
    "MSR(sample)":       MaximumSharpe(sample_cov, sample_mean),
    "MSR(ledoit_wolf)":  MaximumSharpe(ledoit_wolf_cov, sample_mean),
    "MDP(ledoit_wolf)":  MostDiversified(ledoit_wolf_cov),
    "HRP(ledoit_wolf)":  HierarchicalRiskParity(ledoit_wolf_cov),
    "SWITCH(ledoit_wolf)": SwitchingStrategy(
        SWITCH_V1_RULE, MostDiversified(ledoit_wolf_cov)
    ),
}


def ann_sharpe(s: pd.Series) -> float:
    return s.mean() / s.std() * np.sqrt(TRADING_DAYS)


def main() -> None:
    prices = pd.read_parquet(PRICES_CACHE).drop(columns=["BTC-USD.CC"])
    regimes = pd.read_parquet(REGIME_CACHE)
    panel = Panel({"prices": prices, "regimes": regimes})

    print(f"Panel: {prices.shape[1]} assets, {prices.shape[0]} days (BTC dropped)")

    results: dict[str, pd.Series] = {}
    for name, strat in STRATEGIES.items():
        print(f"  {name}...", end="", flush=True)
        res = run_horse_race(panel, strat, start=START, end=END)
        s = res["portfolio_returns"].dropna()
        results[name] = s
        print(f"  Sharpe={ann_sharpe(s):.3f}")

    regime_sig = regimes["dominant_regime"].dropna()
    base_df = pd.DataFrame({k: results[k] for k in ["MSR(ledoit_wolf)", "MSR(sample)", "MDP(ledoit_wolf)"]})
    switch_v2a = assemble_switch_returns(
        base_df, regime_sig, SWITCH_V2A_RULE, "MDP(ledoit_wolf)"
    ).rename("SWITCH(v2a)")
    results["SWITCH(v2a)"] = switch_v2a
    print(f"  SWITCH(v2a)  Sharpe={ann_sharpe(switch_v2a):.3f}")

    vmp_msr = assemble_vmp_returns(results["MSR(ledoit_wolf)"]).rename("VMP(MSR(ledoit_wolf))")
    results["VMP(MSR(ledoit_wolf))"] = vmp_msr
    print(f"  VMP(MSR(LW)) Sharpe={ann_sharpe(vmp_msr):.3f}")

    SAVE_COLS = [
        "EW", "GMV(ledoit_wolf)", "MSR(ledoit_wolf)", "MDP(ledoit_wolf)",
        "HRP(ledoit_wolf)", "SWITCH(ledoit_wolf)", "SWITCH(v2a)", "VMP(MSR(ledoit_wolf))",
    ]
    out = pd.DataFrame({c: results[c] for c in SAVE_COLS})
    out.to_parquet(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}  shape={out.shape}")

    print("\n── Sharpe summary ──")
    for c in SAVE_COLS:
        print(f"  {c:35s}  {ann_sharpe(out[c]):.3f}")


if __name__ == "__main__":
    main()
