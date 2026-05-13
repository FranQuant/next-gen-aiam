"""
SWITCH v2 experiment — assembles new switching rules from the cached portfolio_returns
without re-running any optimisation.

Defines:
  SWITCH_LW_v1          — mirrors the original SwitchingStrategy(ledoit_wolf) baseline
  SWITCH_v2a_conservative — R0→MSR(LW), R5→MSR(sample), others→MDP(LW)
  SWITCH_v2b_aggressive   — v2a + R7→RP(sample)

Sanity-checks that SWITCH_LW_v1 assembled from cache reproduces the original
horse-race Sharpe within 1e-6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from aiam.evaluation.performance import performance_stats
from aiam.evaluation.switch_assembly import assemble_switch_returns

PORTFOLIO_CACHE = "data/cache/portfolio_returns/14strategies_2008_2026.parquet"
REGIME_CACHE = "data/cache/regime_signals.parquet"

# ── Switching rules (strategy names must be columns in the cache) ────────────

# Original paam_lab 19d rule reproduced from SwitchingStrategy(ledoit_wolf)
SWITCH_LW_V1_RULE: dict[int, str] = {
    0: "EW",
    1: "MDP(ledoit_wolf)",
    2: "MDP(ledoit_wolf)",
    3: "MDP(ledoit_wolf)",
    4: "MDP(ledoit_wolf)",
    5: "MSR(ledoit_wolf)",
    6: "MDP(ledoit_wolf)",
    7: "MDP(ledoit_wolf)",
}
SWITCH_LW_V1_DEFAULT = "MDP(ledoit_wolf)"

# V2a: swap R0→MSR(LW) and R5→MSR(sample) (analysis-recommended non-SWITCH bests)
SWITCH_V2A_RULE: dict[int, str] = {
    0: "MSR(ledoit_wolf)",
    1: "MDP(ledoit_wolf)",
    2: "MDP(ledoit_wolf)",
    3: "MDP(ledoit_wolf)",
    4: "MDP(ledoit_wolf)",
    5: "MSR(sample)",
    6: "MDP(ledoit_wolf)",
    7: "MDP(ledoit_wolf)",
}
SWITCH_V2A_DEFAULT = "MDP(ledoit_wolf)"

# V2b: V2a + R7→RP(sample) (best non-SWITCH for regime 7)
SWITCH_V2B_RULE: dict[int, str] = {**SWITCH_V2A_RULE, 7: "RP(sample)"}
SWITCH_V2B_DEFAULT = "MDP(ledoit_wolf)"


def _fmt_row(name: str, s: dict) -> dict:
    return {
        "strategy": name,
        "ann_return": f"{s['annualized_return']:.2%}",
        "ann_vol": f"{s['annualized_volatility']:.2%}",
        "sharpe": f"{s['sharpe_ratio']:.3f}",
        "max_drawdown": f"{s['max_drawdown']:.2%}",
    }


def main() -> None:
    print("Loading cache...")
    wide = pd.read_parquet(PORTFOLIO_CACHE)
    regimes_df = pd.read_parquet(REGIME_CACHE)
    dominant_regime = regimes_df["dominant_regime"].dropna()

    # ── Assemble SWITCH variants ─────────────────────────────────────────────
    variants: dict[str, tuple[dict, str]] = {
        "SWITCH_LW_v1":            (SWITCH_LW_V1_RULE, SWITCH_LW_V1_DEFAULT),
        "SWITCH_v2a_conservative": (SWITCH_V2A_RULE,   SWITCH_V2A_DEFAULT),
        "SWITCH_v2b_aggressive":   (SWITCH_V2B_RULE,   SWITCH_V2B_DEFAULT),
    }

    assembled: dict[str, pd.Series] = {}
    for label, (rule, default) in variants.items():
        assembled[label] = assemble_switch_returns(wide, dominant_regime, rule, default).dropna()

    # ── Sanity check: SWITCH_LW_v1 ≈ cached SWITCH(ledoit_wolf) ────────────
    original = wide["SWITCH(ledoit_wolf)"].dropna()
    v1 = assembled["SWITCH_LW_v1"]
    # Align on common dates before comparing
    common = original.index.intersection(v1.index)
    diff = (original.reindex(common) - v1.reindex(common)).abs().max()

    orig_sharpe = performance_stats(original)["sharpe_ratio"]
    v1_sharpe   = performance_stats(v1)["sharpe_ratio"]
    sharpe_diff = abs(orig_sharpe - v1_sharpe)

    print(f"\nSanity check — SWITCH_LW_v1 vs original SWITCH(ledoit_wolf):")
    print(f"  Max daily return diff : {diff:.2e}")
    print(f"  Sharpe diff           : {sharpe_diff:.2e}")
    if sharpe_diff > 1e-6:
        print("  FAIL: Sharpe diverges beyond 1e-6 — assembly rule may not match original")
        sys.exit(1)
    else:
        print("  PASS: within 1e-6")

    # ── Build comparison table ────────────────────────────────────────────────
    baselines = ["EW", "MDP(ledoit_wolf)", "MSR(ledoit_wolf)", "SWITCH(ledoit_wolf)"]

    rows = []
    for b in baselines:
        rows.append(_fmt_row(b, performance_stats(wide[b].dropna())))
    rows.append({"strategy": "─" * 24, "ann_return": "", "ann_vol": "", "sharpe": "", "max_drawdown": ""})
    for label, series in assembled.items():
        rows.append(_fmt_row(label, performance_stats(series)))

    df = pd.DataFrame(rows).set_index("strategy")

    print("\n" + "=" * 72)
    print("SWITCH v2 experiment — 2008-01-01 to 2026-04-30")
    print("=" * 72)
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
