"""
Part 5: SWITCH(v2a) OOS re-derivation.

Derives the v2a rule from training-only data (2003-2022) and evaluates it on
the test period (2023-2026). Compares against the original full-sample v2a rule.

Outputs:
  data/cache/portfolio_returns/switch_v2a_oos_29assets.parquet  (train + test series)
  Prints training-only regime-conditional Sharpe table and new rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aiam.data.split import TRAIN_END, TEST_START, split_train_test
from aiam.evaluation.performance import performance_stats
from aiam.evaluation.regime_conditional import regime_conditional_performance
from aiam.evaluation.switch_assembly import assemble_switch_returns

PORTFOLIO_CACHE = Path("data/cache/portfolio_returns/31strategies_29assets_2003_2026.parquet")
REGIME_CACHE = Path("data/cache/regime_signals_2003_2026.parquet")
OUT_PATH = Path("data/cache/portfolio_returns/switch_v2a_oos_29assets.parquet")

# Original v2a rule (derived in-sample on full 2008-2026 period with 30 assets)
ORIGINAL_V2A_RULE = {0: "MSR(ledoit_wolf)", 5: "MSR(sample)"}
ORIGINAL_V2A_DEFAULT = "MDP(ledoit_wolf)"

BASELINE_STRATEGIES = [
    "EW", "GMV(sample)", "GMV(ledoit_wolf)", "GMV(oas)",
    "MSR(sample)", "MSR(ledoit_wolf)",
    "MDP(sample)", "MDP(ledoit_wolf)",
    "RP(sample)", "RP(ledoit_wolf)",
    "HRP(sample)", "HRP(ledoit_wolf)",
]
EXCLUDE = {"GMV(sample)"}  # degenerate cash corner
MIN_TRAIN_DAYS = 252


def ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 1 else np.nan


def main() -> None:
    if not PORTFOLIO_CACHE.exists():
        print(f"ERROR: {PORTFOLIO_CACHE} not found — run build_all_strategies_29.py first")
        return

    print("Loading portfolio returns and regime signals...")
    wide = pd.read_parquet(PORTFOLIO_CACHE)
    regimes = pd.read_parquet(REGIME_CACHE)
    dominant_regime = regimes["dominant_regime"].dropna()

    # Restrict to the 12 baseline strategies for SWITCH analysis
    available = [s for s in BASELINE_STRATEGIES if s in wide.columns]
    port_train = wide[available].loc[wide.index <= TRAIN_END]

    print(f"Training period: {port_train.index.min().date()} → {port_train.index.max().date()}")
    print(f"Test period: {TEST_START.date()} → {wide.index.max().date()}")

    # Regime signals trimmed to training period
    regime_train = dominant_regime[dominant_regime.index <= TRAIN_END]

    # Compute regime-conditional Sharpe on TRAINING data only
    print("\nComputing regime-conditional Sharpe (training-only)...")
    returns_dict_train = {s: port_train[s].dropna() for s in available}
    tables = regime_conditional_performance(returns_dict_train, regime_train, min_days=MIN_TRAIN_DAYS)

    sharpe_train = tables["sharpe"]
    n_days_train = tables["n_days"]
    coverage = n_days_train.iloc[0].astype(int)

    print("\n── Training-only Regime-Conditional Sharpe ──")
    cov_row = pd.DataFrame([coverage], index=["n_days"])
    display = pd.concat([cov_row, sharpe_train.round(3)])
    print(display.to_string())

    # Derive new v2a rule from training data
    print("\n── Best Strategy per Regime (training-only, excl. GMV(sample)) ──")
    new_rule: dict[int, str] = {}
    for r in range(8):
        col = sharpe_train[r].drop(labels=list(EXCLUDE), errors="ignore")
        if col.notna().any():
            best = col.idxmax()
            new_rule[r] = best
        else:
            best = "N/A"
        days = int(coverage.get(r, 0))
        sparse = " *" if days < MIN_TRAIN_DAYS else ""
        print(f"  R{r}: {best:28s}  sharpe={col.max():.3f}  n_days={days}{sparse}")

    # Default = most common best strategy
    from collections import Counter
    default_counts = Counter(new_rule.values())
    new_default = default_counts.most_common(1)[0][0]
    print(f"\nNew v2a rule (training-only):")
    print(f"  Rule: {new_rule}")
    print(f"  Default: {new_default}")

    # Compare original v2a rule vs new training-only v2a rule
    print("\n── Original v2a vs New (training-only) v2a Rule ──")
    print(f"{'Regime':<8}  {'Original v2a':28}  {'New v2a (train-only)':28}")
    for r in range(8):
        orig = ORIGINAL_V2A_RULE.get(r, ORIGINAL_V2A_DEFAULT)
        new = new_rule.get(r, new_default)
        match = "=" if orig == new else "≠"
        print(f"  R{r}:   {orig:28s}  {new:28s}  {match}")

    # Build SWITCH series for both rules using FULL portfolio returns
    print("\nAssembling SWITCH return series...")
    orig_series = assemble_switch_returns(
        wide[available], dominant_regime, ORIGINAL_V2A_RULE, ORIGINAL_V2A_DEFAULT
    ).rename("SWITCH_v2a_original")

    new_series = assemble_switch_returns(
        wide[available], dominant_regime, new_rule, new_default
    ).rename("SWITCH_v2a_train_only")

    out = pd.DataFrame({
        "SWITCH_v2a_original": orig_series,
        "SWITCH_v2a_train_only": new_series,
    })
    out.to_parquet(OUT_PATH)
    print(f"Saved → {OUT_PATH}")

    # Performance comparison
    print("\n── Full-Sample Sharpe ──")
    for name, s in [("SWITCH_v2a_original", orig_series), ("SWITCH_v2a_train_only", new_series)]:
        stats = performance_stats(s.dropna())
        print(f"  {name:35s}  Sharpe={stats['sharpe_ratio']:.3f}  "
              f"ret={stats['annualized_return']:.2%}  vol={stats['annualized_volatility']:.2%}")

    print("\n── Training Period Sharpe (2003-2022) ──")
    for name, s in [("SWITCH_v2a_original", orig_series), ("SWITCH_v2a_train_only", new_series)]:
        train_s = s.loc[s.index <= TRAIN_END].dropna()
        print(f"  {name:35s}  Sharpe={ann_sharpe(train_s):.3f}  n={len(train_s)}")

    print("\n── Test Period Sharpe (2023-2026) ──")
    for name, s in [("SWITCH_v2a_original", orig_series), ("SWITCH_v2a_train_only", new_series)]:
        test_s = s.loc[s.index >= TEST_START].dropna()
        print(f"  {name:35s}  Sharpe={ann_sharpe(test_s):.3f}  n={len(test_s)}")

    # Top 5 strategies by test-period Sharpe
    print("\n── Top 10 Strategies by Test-Period Sharpe (2023-2026) ──")
    test_sharpes = {}
    for col in wide.columns:
        ts = wide[col].loc[wide.index >= TEST_START].dropna()
        if len(ts) > 20:
            test_sharpes[col] = ann_sharpe(ts)
    # Add SWITCH variants
    for name, s in [("SWITCH_v2a_original", orig_series), ("SWITCH_v2a_train_only", new_series)]:
        ts = s.loc[s.index >= TEST_START].dropna()
        if len(ts) > 20:
            test_sharpes[name] = ann_sharpe(ts)

    top10 = sorted(test_sharpes.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, sh in top10:
        print(f"  {name:35s}  {sh:.3f}")


if __name__ == "__main__":
    main()
