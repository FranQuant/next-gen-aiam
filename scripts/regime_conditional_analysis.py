"""
Regime-conditional analysis — diagnostic script (paam_lab 19c port).

Prints three tables:
  1. Sharpe by strategy × regime (with coverage row)
  2. Best strategy per regime
  3. Switching-rule comparison vs paam_lab 19d rule

Does NOT modify SwitchingStrategy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.estimators.mean import sample_mean
from aiam.evaluation.regime_conditional import regime_conditional_performance
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.hierarchical_risk_parity import HierarchicalRiskParity
from aiam.strategy.max_sharpe import MaximumSharpe
from aiam.strategy.most_diversified import MostDiversified
from aiam.strategy.risk_parity import RiskParity
from aiam.strategy.switching import SwitchingStrategy

CACHE = "data/cache/prices_30.parquet"
REGIME_CACHE = "data/cache/regime_signals.parquet"
PORTFOLIO_CACHE = "data/cache/portfolio_returns/14strategies_2008_2026.parquet"
START = "2008-01-01"
END = "2026-04-30"
MIN_DAYS = 252

# Excluded from per-regime recommendation: degenerate cash corner
EXCLUDE_FROM_REC = {"GMV(sample)"}

# paam_lab 19d rule expressed as strategy families: R0→EW, R5→MSR, others→MDP
CURRENT_RULE: dict[int, str] = {
    0: "EW", 1: "MDP", 2: "MDP", 3: "MDP",
    4: "MDP", 5: "MSR", 6: "MDP", 7: "MDP",
}


def build_strategies() -> dict[str, object]:
    strategies: dict[str, object] = {
        "EW": EqualWeight(),
        "GMV(sample)": GlobalMinVariance(sample_cov),
        "GMV(ledoit_wolf)": GlobalMinVariance(ledoit_wolf_cov),
        "GMV(oas)": GlobalMinVariance(oas_cov),
        "MSR(sample)": MaximumSharpe(sample_cov, sample_mean),
        "MSR(ledoit_wolf)": MaximumSharpe(ledoit_wolf_cov, sample_mean),
        "MDP(sample)": MostDiversified(sample_cov),
        "MDP(ledoit_wolf)": MostDiversified(ledoit_wolf_cov),
        "RP(sample)": RiskParity(sample_cov),
        "RP(ledoit_wolf)": RiskParity(ledoit_wolf_cov),
        "HRP(sample)": HierarchicalRiskParity(sample_cov),
        "HRP(ledoit_wolf)": HierarchicalRiskParity(ledoit_wolf_cov),
    }
    for cov_label, cov_est in [("sample", sample_cov), ("ledoit_wolf", ledoit_wolf_cov)]:
        ew = EqualWeight()
        mdp = MostDiversified(cov_estimator=cov_est)
        msr = MaximumSharpe(cov_estimator=cov_est, mean_estimator=sample_mean)
        strategies[f"SWITCH({cov_label})"] = SwitchingStrategy(
            switching_rule={0: ew, 1: mdp, 2: mdp, 3: mdp, 4: mdp, 6: mdp, 7: mdp, 5: msr},
            default_strategy=mdp,
        )
    return strategies


def strategy_family(name: str) -> str:
    for family in ("EW", "GMV", "MSR", "MDP", "RP", "HRP", "SWITCH"):
        if name.startswith(family):
            return family
    return name


def match_marker(current_family: str, recommended: str) -> str:
    rec_family = strategy_family(recommended)
    if rec_family == current_family:
        return "✓" if "(" not in recommended else "≈ (variant)"
    return "✗"


def _load_or_run_horse_race(
    panel: Panel,
    strategies: dict[str, object],
) -> dict[str, pd.Series]:
    cache_path = Path(PORTFOLIO_CACHE)
    prices_path = Path(CACHE)

    if cache_path.exists() and cache_path.stat().st_mtime > prices_path.stat().st_mtime:
        print("loaded from cache")
        wide = pd.read_parquet(cache_path)
        return {col: wide[col].dropna() for col in wide.columns}

    print(f"regenerating cache — running {len(strategies)} strategies ({START} → {END})...")
    portfolio_returns: dict[str, pd.Series] = {}
    for name, strategy in strategies.items():
        print(f"  {name}...", end="", flush=True)
        result = run_horse_race(panel, strategy, start=START, end=END)
        portfolio_returns[name] = result["portfolio_returns"].dropna()
        print(f"  Sharpe={result['stats']['sharpe_ratio']:.3f}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(portfolio_returns).to_parquet(cache_path)
    return portfolio_returns


def main() -> None:
    print("Loading data...")
    prices = pd.read_parquet(CACHE)
    regimes_df = pd.read_parquet(REGIME_CACHE)
    panel = Panel({"prices": prices, "regimes": regimes_df})
    dominant_regime = regimes_df["dominant_regime"].dropna()

    strategies = build_strategies()
    portfolio_returns = _load_or_run_horse_race(panel, strategies)

    print("\nComputing regime-conditional performance...")
    tables = regime_conditional_performance(
        portfolio_returns, dominant_regime, min_days=MIN_DAYS
    )

    sharpe_df = tables["sharpe"]
    n_days_df = tables["n_days"]

    # Use first strategy's n_days as canonical coverage (same panel/harness for all)
    coverage = n_days_df.iloc[0].astype(int)

    # ── Table 1: Sharpe by strategy × regime ────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 1 — Sharpe ratio by strategy × regime  (columns = regime 0..7)")
    print("=" * 80)
    cov_row = pd.DataFrame([coverage], index=["n_days (coverage)"])
    display = pd.concat([cov_row, sharpe_df.round(3)])
    print(display.to_string())

    # ── Table 2: Best strategy per regime ───────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 2 — Best strategy per regime  (* = fewer than 252 trading days)")
    print("=" * 80)
    rows = []
    for r in range(8):
        col = sharpe_df[r].drop(labels=list(EXCLUDE_FROM_REC), errors="ignore")
        best_name = col.idxmax() if col.notna().any() else "N/A"
        best_sharpe = col.max()
        days = int(coverage[r])
        rows.append({
            "regime": r,
            "best_strategy": best_name if best_name else "N/A",
            "sharpe": f"{best_sharpe:.3f}" if pd.notna(best_sharpe) else "NaN",
            "n_days": days,
            "sparse": "*" if days < MIN_DAYS else "",
        })
    best_df = pd.DataFrame(rows).set_index("regime")
    print(best_df.to_string())

    # ── Table 3: Switching rule comparison ──────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 3 — Switching rule: paam_lab 19d vs. analysis-recommended")
    print("  Current rule: R0→EW, R5→MSR, others→MDP  (from SWITCH(ledoit_wolf) config)")
    print("=" * 80)
    col_c = 22
    col_r = 32
    header = f"{'Current rule':<{col_c}}  {'Analysis-recommended':<{col_r}}  Match"
    print(header)
    print("-" * len(header))

    for r in range(8):
        current_family = CURRENT_RULE[r]
        col = sharpe_df[r].drop(labels=list(EXCLUDE_FROM_REC), errors="ignore")
        col = col[~col.index.str.startswith("SWITCH(")]
        recommended = col.idxmax() if col.notna().any() else "N/A"
        days = int(coverage[r])

        if recommended == "N/A":
            marker = "?"
            suffix = ""
        else:
            marker = match_marker(current_family, recommended)
            suffix = f" — but only {days} days coverage" if days < MIN_DAYS and marker == "✗" else ""

        c_label = f"R{r} → {current_family}"
        r_label = f"R{r} → {recommended}"
        print(f"{c_label:<{col_c}}  {r_label:<{col_r}}  {marker}{suffix}")

    print()


if __name__ == "__main__":
    main()
