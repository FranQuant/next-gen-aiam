import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.estimators.mean import sample_mean
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.max_sharpe import MaximumSharpe
from aiam.strategy.most_diversified import MostDiversified
from aiam.strategy.hierarchical_risk_parity import HierarchicalRiskParity
from aiam.strategy.risk_parity import RiskParity
from aiam.strategy.switching import SwitchingStrategy

CACHE = "data/cache/prices_30.parquet"
REGIME_CACHE = "data/cache/regime_signals.parquet"
START = "2008-01-01"
END = "2026-04-30"


def load_panel() -> Panel:
    prices = pd.read_parquet(CACHE)
    regimes = pd.read_parquet(REGIME_CACHE)
    return Panel({"prices": prices, "regimes": regimes})


def test_equal_weight_horse_race():
    panel = load_panel()
    strategy = EqualWeight()

    result = run_horse_race(panel, strategy, start=START, end=END)

    stats = result["stats"]
    assert 0.05 <= stats["annualized_volatility"] <= 0.30, (
        f"annualized_volatility={stats['annualized_volatility']:.4f} outside [0.05, 0.30]"
    )


def test_strategy_comparison(capsys):
    panel = load_panel()

    strategies = {
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

    # SWITCH variants — paam_lab 19d rule: R0→EW, R5→MSR, all others→MDP
    for cov_label, cov_est, mean_est in [
        ("sample", sample_cov, sample_mean),
        ("ledoit_wolf", ledoit_wolf_cov, sample_mean),
    ]:
        ew = EqualWeight()
        mdp = MostDiversified(cov_estimator=cov_est)
        msr = MaximumSharpe(cov_estimator=cov_est, mean_estimator=mean_est)
        switch = SwitchingStrategy(
            switching_rule={
                0: ew, 1: mdp, 2: mdp, 3: mdp, 4: mdp, 6: mdp, 7: mdp,
                5: msr,
            },
            default_strategy=mdp,
        )
        strategies[f"SWITCH({cov_label})"] = switch

    rows = []
    for name, strategy in strategies.items():
        result = run_horse_race(panel, strategy, start=START, end=END)
        s = result["stats"]
        rows.append({
            "strategy": name,
            "annualized_return": s["annualized_return"],
            "annualized_volatility": s["annualized_volatility"],
            "sharpe_ratio": s["sharpe_ratio"],
            "max_drawdown": s["max_drawdown"],
        })

    df = pd.DataFrame(rows).set_index("strategy")

    fmt = {
        "annualized_return": "{:.2%}",
        "annualized_volatility": "{:.2%}",
        "sharpe_ratio": "{:.3f}",
        "max_drawdown": "{:.2%}",
    }
    display = df.copy()
    for col, f in fmt.items():
        display[col] = df[col].map(lambda v, f=f: f.format(v))

    print("\n--- Strategy comparison (2008-01-01 to 2026-04-30) ---")
    print(display.to_string())

    for name in strategies:
        vol = df.loc[name, "annualized_volatility"]
        assert 0.005 <= vol <= 0.50, f"{name}: vol={vol:.4f} out of range"
