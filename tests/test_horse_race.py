import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.global_min_variance import GlobalMinVariance

CACHE = "data/cache/prices_30.parquet"
START = "2008-01-01"
END = "2026-04-30"


def load_panel() -> Panel:
    prices = pd.read_parquet(CACHE)
    return Panel({"prices": prices})


def test_equal_weight_horse_race():
    panel = load_panel()
    strategy = EqualWeight()

    result = run_horse_race(panel, strategy, start=START, end=END)

    stats = result["stats"]
    assert 0.05 <= stats["annualized_volatility"] <= 0.30, (
        f"annualized_volatility={stats['annualized_volatility']:.4f} outside [0.05, 0.30]"
    )


def test_four_way_comparison(capsys):
    panel = load_panel()

    strategies = {
        "EW": EqualWeight(),
        "GMV(sample)": GlobalMinVariance(sample_cov),
        "GMV(ledoit_wolf)": GlobalMinVariance(ledoit_wolf_cov),
        "GMV(oas)": GlobalMinVariance(oas_cov),
    }

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

    print("\n--- Four-way strategy comparison (2008-01-01 to 2026-04-30) ---")
    print(display.to_string())

    for name in strategies:
        vol = df.loc[name, "annualized_volatility"]
        assert 0.005 <= vol <= 0.35, f"{name}: vol={vol:.4f} out of range"
