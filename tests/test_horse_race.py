import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight

CACHE = "data/cache/prices_30.parquet"


def load_panel() -> Panel:
    prices = pd.read_parquet(CACHE)
    return Panel({"prices": prices})


def test_equal_weight_horse_race():
    panel = load_panel()
    strategy = EqualWeight()

    result = run_horse_race(
        panel,
        strategy,
        start="2020-01-01",
        end="2024-12-31",
    )

    stats = result["stats"]
    print("\n--- EqualWeight stats (2020-01-01 to 2024-12-31) ---")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    assert 0.05 <= stats["annualized_volatility"] <= 0.30, (
        f"annualized_volatility={stats['annualized_volatility']:.4f} outside [0.05, 0.30]"
    )
