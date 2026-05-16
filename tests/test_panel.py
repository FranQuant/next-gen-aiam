import pandas as pd
import pytest

from aiam.data.panel import Panel

PRICES_PATH = "data/cache/prices_29.parquet"


@pytest.fixture(scope="module")
def panel():
    return Panel(data={"prices": pd.read_parquet(PRICES_PATH)})


def test_construction_and_universe(panel):
    assert len(panel.universe) == 30


def test_slice_shape_and_asof(panel):
    result = panel.slice(asof="2024-01-15", kind="prices", lookback=60)
    assert result.shape == (60, 30)
    assert result.index.max() <= pd.Timestamp("2024-01-15")


def test_slice_copy_semantics(panel):
    result = panel.slice(asof="2024-01-15", kind="prices", lookback=60)
    first_date = result.index[0]
    first_col = result.columns[0]
    result.iloc[0, 0] = 999_999.0
    assert panel.data["prices"].loc[first_date, first_col] != 999_999.0
