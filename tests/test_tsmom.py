from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.strategy.tsmom import TSMOM

# Use tiny lookbacks so tests run in milliseconds
_SIG = 10   # signal_lookback
_VOL = 5    # vol_lookback
_TOTAL = _SIG + _VOL + 1  # 16 rows of prices needed after weekday filter
_N_ROWS = _TOTAL * 2 + 10  # enough for the 2× request inside the strategy


def _make_panel(price_matrix: np.ndarray, tickers: list[str]) -> tuple[Panel, pd.Timestamp]:
    """Build a Panel from a price matrix on a business-day index."""
    dates = pd.bdate_range("2020-01-02", periods=len(price_matrix))
    prices = pd.DataFrame(price_matrix, index=dates, columns=tickers)
    return Panel({"prices": prices}), dates[-1]


def test_all_trending_up_gives_positive_weights():
    """All assets trending up → all signals = +1 → all weights positive and sum to 1."""
    n_assets = 4
    tickers = [f"A{i}" for i in range(n_assets)]
    # Monotonically increasing prices for all assets
    prices = np.column_stack([
        np.linspace(100, 110, _N_ROWS) + i for i in range(n_assets)
    ])
    panel, asof = _make_panel(prices, tickers)

    strategy = TSMOM(signal_lookback=_SIG, vol_lookback=_VOL)
    weights = strategy.predict_weights(panel, asof)

    assert set(weights.index) <= set(tickers)
    assert (weights > 0).all(), "All up-trending assets should have positive weight"
    assert abs(weights.sum() - 1.0) < 1e-10


def test_long_only_zeros_out_down_trending():
    """Half assets trending up, half down → long-only TSMOM zeros the down-half."""
    up = ["UP0", "UP1", "UP2"]
    dn = ["DN0", "DN1", "DN2"]
    tickers = up + dn

    # Up assets: monotonically increasing; down assets: monotonically decreasing
    up_prices = np.column_stack([np.linspace(100, 120, _N_ROWS)] * len(up))
    dn_prices = np.column_stack([np.linspace(120, 100, _N_ROWS)] * len(dn))
    prices = np.hstack([up_prices, dn_prices])
    panel, asof = _make_panel(prices, tickers)

    strategy = TSMOM(signal_lookback=_SIG, vol_lookback=_VOL, long_only=True)
    weights = strategy.predict_weights(panel, asof)

    for t in up:
        assert weights[t] > 0, f"Up-trending asset {t} should have positive weight"
    for t in dn:
        assert weights[t] == 0.0, f"Down-trending asset {t} should have zero weight"
    assert abs(weights.sum() - 1.0) < 1e-10


def test_all_flat_returns_equal_weight_fallback():
    """Constant prices → zero momentum → all-flat → EW fallback."""
    n_assets = 3
    tickers = [f"F{i}" for i in range(n_assets)]
    # Perfectly flat prices → zero returns → signal = 0 for all → total = 0
    prices = np.ones((_N_ROWS, n_assets)) * 100.0
    panel, asof = _make_panel(prices, tickers)

    strategy = TSMOM(signal_lookback=_SIG, vol_lookback=_VOL)
    weights = strategy.predict_weights(panel, asof)

    expected = 1.0 / n_assets
    np.testing.assert_allclose(weights.values, expected, atol=1e-10)


def test_asof_guard_raises_on_look_ahead():
    """predict_weights raises ValueError when asof > train_until (PointInTime contract)."""
    n_assets = 2
    tickers = ["X", "Y"]
    prices = np.column_stack([np.linspace(100, 110, _N_ROWS)] * n_assets)
    panel, _ = _make_panel(prices, tickers)

    train_date = panel.dates[_N_ROWS // 2]
    future_date = panel.dates[-1]

    strategy = TSMOM(signal_lookback=_SIG, vol_lookback=_VOL)
    strategy.fit(panel, train_until=train_date)

    with pytest.raises(ValueError, match="Look-ahead violation"):
        strategy.predict_weights(panel, asof=future_date)


def test_weights_sum_to_one():
    """Weights always sum to 1.0 regardless of signal mix."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i}" for i in range(6)]
    prices = rng.lognormal(0, 0.02, size=(_N_ROWS, 6)).cumprod(axis=0) * 100
    panel, asof = _make_panel(prices, tickers)

    strategy = TSMOM(signal_lookback=_SIG, vol_lookback=_VOL)
    weights = strategy.predict_weights(panel, asof)

    assert abs(weights.sum() - 1.0) < 1e-10
    assert (weights >= 0).all()
