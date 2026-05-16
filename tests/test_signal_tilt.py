from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.data.panel import Panel
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.signal_tilt import SignalTilt, momentum_signal_fn

N_ASSETS = 5
N_ROWS = 300
TICKERS = [f"A{i}" for i in range(N_ASSETS)]
DATES = pd.bdate_range("2020-01-01", periods=N_ROWS)


def _make_panel(seed: int = 42) -> Panel:
    rng = np.random.default_rng(seed)
    prices = pd.DataFrame(
        rng.lognormal(0, 0.01, (N_ROWS, N_ASSETS)).cumprod(axis=0) * 100,
        index=DATES,
        columns=TICKERS,
    )
    returns = prices.pct_change().fillna(0)
    return Panel({"prices": prices, "returns": returns})


# ── Basic constraints ─────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    panel = _make_panel()
    strat = SignalTilt(signal_fn=lambda r: r.sum())
    w = strat.predict_weights(panel, DATES[-1])
    assert abs(w.sum() - 1.0) < 1e-9


def test_weights_nonnegative():
    panel = _make_panel()
    strat = SignalTilt(signal_fn=lambda r: r.sum())
    w = strat.predict_weights(panel, DATES[-1])
    assert (w >= 0).all()


# ── Tilt behaviour ────────────────────────────────────────────────────────────

def test_zero_tilt_matches_base():
    panel = _make_panel()
    base = EqualWeight()
    strat = SignalTilt(signal_fn=lambda r: r.sum(), tilt_strength=0.0, base=base)
    w = strat.predict_weights(panel, DATES[-1])
    base_w = base.predict_weights(panel, DATES[-1])
    common = w.index.intersection(base_w.index)
    np.testing.assert_allclose(w.loc[common].values, base_w.loc[common].values, atol=1e-9)


def test_strong_tilt_concentrates_toward_top():
    panel = _make_panel()

    def extreme_signal(returns: pd.DataFrame) -> pd.Series:
        s = pd.Series(0.0, index=returns.columns)
        s.iloc[0] = 1000.0
        return s

    strat = SignalTilt(signal_fn=extreme_signal, tilt_strength=1.0)
    w = strat.predict_weights(panel, DATES[-1])
    assert w.idxmax() == TICKERS[0]


# ── No look-ahead ─────────────────────────────────────────────────────────────

def test_no_look_ahead():
    """Perturbing future returns must not change weights at t_mid."""
    panel = _make_panel()
    t_mid = DATES[N_ROWS // 2]
    strat = SignalTilt(signal_fn=lambda r: r.sum(), tilt_strength=0.5)
    w1 = strat.predict_weights(panel, t_mid)

    ret_mod = panel.data["returns"].copy()
    ret_mod.loc[ret_mod.index > t_mid] *= 100.0
    panel2 = Panel({"prices": panel.data["prices"], "returns": ret_mod})
    w2 = strat.predict_weights(panel2, t_mid)

    np.testing.assert_allclose(w1.values, w2.values, atol=1e-12)


# ── Default signal function ───────────────────────────────────────────────────

def test_default_momentum_signal_fn_weights_valid():
    panel = _make_panel()
    strat = SignalTilt(signal_fn=momentum_signal_fn)
    w = strat.predict_weights(panel, DATES[-1])
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_momentum_signal_fn_returns_series():
    panel = _make_panel()
    returns = panel.slice(DATES[-1], "returns", lookback=252)
    s = momentum_signal_fn(returns)
    assert isinstance(s, pd.Series)
    assert len(s) == N_ASSETS
