from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiam.llm.client import MockClient
from aiam.llm.schemas import ParseError
from aiam.llm.views import LLMViewGenerator


def _make_returns(tickers=("SPY", "IEF", "GLD"), n_obs: int = 280, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_obs)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (n_obs, len(tickers))),
        index=dates,
        columns=list(tickers),
    )


def _response(*views: dict) -> str:
    return json.dumps({"views": list(views)})


# ── Shape and value checks ───────────────────────────────────────────────────

def test_p_shape_k_by_n():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.10, "confidence": 0.7},
        {"asset": "IEF", "expected_excess_return": -0.02, "confidence": 0.4},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    n = len(returns.columns)
    assert P.shape == (2, n), f"P.shape={P.shape}"
    assert Q.shape == (2,)
    assert Omega.shape == (2, 2)


def test_p_rows_are_one_hot():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.6},
        {"asset": "GLD", "expected_excess_return": 0.05, "confidence": 0.5},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, _, _ = gen(returns, asof)

    for row in P:
        assert row.sum() == pytest.approx(1.0)
        assert (row >= 0).all()
        assert (row <= 1).all()


def test_q_values_match_views():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.12, "confidence": 0.8},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    assert Q[0] == pytest.approx(0.12)


def test_asset_alignment():
    """P rows must be aligned to returns.columns order, not view order."""
    tickers = ("SPY", "IEF", "GLD")
    returns = _make_returns(tickers=tickers)
    asof = returns.index[-1]

    resp = _response(
        {"asset": "GLD", "expected_excess_return": 0.07, "confidence": 0.6},
        {"asset": "SPY", "expected_excess_return": 0.10, "confidence": 0.7},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    col_idx = {c: i for i, c in enumerate(tickers)}
    # First view in P is GLD → column 2, second is SPY → column 0
    assert P[0, col_idx["GLD"]] == pytest.approx(1.0)
    assert P[1, col_idx["SPY"]] == pytest.approx(1.0)


def test_omega_diagonal_and_positive():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.7},
        {"asset": "IEF", "expected_excess_return": -0.03, "confidence": 0.3},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    _, _, Omega = gen(returns, asof)

    assert np.allclose(Omega - np.diag(np.diagonal(Omega)), 0), "Omega not diagonal"
    assert (np.diagonal(Omega) > 0).all(), "Omega diagonal must be positive"


def test_higher_confidence_lower_omega():
    """Higher confidence view must produce smaller Omega diagonal entry."""
    returns = _make_returns(tickers=("SPY", "IEF"))
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.9},
        {"asset": "IEF", "expected_excess_return": 0.02, "confidence": 0.1},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    _, _, Omega = gen(returns, asof)

    # SPY (higher confidence) should have smaller diagonal entry
    # Note: P row 0 → SPY, row 1 → IEF (alphabetical order of views)
    assert Omega[0, 0] < Omega[1, 1]


# ── Asset filtering ──────────────────────────────────────────────────────────

def test_unknown_assets_excluded():
    """Assets returned by LLM that are not in universe get no P row."""
    returns = _make_returns(tickers=("SPY", "IEF"))
    asof = returns.index[-1]
    resp = _response(
        {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.7},
        {"asset": "UNKNOWN_TICKER", "expected_excess_return": 0.05, "confidence": 0.5},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    assert P.shape == (1, 2)  # only SPY
    assert Q.shape == (1,)


def test_all_assets_omitted_returns_empty():
    """LLM returns views only for unknown tickers → empty arrays."""
    returns = _make_returns(tickers=("SPY", "IEF"))
    asof = returns.index[-1]
    resp = _response(
        {"asset": "NOT_IN_UNIVERSE", "expected_excess_return": 0.08, "confidence": 0.7},
    )
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    assert P.shape == (0, 2)
    assert len(Q) == 0


def test_empty_views_list_returns_empty():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = json.dumps({"views": []})
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    assert P.shape == (0, 3)
    assert len(Q) == 0


# ── Fail-closed / strict ─────────────────────────────────────────────────────

def test_fail_closed_on_garbage_output():
    returns = _make_returns()
    asof = returns.index[-1]
    gen = LLMViewGenerator(MockClient(["this is not json at all"]))
    P, Q, Omega = gen(returns, asof)

    assert P.shape == (0, 3)
    assert len(Q) == 0
    assert Omega.shape == (0, 0)


def test_fail_closed_on_schema_violation():
    returns = _make_returns()
    asof = returns.index[-1]
    # confidence > 1 → validation error
    resp = json.dumps({"views": [{"asset": "SPY", "expected_excess_return": 0.08, "confidence": 5.0}]})
    gen = LLMViewGenerator(MockClient([resp]))
    P, Q, Omega = gen(returns, asof)

    assert len(Q) == 0


def test_strict_raises_on_garbage():
    returns = _make_returns()
    asof = returns.index[-1]
    gen = LLMViewGenerator(MockClient(["garbage"]), strict=True)

    with pytest.raises(ParseError):
        gen(returns, asof)


def test_strict_raises_on_schema_violation():
    returns = _make_returns()
    asof = returns.index[-1]
    resp = json.dumps({"views": [{"asset": "SPY", "expected_excess_return": 99.0, "confidence": 0.5}]})
    gen = LLMViewGenerator(MockClient([resp]), strict=True)

    with pytest.raises(ParseError):
        gen(returns, asof)
