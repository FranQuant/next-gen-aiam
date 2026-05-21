from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.llm.evidence import build_evidence, evidence_to_text


def _make_returns(n_assets: int = 3, n_obs: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_obs)
    data = rng.normal(0.0, 0.01, size=(n_obs, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ── Shape and column checks ──────────────────────────────────────────────────

def test_output_shape(returns=None):
    returns = _make_returns(n_assets=4, n_obs=300)
    asof = returns.index[-1]
    ev = build_evidence(returns, asof)
    assert ev.shape[0] == 4  # one row per asset
    assert "ret_21d" in ev.columns
    assert "ret_63d" in ev.columns
    assert "ret_252d" in ev.columns
    assert "ann_vol" in ev.columns


def test_custom_lookbacks():
    returns = _make_returns()
    asof = returns.index[-1]
    ev = build_evidence(returns, asof, lookbacks=(5, 10))
    assert "ret_5d" in ev.columns
    assert "ret_10d" in ev.columns
    assert "ret_21d" not in ev.columns


def test_index_matches_columns():
    returns = _make_returns(n_assets=3)
    asof = returns.index[-1]
    ev = build_evidence(returns, asof)
    assert list(ev.index) == list(returns.columns)


def test_ann_vol_positive():
    returns = _make_returns()
    asof = returns.index[-1]
    ev = build_evidence(returns, asof)
    assert (ev["ann_vol"] > 0).all()


# ── No look-ahead ────────────────────────────────────────────────────────────

def test_no_lookahead_equivalence():
    """Truncating to asof before calling must give the same result."""
    returns = _make_returns(n_obs=300)
    asof = returns.index[200]  # mid-way

    result_full = build_evidence(returns, asof)
    result_trunc = build_evidence(returns.loc[:asof], asof)

    pd.testing.assert_frame_equal(result_full, result_trunc)


def test_future_rows_do_not_affect_output():
    """Adding rows after asof must not change evidence values."""
    returns = _make_returns(n_obs=300)
    asof = returns.index[200]

    base = build_evidence(returns, asof)

    # Append rows with extreme values after asof
    extra_dates = pd.bdate_range(asof + pd.Timedelta("1D"), periods=50)
    extra = pd.DataFrame(
        99.0,  # extreme value — would dominate if included
        index=extra_dates,
        columns=returns.columns,
    )
    extended = pd.concat([returns, extra])
    result = build_evidence(extended, asof)

    pd.testing.assert_frame_equal(base, result)


# ── evidence_to_text ─────────────────────────────────────────────────────────

def test_evidence_to_text_is_string():
    returns = _make_returns()
    asof = returns.index[-1]
    ev = build_evidence(returns, asof)
    text = evidence_to_text(ev)
    assert isinstance(text, str)
    assert len(text) > 0
    # Each asset ticker should appear in the table
    for col in returns.columns:
        assert col in text
