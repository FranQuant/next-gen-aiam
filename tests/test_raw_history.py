"""Tests for src/aiam/features/raw_history.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.features.raw_history import extract_raw_history, get_raw_feature_cols

LOOKBACK = 10
N_DATES = 30
N_ASSETS = 2


def _make_returns(n_dates=N_DATES, n_assets=N_ASSETS, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.standard_normal((n_dates, n_assets)) * 0.01, index=dates, columns=cols)


# ── get_raw_feature_cols ──────────────────────────────────────────────────────

def test_get_cols_returns_only():
    cols = get_raw_feature_cols(5, include_rp=False)
    assert cols == ["r_lag_000", "r_lag_001", "r_lag_002", "r_lag_003", "r_lag_004"]


def test_get_cols_with_rp():
    cols = get_raw_feature_cols(3, include_rp=True)
    assert cols == ["r_lag_000", "r_lag_001", "r_lag_002", "rp_lag_000", "rp_lag_001", "rp_lag_002"]


def test_get_cols_deterministic():
    a = get_raw_feature_cols(252, include_rp=True)
    b = get_raw_feature_cols(252, include_rp=True)
    assert a == b
    assert len(a) == 504


# ── extract_raw_history — returns-only mode ───────────────────────────────────

def test_returns_only_shape():
    ret = _make_returns(N_DATES, N_ASSETS)
    panel = extract_raw_history(ret, lookback=LOOKBACK)
    # valid dates = N_DATES - LOOKBACK + 1; rows = valid_dates × n_assets
    expected_rows = (N_DATES - LOOKBACK + 1) * N_ASSETS
    assert panel.shape == (expected_rows, LOOKBACK)


def test_returns_only_column_order():
    ret = _make_returns()
    panel = extract_raw_history(ret, lookback=LOOKBACK)
    expected = [f"r_lag_{k:03d}" for k in range(LOOKBACK)]
    assert panel.columns.tolist() == expected


def test_returns_only_no_nan():
    ret = _make_returns()
    panel = extract_raw_history(ret, lookback=LOOKBACK)
    assert not panel.isna().any().any()


# ── extract_raw_history — returns + RP mode ───────────────────────────────────

def test_rp_mode_shape():
    ret = _make_returns()
    rng = np.random.default_rng(1)
    rp = pd.DataFrame(
        np.abs(rng.standard_normal((N_DATES, N_ASSETS))) * 1e-4,
        index=ret.index,
        columns=ret.columns,
    )
    panel = extract_raw_history(ret, rp, lookback=LOOKBACK)
    expected_rows = (N_DATES - LOOKBACK + 1) * N_ASSETS
    assert panel.shape == (expected_rows, 2 * LOOKBACK)


def test_rp_mode_column_order():
    """r_lag columns first, rp_lag columns last, both zero-padded ascending."""
    ret = _make_returns()
    rp = _make_returns(seed=99)
    panel = extract_raw_history(ret, rp, lookback=LOOKBACK)
    cols = panel.columns.tolist()
    r_cols = [c for c in cols if c.startswith("r_lag")]
    rp_cols = [c for c in cols if c.startswith("rp_lag")]
    assert cols == r_cols + rp_cols
    assert r_cols == sorted(r_cols)
    assert rp_cols == sorted(rp_cols)


# ── MultiIndex structure ──────────────────────────────────────────────────────

def test_multiindex_names():
    ret = _make_returns()
    panel = extract_raw_history(ret, lookback=LOOKBACK)
    assert panel.index.names == ["date", "asset"]


def test_nan_rows_dropped():
    """Rows at dates with insufficient lookback must be absent from output."""
    ret = _make_returns(N_DATES, N_ASSETS)
    panel = extract_raw_history(ret, lookback=LOOKBACK)
    min_date = panel.index.get_level_values("date").min()
    # First LOOKBACK-1 dates (0-indexed 0..LOOKBACK-2) must not appear
    first_valid = ret.index[LOOKBACK - 1]  # 0-indexed: LOOKBACK-1
    assert min_date == first_valid, (
        f"Expected first valid date {first_valid.date()}, got {min_date.date()}"
    )


# ── Date alignment check ──────────────────────────────────────────────────────

def test_misaligned_indices_raise():
    ret = _make_returns()
    rp = _make_returns().iloc[1:]  # different length/index
    with pytest.raises(ValueError, match="date ind"):
        extract_raw_history(ret, rp, lookback=LOOKBACK)
