from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.evaluation.ic import ic_summary, information_coefficient

N_DATES = 100
N_ASSETS = 8
TICKERS = [f"A{i}" for i in range(N_ASSETS)]
DATES = pd.bdate_range("2020-01-01", periods=N_DATES)


def _df(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=TICKERS)


def _random(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _df(rng.standard_normal((N_DATES, N_ASSETS)))


# ── Perfect / anti predictors ─────────────────────────────────────────────────

def test_perfect_predictor_spearman():
    df = _random()
    ic = information_coefficient(df, df, method="spearman")
    assert ic.mean() > 0.99


def test_perfect_predictor_pearson():
    df = _random()
    ic = information_coefficient(df, df, method="pearson")
    assert ic.mean() > 0.99


def test_anti_predictor_spearman():
    df = _random()
    ic = information_coefficient(-df, df, method="spearman")
    assert ic.mean() < -0.99


def test_anti_predictor_pearson():
    df = _random()
    ic = information_coefficient(-df, df, method="pearson")
    assert ic.mean() < -0.99


# ── Random signal ─────────────────────────────────────────────────────────────

def test_random_signal_low_ic():
    sig = _random(seed=1)
    fwd = _random(seed=2)
    ic = information_coefficient(sig, fwd, method="spearman")
    assert abs(ic.mean()) < 0.25


# ── min_assets filter ─────────────────────────────────────────────────────────

def test_min_assets_filter_returns_nan():
    sig = _random().copy()
    fwd = _random().copy()
    # Only 2 valid assets in first 10 rows
    sig.iloc[:10, 2:] = np.nan
    ic = information_coefficient(sig, fwd, method="spearman", min_assets=5)
    assert ic.iloc[:10].isna().all()
    # Rows with full data should be valid
    assert ic.iloc[10:].notna().all()


# ── Spearman vs Pearson ───────────────────────────────────────────────────────

def test_spearman_vs_pearson_differ_on_nonlinear():
    rng = np.random.default_rng(42)
    base = _df(rng.standard_normal((N_DATES, N_ASSETS)))
    # x^3 is a monotone transform: Spearman IC = 1, Pearson IC < 1
    sig = base ** 3
    fwd = base
    ic_sp = information_coefficient(sig, fwd, method="spearman")
    ic_pe = information_coefficient(sig, fwd, method="pearson")
    assert ic_sp.mean() > 0.99
    assert not np.allclose(ic_sp.dropna().values, ic_pe.dropna().values, atol=1e-6)


# ── ic_summary ────────────────────────────────────────────────────────────────

def test_ic_summary_keys():
    df = _random()
    ic = information_coefficient(df, df)
    summary = ic_summary(ic)
    assert set(summary.keys()) == {"mean", "std", "t_stat", "hit_rate", "ir", "n_obs"}


def test_ic_summary_perfect_predictor():
    df = _random()
    ic = information_coefficient(df, df)
    s = ic_summary(ic)
    assert s["mean"] > 0.99
    assert s["hit_rate"] >= 0.99
    # std=0 when all IC=1.0 → IR undefined; either IR is high or std is effectively zero
    assert s["std"] < 1e-10 or s["ir"] > 10
    assert s["n_obs"] == N_DATES


def test_ic_summary_nan_excluded_from_n_obs():
    df = _random().copy()
    df.iloc[:10, :] = np.nan  # first 10 rows all NaN → IC = NaN
    ic = information_coefficient(df, df, min_assets=5)
    s = ic_summary(ic)
    assert s["n_obs"] == N_DATES - 10
