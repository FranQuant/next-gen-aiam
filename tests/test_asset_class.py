"""Tests for src/aiam/features/asset_class.py."""
from __future__ import annotations

import warnings

import pytest

from aiam.features.asset_class import ASSET_CLASS_MAP, asset_class_one_hot


def test_asset_class_map_29_assets():
    assert len(ASSET_CLASS_MAP) == 29


def test_asset_class_map_7_classes():
    classes = set(ASSET_CLASS_MAP.values())
    assert len(classes) == 7, f"Expected 7 classes, got {len(classes)}: {classes}"


def test_one_hot_shape():
    oh = asset_class_one_hot(list(ASSET_CLASS_MAP.keys()))
    assert oh.shape == (29, 7), f"Expected (29, 7), got {oh.shape}"


def test_one_hot_row_sums_to_one():
    oh = asset_class_one_hot(list(ASSET_CLASS_MAP.keys()))
    assert (oh.sum(axis=1) == 1).all(), "Every row must sum to exactly 1"


def test_one_hot_columns_are_class_names():
    oh = asset_class_one_hot(list(ASSET_CLASS_MAP.keys()))
    expected = {
        "us_single_stock", "us_sector_etf", "broad_equity_etf",
        "intl_equity_etf", "fixed_income_etf", "commodity_etf", "fx_spot",
    }
    assert set(oh.columns) == expected


def test_one_hot_known_asset_correct_class():
    oh = asset_class_one_hot(["GLD.US", "TLT.US"])
    assert bool(oh.loc["GLD.US", "commodity_etf"]) is True
    assert bool(oh.loc["TLT.US", "fixed_income_etf"]) is True


def test_one_hot_unknown_asset_all_zeros():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        oh = asset_class_one_hot(["UNKNOWN.XX"])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    assert (oh.loc["UNKNOWN.XX"] == 0).all()
    assert oh.sum().sum() == 0


def test_one_hot_mixed_known_unknown():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        oh = asset_class_one_hot(["AAPL.US", "UNKNOWN.XX"])
    assert oh.shape == (2, 7)
    assert bool(oh.loc["AAPL.US", "us_single_stock"]) is True
    assert (oh.loc["UNKNOWN.XX"] == 0).all()


def test_one_hot_accepts_pandas_index():
    import pandas as pd
    assets = pd.Index(list(ASSET_CLASS_MAP.keys()))
    oh = asset_class_one_hot(assets)
    assert oh.shape == (29, 7)
