"""Train/test split constants and helpers for OOS evaluation."""
from __future__ import annotations

import pandas as pd

TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")


def split_train_test(df: pd.DataFrame | pd.Series, date_col: str | None = None):
    """Return (train, test) tuple split at TRAIN_END / TEST_START boundary."""
    if date_col is not None:
        idx = df[date_col]
        return df[idx <= TRAIN_END], df[idx >= TEST_START]
    return df.loc[df.index <= TRAIN_END], df.loc[df.index >= TEST_START]
