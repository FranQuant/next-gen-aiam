from __future__ import annotations

import numpy as np
import pandas as pd


def build_evidence(
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookbacks: tuple[int, ...] = (21, 63, 252),
) -> pd.DataFrame:
    """Per-asset trailing returns and annualized vol up to asof (no look-ahead)."""
    data = returns.loc[:asof]
    records = []
    max_lb = max(lookbacks)

    for ticker in data.columns:
        col = data[ticker].dropna()
        row: dict = {"asset": ticker}
        for lb in lookbacks:
            subset = col.iloc[-lb:] if len(col) >= lb else col
            row[f"ret_{lb}d"] = float((1 + subset).prod() - 1) if len(subset) > 0 else float("nan")
        ann_window = col.iloc[-max_lb:] if len(col) >= 2 else col
        row["ann_vol"] = float(ann_window.std() * np.sqrt(252)) if len(ann_window) > 1 else float("nan")
        records.append(row)

    return pd.DataFrame(records).set_index("asset")


def evidence_to_text(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:+.4f}")
