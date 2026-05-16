"""Asset-class taxonomy and one-hot encoding for the 29-asset universe."""
from __future__ import annotations

import warnings

import pandas as pd


ASSET_CLASS_MAP: dict[str, str] = {
    "AAPL.US":      "us_single_stock",
    "MSFT.US":      "us_single_stock",
    "GOOGL.US":     "us_single_stock",
    "NVDA.US":      "us_single_stock",
    "JPM.US":       "us_single_stock",
    "JNJ.US":       "us_single_stock",
    "XOM.US":       "us_single_stock",
    "WMT.US":       "us_single_stock",
    "XLK.US":       "us_sector_etf",
    "XLF.US":       "us_sector_etf",
    "XLE.US":       "us_sector_etf",
    "XLV.US":       "us_sector_etf",
    "XLP.US":       "us_sector_etf",
    "XLU.US":       "us_sector_etf",
    "SPY.US":       "broad_equity_etf",
    "IWM.US":       "broad_equity_etf",
    "EFA.US":       "intl_equity_etf",
    "EEM.US":       "intl_equity_etf",
    "FXI.US":       "intl_equity_etf",
    "AGG.US":       "fixed_income_etf",
    "TLT.US":       "fixed_income_etf",
    "IEF.US":       "fixed_income_etf",
    "SHY.US":       "fixed_income_etf",
    "HYG.US":       "fixed_income_etf",
    "GLD.US":       "commodity_etf",
    "SLV.US":       "commodity_etf",
    "DBC.US":       "commodity_etf",
    "USO.US":       "commodity_etf",
    "EURUSD.FOREX": "fx_spot",
}
# 7 asset classes, 29 assets

_ALL_CLASSES = [
    "us_single_stock",
    "us_sector_etf",
    "broad_equity_etf",
    "intl_equity_etf",
    "fixed_income_etf",
    "commodity_etf",
    "fx_spot",
]


def asset_class_one_hot(assets: list[str] | pd.Index) -> pd.DataFrame:
    """Return a (n_assets × 7) boolean one-hot DataFrame indexed by asset symbol.

    Columns are the 7 canonical asset classes. Unknown assets get all-zero rows
    and emit a UserWarning. Intended as a feature panel join key in ML training.
    """
    rows = {}
    unknown = []
    for asset in assets:
        cls = ASSET_CLASS_MAP.get(asset)
        if cls is None:
            unknown.append(asset)
            rows[asset] = {c: False for c in _ALL_CLASSES}
        else:
            rows[asset] = {c: (c == cls) for c in _ALL_CLASSES}
    if unknown:
        warnings.warn(f"asset_class_one_hot: unknown assets set to all-zero: {unknown}", UserWarning)
    return pd.DataFrame.from_dict(rows, orient="index", columns=_ALL_CLASSES)
