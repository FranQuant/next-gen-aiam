"""Universe definitions for the AIAM harness."""
from __future__ import annotations

# 29-asset universe (BTC-USD dropped for survivorship; new default from Session 1.5A)
UNIVERSE_29: list[str] = [
    # US Equity Mega-Caps (8)
    "AAPL.US", "MSFT.US", "GOOGL.US", "NVDA.US", "JPM.US", "JNJ.US", "XOM.US", "WMT.US",
    # US Sector ETFs (6)
    "XLK.US", "XLF.US", "XLE.US", "XLV.US", "XLP.US", "XLU.US",
    # Broad US Equity (2)
    "SPY.US", "IWM.US",
    # International (3)
    "EFA.US", "EEM.US", "FXI.US",
    # Bonds (5)
    "SHY.US", "IEF.US", "TLT.US", "AGG.US", "HYG.US",
    # Commodities + FX (5)
    "GLD.US", "SLV.US", "DBC.US", "USO.US", "EURUSD.FOREX",
]

# 30-asset universe (includes BTC-USD; retained for backward compatibility with 2008-2026 caches)
UNIVERSE_30: list[str] = UNIVERSE_29 + ["BTC-USD.CC"]

# Tickers with shorter history — pre-inception rows are NaN in the panel
SHORT_HISTORY: dict[str, str] = {
    "FXI.US": "2004-01-01",
    "DBC.US": "2006-01-01",
    "USO.US": "2006-01-01",
    "HYG.US": "2007-01-01",
}
