# Published Datasets — next-gen-aiam

Reproduction-grade artifacts for the paper "Comparative Asset Allocation Harness: A 62-Strategy Walk-Forward Study" (Salazar, 2026). Use these to verify, reproduce, or extend the published results.

Source commit: e4b07bae7886c6c1810fe165c3afa1b94b9bbc92

## Contents

| File | Format | Rows | Description |
|---|---|---|---|
| `master_table_62strategies.csv` | CSV | 62 | Per-strategy summary statistics (Family, Strategy, Ann Ret, Ann Vol, Sharpe, Hit%, Max DD, Calmar, Turnover, Net 10bps, NetStrat). Matches Table 1 in §3.1 of the paper. |
| `strategy_returns_base.parquet` | Parquet | 5,868 daily rows × 31 base strategies | Daily return series for all 31 base strategies, 2003-01-03 to 2026-04-30. Column names match the master table's Strategy column. |
| `strategy_returns_vmp.parquet` | Parquet | 5,868 daily rows × 31 VMP variants | Daily return series for all 31 VMP-overlay variants. Same date range. |
| `regime_signals.parquet` | Parquet | 317 monthly rows × 9 columns | Monthly regime classifications, 2000-01-31 to 2026-05-31. Eight individual indicator columns (regime_GDP, regime_VIX, regime_SPX, regime_CPI, regime_UNEM, regime_YC10, regime_YC2, regime_YCSTEP) plus a dominant_regime column (integers 0-7), produced by the 8-indicator FRED-based regime engine (see paper §2.5). |
| `ohlcv_29assets_2003_2026.csv` | CSV | 170,201 daily rows × 8 fields | Open/High/Low/Close/AdjClose/Volume for the 29-asset universe (long format, columns: date, ticker, open, high, low, close, adj_close, volume). Source: EODHD Pro. Date range: 2003-01-02 to 2026-04-30. |

## Universe (29 assets)

US single stocks: AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, WMT.
US sector ETFs: XLK, XLF, XLE, XLV, XLP, XLU.
Broad equity ETFs: SPY, IWM.
International equity ETFs: EFA, EEM, FXI.
Fixed income ETFs: AGG, TLT, IEF, SHY, HYG.
Commodities & FX: GLD, SLV, DBC, USO.
EUR/USD spot: EURUSD.

Four tickers have shorter histories: GOOGL (from 2004-08-19), FXI (from 2004-10-08), GLD (from 2004-11-18), HYG (from 2007-04-11). The walk-forward harness handles variable universe size N(t) via column-level NaN filtering — see paper §2.2.

BTC-USD is excluded entirely for survivorship hygiene (see paper §2.14 and Finding 15).

## Reproduction levels

**Level 1 — verify numbers (1 minute):** Open `master_table_62strategies.csv` in Excel/pandas/R. The Sharpe, Ann Ret, Max DD columns reproduce Table 1 in §3.1 of the paper directly.

**Level 2 — reproduce metrics from returns (5 minutes):** Load `strategy_returns_base.parquet` and `strategy_returns_vmp.parquet`. Compute annualized Sharpe = mean × √252 / std, max drawdown, turnover, etc. Results should match `master_table_62strategies.csv` within ±0.001 Sharpe / ±0.05%pt Max DD.

**Level 3 — re-derive returns from prices (1-2 hours):** Load `ohlcv_29assets_2003_2026.csv`. Run the full pipeline via the scripts in `scripts/`:
```
python scripts/build_returns_29.py
python scripts/build_regime_signals_2003.py
python scripts/build_all_strategies_29.py
python scripts/build_weights_cache.py
python scripts/build_switch_v2a_weights.py
```
Strategy returns will be regenerated into `data/cache/portfolio_returns/`; compare against the parquets in this directory.

## Versioning

These artifacts are versioned with the paper. If the paper is updated, the artifacts in this directory will be updated to match. The git commit hash at the top of this file identifies the paper version these correspond to.

## Citation

If you use this dataset, please cite the paper:

> Salazar, F. (2026). Comparative Asset Allocation Harness: A 62-Strategy Walk-Forward Study.

## Licensing & data source notes

- OHLCV price data sourced from EODHD Pro (https://eodhd.com). Redistribution of this OHLCV subset is intended for academic/reproduction purposes. Commercial use of the underlying price data should follow EODHD's licensing terms.
- Macroeconomic indicators for regime classification sourced from FRED (Federal Reserve Economic Data, https://fred.stlouisfed.org), which is in the public domain.
- The strategy returns and regime signals in this directory are derived artifacts produced by code in this repository (Apache 2.0).
