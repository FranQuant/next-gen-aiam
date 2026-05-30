# Appendix A — Universe Construction

## A.1 Sample Window

The study spans **2003-01-02 to 2026-04-30**, comprising **5,869 NYSE trading days** (23.3 calendar years). Daily OHLCV data are sourced from EODHD at ticker-level granularity and adjusted for splits and dividends via the `adj_close` field. All price series are pre-processed at daily frequency; intraday data are not used in the main harness (they are used only in the Notebook 05 direct-weight extension).

## A.2 Asset Universe

The 29-asset universe spans six groups designed to represent a diversified multi-asset allocation problem. Tickers are given in EODHD notation (`TICKER.EXCHANGE`).

| Ticker | Name | Asset Group | First Obs. |
|---|---|---|---|
| AAPL.US | Apple Inc. | US Equity — Single Names | 2003-01-02 |
| MSFT.US | Microsoft Corporation | US Equity — Single Names | 2003-01-02 |
| GOOGL.US | Alphabet Inc. (Class A) | US Equity — Single Names | 2004-08-19 |
| NVDA.US | NVIDIA Corporation | US Equity — Single Names | 2003-01-02 |
| JPM.US | JPMorgan Chase & Co. | US Equity — Single Names | 2003-01-02 |
| JNJ.US | Johnson & Johnson | US Equity — Single Names | 2003-01-02 |
| XOM.US | ExxonMobil Corporation | US Equity — Single Names | 2003-01-02 |
| WMT.US | Walmart Inc. | US Equity — Single Names | 2003-01-02 |
| XLK.US | Technology Select Sector SPDR | US Sector ETF | 2003-01-02 |
| XLF.US | Financial Select Sector SPDR | US Sector ETF | 2003-01-02 |
| XLE.US | Energy Select Sector SPDR | US Sector ETF | 2003-01-02 |
| XLV.US | Health Care Select Sector SPDR | US Sector ETF | 2003-01-02 |
| XLP.US | Consumer Staples Select Sector SPDR | US Sector ETF | 2003-01-02 |
| XLU.US | Utilities Select Sector SPDR | US Sector ETF | 2003-01-02 |
| SPY.US | SPDR S&P 500 ETF | Broad US Equity ETF | 2003-01-02 |
| IWM.US | iShares Russell 2000 ETF | Broad US Equity ETF | 2003-01-02 |
| EFA.US | iShares MSCI EAFE ETF | International Equity ETF | 2003-01-02 |
| EEM.US | iShares MSCI Emerging Markets ETF | International Equity ETF | 2003-04-11 |
| FXI.US | iShares China Large-Cap ETF | International Equity ETF | 2004-10-08 |
| SHY.US | iShares 1–3 Year Treasury Bond ETF | Fixed Income ETF | 2003-01-02 |
| IEF.US | iShares 7–10 Year Treasury Bond ETF | Fixed Income ETF | 2003-01-02 |
| TLT.US | iShares 20+ Year Treasury Bond ETF | Fixed Income ETF | 2003-01-02 |
| AGG.US | iShares Core U.S. Aggregate Bond ETF | Fixed Income ETF | 2003-09-26 |
| HYG.US | iShares iBoxx $ HY Corporate Bond ETF | Fixed Income ETF | 2007-04-11 |
| GLD.US | SPDR Gold Shares | Commodities / FX | 2004-11-18 |
| SLV.US | iShares Silver Trust | Commodities / FX | 2006-04-28 |
| DBC.US | Invesco DB Commodity Index Tracking Fund | Commodities / FX | 2006-02-03 |
| USO.US | United States Oil Fund | Commodities / FX | 2006-04-10 |
| EURUSD.FOREX | Euro / U.S. Dollar spot rate | Commodities / FX | 2003-01-02 |

First-observation dates are the earliest date with a non-missing `adj_close` in the published dataset (`data/published/ohlcv_29assets_2003_2026.csv`).

## A.3 Variable Universe N(t)

Nine tickers have first observations after the study start date of 2003-01-02 (EEM, AGG, GOOGL, FXI, GLD, SLV, DBC, USO, HYG). The universe size N(t) therefore grows from 20 at inception to 29 by 2007-04-11, when HYG, the last entrant, first trades.

Each strategy in the harness handles this transparently: a ticker is included in covariance estimation and weight allocation if and only if its return series has fewer than 10% missing observations in the relevant lookback window. Pre-inception rows are stored as NaN in the panel and are thus automatically excluded from `Σ̂` computation without any manual splice or override. Strategies that produce a weight vector return zero weight on excluded tickers; weights sum to 1 over the active sub-universe at each rebalance date.

This variable-N(t) design eliminates the forward-fill survivorship bias that would arise from imputing constant values in periods before a security's launch.

## A.4 BTC Exclusion

Bitcoin (BTC-USD) was considered for inclusion as an alternative asset but was excluded from the 29-asset universe for the following reasons. First, Bitcoin's sample history begins in 2010, which would truncate the full-period comparison by seven years and discard the 2003–2009 crisis and recovery regime that is a key source of cross-strategy discrimination in this study. Second, including a single asset with a dramatically higher realized volatility (≈ 80 % annualized versus ≈ 14 % for the equity names) would distort mean-variance estimators and make cross-strategy comparisons less interpretable. Third, the harness is designed to evaluate strategies as deployed by institutional managers; Bitcoin has limited institutional custody history prior to 2021 and zero weight in standard benchmark indices throughout most of the sample.

The 30-asset universe that includes BTC-USD (`UNIVERSE_30`) is retained in `src/aiam/data/universe.py` for backward compatibility with earlier pipeline artifacts but is not used in any result reported in this paper.
