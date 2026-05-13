# Architecture — next-gen-aiam

Contract document. Describes stable commitments; implementation fills in below this level.

---

## 1. Universe & Calendar

**30-ticker universe in 6 buckets**

| Bucket | Tickers |
|---|---|
| Large-cap equity | AAPL, MSFT, GOOGL, NVDA, JPM, JNJ, XOM, WMT |
| Sector ETFs | XLK, XLF, XLE, XLV, XLP, XLU |
| Broad equity ETFs | SPY, IWM |
| International equity ETFs | EFA, EEM, FXI |
| Fixed income ETFs | SHY, IEF, TLT, AGG, HYG |
| Commodities + FX/crypto | GLD, SLV, DBC, USO, EURUSD.FOREX, BTC-USD.CC |

- `paam_lab` 8-ticker basket (AAPL, NVDA, JPM, SPY, GLD, TLT, EURUSD, BTC-USD) is a strict subset.
- Canonical calendar: **US business days** (`B` freq, pandas). 2 512 trading days over the 10-year window 2016-05-16 → 2026-05-12.
- Raw fetch is union-of-calendars (3 651 rows). Panel alignment clips to US business days; FOREX and crypto are forward-filled over holidays.

---

## 2. Data Layer

**EODHDAdapter** (`aiam.data.eodhd`)

- Wraps `eodhd.APIClient`. Key read from `EODHD_API_KEY` env var; `load_dotenv()` populates from `.env`.
- Fetches full OHLCV per ticker. Extracts `adjusted_close`.
- Cache: `data/cache/prices_30.parquet` — wide DataFrame, index=date, cols=tickers.
- Cache read-through: if parquet exists and covers requested range, skip network call.

**FREDAdapter** (`aiam.data.fred`)

- Wraps `fredapi.Fred`. Key read from `FRED_API_KEY` env var; same `load_dotenv()` pattern.
- Fetches macro series (e.g. CPIAUCSL). Cache: `data/cache/<SERIES_ID>.parquet`.

**Invariants**

- No secrets in code or git. `data/cache/` is gitignored.
- Both adapters expose a single `.fetch(start, end) -> pd.DataFrame` method.

---

## 3. Panel & Strategy ABC

**Panel** (`aiam.data.panel.Panel`)

- Immutable container: `dict[str, pd.DataFrame]` keyed by data kind (`"prices"`, `"macro"`, …).
- All DataFrames share the same DatetimeIndex (US business days, no gaps after alignment).
- One public method: `panel.slice(asof, kind, lookback, freq, fill="ffill") -> pd.DataFrame`
  - `asof`: last date visible to a strategy (enforces point-in-time; no lookahead).
  - `lookback`: integer number of `freq` periods to include.
  - `freq`: pandas offset string (`"B"`, `"W"`, `"ME"`, …).
  - `fill`: forward-fill policy applied inside the slice only, default `"ffill"`.
  - Returns a **copy** — mutations on the slice never affect Panel state.

**Strategy ABC** (`aiam.strategies.base.Strategy`)

```
class Strategy(ABC):
    def fit(self, panel: Panel, train_until: pd.Timestamp) -> None: ...  # no-op default
    @abstractmethod
    def predict_weights(self, panel: Panel, asof: pd.Timestamp) -> pd.Series: ...
```

- `fit` is optional; stateless strategies skip it.
- `predict_weights` receives the full Panel and `asof`; it calls `panel.slice(...)` itself.
- **Panel is never stashed on `self`**. Estimators (covariance, optimizer, …) are constructor `Callable` arguments, not base-class concerns.
- Weight series index = ticker symbols; must sum to 1.0 (checked by harness).

**PointInTimeStrategy** (`aiam.strategies.base.PointInTimeStrategy`)

- Thin subclass that enforces `asof <= train_until` guard in `predict_weights`.
- All strategies in `aiam.strategies.*` inherit from this.

---

## 4. Harness & Evaluation

**Walk-forward harness** (`aiam.harness.run_horse_race`)

```python
run_horse_race(
    panel: Panel,
    strategies: list[Strategy],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    refit_frequency: str = "QE",   # pandas offset: "ME", "QE", "YE"
) -> dict[str, pd.DataFrame]       # strategy_name -> returns DataFrame
```

- Train/test split is owned by the harness, never by Panel or Strategy.
- At each refit date: calls `strategy.fit(panel, asof=refit_date)`, then rolls forward predicting weights until the next refit.
- Rebalancing = close-of-day prices; no transaction costs in v0.

**Static baselines** — always included alongside the active strategy:

| Name | Method |
|---|---|
| EW | Equal weight, 1/N |
| GMV | Global minimum variance |
| MSR | Maximum Sharpe ratio |
| MDP | Maximum diversification |
| RP | Risk parity (equal risk contribution) |
| HRP | Hierarchical risk parity |

Baselines share the same refit cadence as the active strategy.

**Evaluation** (`aiam.evaluation.performance_stats`)

- Input: daily returns series or DataFrame. Output: `pd.Series` of stats per strategy.
- Canonical stats: annualised return, annualised vol, Sharpe, Sortino, max drawdown, Calmar, hit rate.
- **Sharpe fix**: risk-free rate subtracted in returns space before annualisation (`(r - rf).mean() / (r - rf).std() * sqrt(252)`), not via a scalar offset to the ratio.
- Used by harness to produce the final comparison table; also importable standalone.
