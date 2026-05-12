# paamcode Inventory

> Produced: 2026-05-12  
> Source: `git clone https://github.com/yhilpisch/paamcode.git /tmp/paamcode`  
> Book: *Python and AI for Asset Management* — Dr. Yves J. Hilpisch (The Python Quants)

---

## Top-level README (summary)

The repo accompanies Hilpisch's CPF Program course. It is organized in eight parts covering foundations, classical portfolio theory, risk management, ML/DL/RL, and LLM/agent workflows. Data lives in `data/pyaiam_eod.csv` (a static EOD CSV), and is loaded by every script from a `DATA_PATH` that falls back to a public URL. No live data client is implemented.

---

## Directory Trees

### `code/` (L=2)

```
/tmp/paamcode/code/
├── appx_b_numpy_pandas.py
├── appx_c_sklearn_cheatsheet.py
├── appx_d_pytorch_finance.py
├── appx_f_practical_tools.py
├── appx_g_repo_colab.py
├── ch01_asset_management_basics.py
├── ch02_math_stat_preliminaries.py
├── ch03_python_infrastructure.py
├── ch04_mean_variance.py
├── ch05_capm_factor_models.py
├── ch06_black_litterman.py
├── ch07_risk_measures.py
├── ch08_risk_decomposition.py
├── ch09_active_risk_management.py
├── ch10_data_engineering.py
├── ch11_performance_backtesting.py
├── ch12_ml_workflow.py
├── ch13_linear_glm.py
├── ch14_trees_ensembles.py
├── ch15_deep_learning.py
├── ch16_sequence_models.py
├── ch17_rl_foundations.py
├── ch18_rl_algorithms.py
├── ch19_unsupervised_representation.py
├── ch20_model_risk_explainability.py
├── ch21_tech_stack_deployment.py
├── ch22_llms_assistants.py
├── ch23_llms_agents_value_chain.py
└── README.md
```

### `notebooks/` (L=2)

```
/tmp/paamcode/notebooks/
├── appx_b_numpy_pandas.ipynb
├── appx_c_sklearn_cheatsheet.ipynb
├── appx_d_pytorch_finance.ipynb
├── appx_f_practical_tools.ipynb
├── appx_g_repo_colab.ipynb
├── ch01_asset_management_basics.ipynb
├── ch02_math_stat_preliminaries.ipynb
├── ch03_python_infrastructure.ipynb
├── ch04_mean_variance.ipynb
├── ch05_capm_factor_models.ipynb
├── ch06_black_litterman.ipynb
├── ch07_data_engineering.ipynb
├── ch08_performance_backtesting.ipynb
├── ch09_ml_workflow.ipynb
├── ch10_linear_glm.ipynb
├── ch11_trees_ensembles.ipynb
├── ch12_deep_learning.ipynb
├── ch13_sequence_models.ipynb
├── ch14_rl_foundations.ipynb
├── ch15_rl_algorithms.ipynb
├── ch16_unsupervised_representation.ipynb
├── ch17_model_risk_explainability.ipynb
├── ch18_tech_stack_deployment.ipynb
├── ch19_llms_assistants.ipynb
└── ch20_llms_agents_value_chain.ipynb
```

---

## Python File Inventory (`code/`)

### `ch01_asset_management_basics.py` — 160 lines

**Docstring:** `Python & AI in Asset Management / Chapter 1 · Asset Management Basics and Problem Landscape`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `IPython.display` (optional)

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `display_frame(frame: pd.DataFrame) -> None` | Display a DataFrame via IPython if available, else print with wide context. |
| `prepare_return_panels(price_frame: pd.DataFrame) -> dict` | Forward-fill prices, compute log and simple returns; return dict with keys `prices`, `log`, `simple`. |
| `describe_portfolio(log_returns, weights, risk_free=0.02)` | Return annualized return, vol, Sharpe, and max drawdown for an equal-weight or custom portfolio. |

**Module-level notable objects:** `ASSET_CLASSES` dict (8 assets), `mandate` dict (universe + constraints + risk_budget).

---

### `ch02_math_stat_preliminaries.py` — 186 lines

**Docstring:** `Chapter 2 · Mathematical and Statistical Preliminaries`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `argparse`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `write_tex_macros(path: Path, macros: dict[str, str]) -> None` | Write a LaTeX `\newcommand` file from a dict of macro names → values. |
| `toy_portfolio_mean_vol() -> tuple[float, float]` | Compute expected return and vol for a fixed 3-asset toy portfolio. |
| `parse_args() -> argparse.Namespace` | Parse `--write-stats-tex` CLI argument. |
| `compute_return_panels(price_frame) -> tuple[pd.DataFrame, pd.DataFrame]` | Return (log_returns, simple_returns) from a price DataFrame. |

---

### `ch03_python_infrastructure.py` — 108 lines

**Docstring:** `Chapter 3 · Python Infrastructure for Asset Management Research`

**Imports:** `platform`, `sys`, `pathlib.Path`, `matplotlib.pyplot`, `numpy`, `pandas`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `resample_prices(price_frame: pd.DataFrame, freq: str = "W") -> pd.DataFrame` | Resample a price frame to a lower frequency using last-observation-carry. |
| `rolling_features(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame` | Compute rolling vol and momentum, return stacked DataFrame. |
| `print_structure(root: Path, structure: dict) -> None` | Print a project directory skeleton without creating it. |

**Notable constants:** `STRUCTURE` dict defines canonical project folder layout.

---

### `ch04_mean_variance.py` — 114 lines

**Docstring:** `Chapter 4 · Mean–Variance Portfolio Theory`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `portfolio_stats(weights: np.ndarray, mean_vec: np.ndarray, cov_mat: np.ndarray) -> tuple[float, float]` | Return (annualized return, annualized vol) for a weight vector. |

**Notable:** Computes GMV and Max-Sharpe analytically via closed-form `cov_inv`; also demonstrates shrinkage of the mean vector.

---

### `ch05_capm_factor_models.py` — 77 lines

**Docstring:** `Chapter 5 · CAPM, Multifactor Models, and APT`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `statsmodels.api`, `pathlib.Path`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `regression_frame(asset: str, market: str = "SPY") -> pd.DataFrame` | Build a two-column (asset excess return, market excess return) DataFrame for OLS. |

---

### `ch06_black_litterman.py` — 124 lines

**Docstring:** `Chapter 6 · Black–Litterman and Bayesian Portfolio Construction`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `black_litterman(cov, tau, pi_vec, P, Q, omega)` | Compute BL posterior mean and posterior covariance; no return-type annotation. |

**Full verbatim (key function only):**

```python
def black_litterman(cov: np.ndarray, tau: float, pi_vec: np.ndarray, P: np.ndarray,
Q: np.ndarray, omega: np.ndarray):
    tau_cov = tau * cov
    inv_tau_cov = np.linalg.inv(tau_cov)
    middle = P.T @ np.linalg.inv(omega) @ P
    posterior_cov = np.linalg.inv(inv_tau_cov + middle)
    posterior_mean = posterior_cov @ (inv_tau_cov @ pi_vec + P.T @ np.linalg.inv(omega) @
        Q)
    return posterior_mean, posterior_cov
```

**View setup (module-level, verbatim):**

```python
RISK_AVERSION = 3.0
TAU = 0.05

# ...

pi = RISK_AVERSION * cov_matrix.values @ w_benchmark

P = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, 0],
    ]
)
Q = np.array([0.01, 0.005])
OMEGA = np.diag([0.0004, 0.0009])
```

---

### `ch07_risk_measures.py` — 71 lines

**Docstring:** `Chapter 7 · Coherent and Convex Risk Measures for Portfolios` — simulates daily portfolio returns from a multivariate normal model and computes VaR/ES.

**Imports:** `numpy`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `simulate_portfolio(mu, Sigma, w, n_sims=50_000, seed=7) -> tuple[np.ndarray, np.ndarray]` | Simulate asset and portfolio returns from a multivariate normal model. |
| `risk_measures_from_losses(port_ret, port_loss, alpha=0.99) -> tuple[float, float, float]` | Compute volatility, VaR, and ES from simulated losses. |
| `main() -> None` | Run the Chapter 7 case study and print summary numbers. |

---

### `ch08_risk_decomposition.py` — 74 lines

**Docstring:** `Chapter 8 · Risk Decomposition, Risk Parity, and Risk Budgeting`

**Imports:** `numpy`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `vol_and_rcov(Sigma: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]` | Compute portfolio volatility and component risk contributions. |
| `main() -> None` | Run the Chapter 8 case study comparing EW, MV, and risk-parity portfolios. |

---

### `ch09_active_risk_management.py` — 134 lines

**Docstring:** `Chapter 9 · Active Portfolio Risk Management Beyond Diversification`

**Imports:** `numpy`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `simulate_baseline(n_days=2500, seed=9) -> np.ndarray` | Simulate a simple baseline daily return series from a normal distribution. |
| `rolling_vol(ret: np.ndarray, window: int) -> np.ndarray` | Compute a simple rolling volatility estimate (manual loop). |
| `apply_vol_target(ret, vol_target=0.12, window=20, lam_min=0.0, lam_max=2.0) -> np.ndarray` | Apply a simple volatility-targeting rule (scale returns by vol_target / realized_vol). |
| `equity_curve(ret: np.ndarray, start: float = 1.0) -> np.ndarray` | Compute an equity curve from returns via cumprod. |
| `max_drawdown(equity: np.ndarray) -> float` | Compute maximum drawdown from an equity curve. |
| `apply_drawdown_overlay(ret, dd_threshold=0.10, scale_in_drawdown=0.5) -> np.ndarray` | Apply a simple drawdown-aware overlay — scale returns when drawdown exceeds threshold. |
| `summary_stats(ret: np.ndarray) -> dict[str, float]` | Compute annualized return, volatility, and max drawdown. |
| `main() -> None` | Run Chapter 9 case study; prints baseline, vol-targeted, and combined overlay stats. |

---

### `ch10_data_engineering.py` — 99 lines

**Docstring:** `Chapter 10 · Data Engineering and Cleaning for Financial Time Series`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `build_features(price_frame: pd.DataFrame, window: int = 21) -> pd.DataFrame` | Compute vol, momentum, and z-score features; return stacked panel (dropna). |
| `data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame` | Report pct missing before/after cleaning and whether latest row is non-null. |

---

### `ch11_performance_backtesting.py` — 95 lines

**Docstring:** `Chapter 11 · Performance Measurement, Backtesting, and Pitfalls`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `performance_stats(returns: pd.Series, risk_free: float = 0.02)` | Return annualized return, vol, Sharpe, and max drawdown as a pd.Series. |
| `generate_signal(data: pd.DataFrame) -> pd.Series` | Compute momentum signal (sign of 21-day pct change), shifted by 1 day. |
| `backtest(prices: pd.Series) -> pd.Series` | Apply momentum signal to a single asset and return strategy returns. |
| `plot_performance(returns: pd.Series, title: str)` | Plot equity curve and rolling 3M Sharpe side by side. |
| `drawdown_series(returns: pd.Series) -> pd.Series` | Compute drawdown time series from a return series. |

---

### `ch12_ml_workflow.py` — 103 lines

**Docstring:** `Chapter 12 · Machine Learning Foundations and Workflow`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.linear_model.Ridge`, `sklearn.model_selection.TimeSeriesSplit`, `sklearn.metrics.mean_squared_error`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `performance_stats(returns: pd.Series, risk_free: float = 0.02) -> pd.Series` | Same pattern as ch11; return annualized return, vol, Sharpe, max drawdown. |
| `rolling_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5)` | Generator yielding (X_train, X_val, y_train, y_val) via TimeSeriesSplit. |
| `run_ridge_workflow(X: pd.DataFrame, y: pd.Series, alpha: float = 10.0)` | Fit Ridge via rolling splits; return (out-of-sample predictions, avg fold MSE). |

---

### `ch13_linear_glm.py` — 99 lines

**Docstring:** `Chapter 13 · Linear and Generalized Linear Models for Return Prediction`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.pipeline.Pipeline`, `sklearn.linear_model.{Ridge,Lasso,LogisticRegression}`, `sklearn.metrics.{mean_squared_error,roc_auc_score}`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `performance_stats(returns: pd.Series, risk_free: float = 0.02)` | Same pattern repeated; return annualized return, vol, Sharpe, max drawdown. |

---

### `ch14_trees_ensembles.py` — 101 lines

**Docstring:** `Chapter 14 · Tree-Based Methods and Ensembles`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.tree.DecisionTreeRegressor`, `sklearn.ensemble.{RandomForestRegressor,GradientBoostingRegressor}`, `sklearn.metrics.mean_squared_error`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `performance_stats(returns: pd.Series, risk_free: float = 0.02)` | Same pattern; return annualized return, vol, Sharpe, max drawdown. |

---

### `ch15_deep_learning.py` — 120 lines

**Docstring:** `Chapter 15 · Deep Learning for Cross-Sectional and Panel Data`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset,DataLoader}`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `MLP(nn.Module)` | `__init__(self, hidden=64)` — 1-in → hidden → ReLU → Dropout(0.2) → 1-out | `forward(self, x)` |
| `AutoEncoder(nn.Module)` | `__init__(self, latent=3)` — encoder: 1→8→latent; decoder: latent→8→1 | `forward(self, x) -> tuple(latent, recon)` |

**Functions:**

| Signature | Summary |
|---|---|
| `train(model, epochs=5)` | Training loop using module-level `train_loader`, `val_loader`, `optimizer`, `loss_fn`. |

---

### `ch16_sequence_models.py` — 113 lines

**Docstring:** `Chapter 16 · Sequence Models and Temporal Deep Learning`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset,DataLoader}`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `ReturnLSTM(nn.Module)` | `__init__(self, hidden=32)` — LSTM(1, hidden) + Linear(hidden, 1) | `forward(self, x)` — returns last hidden state projected |
| `SimpleTransformer(nn.Module)` | `__init__(self, d_model=32, nhead=4, num_layers=2)` — Linear(1,d_model) + TransformerEncoder + Linear(d_model,1) | `forward(self, x)` |

**Functions:**

| Signature | Summary |
|---|---|
| `train_seq(model, epochs=5)` | Training loop for sequence models using module-level loaders. |

---

### `ch17_rl_foundations.py` — 79 lines

**Docstring:** `Chapter 17 · Reinforcement Learning Foundations`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `dataclasses.dataclass`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `PortfolioEnv` (dataclass) | `prices: pd.DataFrame, cost: float = 0.0005` | `reset(self)` → initial state; `state(self)` → mean 20-day returns array; `step(self, action)` → `(next_state, reward, done)` |

---

### `ch18_rl_algorithms.py` — 148 lines

**Docstring:** `Chapter 18 · RL Algorithms for Asset Management`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `dataclasses.dataclass`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `SimpleEnv` (dataclass) | `prices: pd.Series, cost: float = 0.0005` | `reset(self)`, `state(self)` → `[momentum, position]`, `step(self, action)` → `(next_state, reward, done)` |

**Functions:**

| Signature | Summary |
|---|---|
| `discretize(state)` | Bin a continuous state tuple for tabular Q-learning. |
| `run_policy(policy_vals, episodes=20)` | Evaluate a Q-table policy; return average episode reward. |
| `softmax_policy(state)` | Linear softmax policy parameterized by module-level `theta`. |

---

### `ch19_unsupervised_representation.py` — 85 lines

**Docstring:** `Chapter 19 · Unsupervised Learning and Representation Learning`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.decomposition.PCA`, `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`

**Classes:** none  
**Functions:** none (all notebook-style top-level cells)

---

### `ch20_model_risk_explainability.py` — 88 lines

**Docstring:** `Chapter 20 · Model Risk Management and Explainability`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.inspection.permutation_importance`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `what_if(row: pd.Series, adjustments: dict) -> pd.Series` | Predict RF output for feature perturbations; returns a series of perturbed predictions. |

---

### `ch21_tech_stack_deployment.py` — 88 lines

**Docstring:** `Chapter 21 · Technology Stack and Deployment Patterns`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `datetime`, `logging`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `ingest(path: Path) -> pd.DataFrame` | Load CSV with Date index. |
| `engineer(df: pd.DataFrame) -> pd.DataFrame` | Compute rolling vol and momentum features. |
| `score(panel: pd.DataFrame) -> pd.DataFrame` | Placeholder scorer (groupby mean). |
| `daily_pipeline() -> None` | Chain ingest → engineer → score → write Parquet. |
| `send_alert(message: str) -> None` | Log a WARNING-level alert. |

---

### `ch22_llms_assistants.py` — 82 lines

**Docstring:** `Chapter 22 · LLMs as Research and Coding Assistants`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `textwrap`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `MockLLMClient` | `__init__(self)` | `complete(self, prompt: str) -> str` — returns mock stub string |

**Functions:**

| Signature | Summary |
|---|---|
| `render_prompt(name: str, **kwargs) -> str` | Format a named prompt template from `PROMPTS` dict with keyword substitution. |

---

### `ch23_llms_agents_value_chain.py` — 90 lines

**Docstring:** `Chapter 23 · LLMs and Agents in the Asset Management Value Chain`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `pprint`

**Classes:**

| Class | `__init__` signature | Methods |
|---|---|---|
| `SimpleAgent` | `__init__(self)` — sets `self.history = []` | `run(self, task: str, **kwargs) -> dict` — dispatches on `task` string; only handles `'diagnose_strategy'` |

**Functions:**

| Signature | Summary |
|---|---|
| `performance_stats(returns: pd.Series, risk_free: float = 0.02) -> pd.Series` | Same pattern; annualized return, vol, Sharpe, max drawdown. |
| `load_prices() -> pd.DataFrame` | Load prices from DATA_PATH. |
| `simple_strategy(asset: str = 'AAPL') -> pd.Series` | 20-day momentum signal strategy; return strategy returns. |
| `strategy_report(asset: str = 'AAPL') -> dict` | Run simple_strategy and return performance_stats as a dict. |

---

### `appx_b_numpy_pandas.py` — 69 lines

**Docstring:** `Appendix B · NumPy and pandas Reference`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`

**Classes/Functions:** none (all top-level cells showing NumPy/pandas recipes)

---

### `appx_c_sklearn_cheatsheet.py` — 73 lines

**Docstring:** `Appendix C · scikit-learn Cheat Sheet`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.linear_model.{Ridge,LogisticRegression}`, `sklearn.pipeline.Pipeline`, `sklearn.model_selection.{TimeSeriesSplit,GridSearchCV}`, `sklearn.metrics.{mean_squared_error,roc_auc_score}`

**Classes/Functions:** none (cheat-sheet cells)

---

### `appx_d_pytorch_finance.py` — 53 lines

**Docstring:** `Appendix D · PyTorch for Finance`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset,DataLoader}`

**Classes/Functions:** none (minimal training loop cells)

---

### `appx_f_practical_tools.py` — 57 lines

**Docstring:** `Appendix F · Practical Tools`

**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `logging`, `time`

**Classes:** none

**Functions:**

| Signature | Summary |
|---|---|
| `make_logger(name: str = 'pyaiam') -> logging.Logger` | Create and return a logging.Logger with basicConfig. |
| `timed(func)` | Decorator; times the wrapped function and logs elapsed time. |
| `demo_sleep()` | Demo target for the `@timed` decorator. |

---

### `appx_g_repo_colab.py` — 28 lines

**Docstring:** `Appendix G · Repository and Colab Guide`

**Imports:** `os`, `sys`, `pathlib.Path`

**Classes/Functions:** none (env-check cells)

---

## Notebooks Inventory

| Filename | Subtitle (## heading) |
|---|---|
| `ch01_asset_management_basics.ipynb` | Chapter 1 · Asset Management Basics and Problem Landscape |
| `ch02_math_stat_preliminaries.ipynb` | Chapter 2 · Mathematical and Statistical Preliminaries |
| `ch03_python_infrastructure.ipynb` | Chapter 3 · Python Infrastructure for Asset Management Research |
| `ch04_mean_variance.ipynb` | Chapter 4 · Mean–Variance Portfolio Theory |
| `ch05_capm_factor_models.ipynb` | Chapter 5 · CAPM, Multifactor Models, and APT |
| `ch06_black_litterman.ipynb` | Chapter 6 · Black–Litterman and Bayesian Portfolio Construction |
| `ch07_data_engineering.ipynb` | Chapter 7 · Data Engineering and Cleaning for Financial Time Series |
| `ch08_performance_backtesting.ipynb` | Chapter 8 · Performance Measurement, Backtesting, and Pitfalls |
| `ch09_ml_workflow.ipynb` | Chapter 9 · Machine Learning Foundations and Workflow |
| `ch10_linear_glm.ipynb` | Chapter 10 · Linear and Generalized Linear Models for Return Prediction |
| `ch11_trees_ensembles.ipynb` | Chapter 11 · Tree-Based Methods and Ensembles |
| `ch12_deep_learning.ipynb` | Chapter 12 · Deep Learning for Cross-Sectional and Panel Data |
| `ch13_sequence_models.ipynb` | Chapter 13 · Sequence Models and Temporal Deep Learning |
| `ch14_rl_foundations.ipynb` | Chapter 14 · Reinforcement Learning Foundations |
| `ch15_rl_algorithms.ipynb` | Chapter 15 · RL Algorithms for Asset Management |
| `ch16_unsupervised_representation.ipynb` | Chapter 16 · Unsupervised Learning and Representation |
| `ch17_model_risk_explainability.ipynb` | Chapter 17 · Model Risk Management and Explainability |
| `ch18_tech_stack_deployment.ipynb` | Chapter 18 · Technology Stack and Deployment Patterns |
| `ch19_llms_assistants.ipynb` | Chapter 19 · LLMs as Research and Coding Assistants |
| `ch20_llms_agents_value_chain.ipynb` | Chapter 20 · LLMs and Agents in the Asset Management Value Chain |
| `appx_b_numpy_pandas.ipynb` | Appendix B · NumPy and pandas Reference |
| `appx_c_sklearn_cheatsheet.ipynb` | Appendix C · scikit-learn Cheat Sheet |
| `appx_d_pytorch_finance.ipynb` | Appendix D · PyTorch for Finance |
| `appx_f_practical_tools.ipynb` | Appendix F · Practical Tools |
| `appx_g_repo_colab.ipynb` | Appendix G · Repository and Colab Guide |

---

## Synthesis

### a. Strategy class hierarchy or ABC?

**No.** Hilpisch defines no `Strategy` ABC, no abstract base class, and no polymorphic strategy hierarchy anywhere in the repo. All strategy logic is implemented as:

1. Standalone module-level functions (e.g., `generate_signal`, `backtest`, `simple_strategy`).
2. Script-level control flow that calls those functions directly.
3. For RL: environment dataclasses (`PortfolioEnv`, `SimpleEnv`) with `reset`/`state`/`step` interfaces, but no strategy ABC wrapping them.

There is no `run(data) -> pd.Series` interface, no `Strategy` superclass, and nothing to inherit from. Each chapter is a self-contained script.

---

### b. Panel-equivalent or Universe class?

**No.** Hilpisch uses no `Panel`, `Universe`, or `DataPanel` class. The asset universe is represented as:

- A plain Python list (`base_universe`, `assets`) of ticker strings.
- A dict (`ASSET_CLASSES`) mapping tickers to asset class labels.
- A dict (`mandate`) bundling universe + constraints + risk_budget.

Price data is a raw `pd.DataFrame` indexed by Date with ticker columns. There is no encapsulating object that carries metadata, constraints, and return panels together.

---

### c. EODHD client pattern?

**Not present.** Hilpisch does not implement an EODHD API client anywhere in the repo. Data access follows a single pattern repeated verbatim in every script:

```python
DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
```

The CSV (`pyaiam_eod.csv`) is a pre-downloaded static file containing adjusted EOD prices for 8 assets (AAPL, NVDA, JPM, SPY, GLD, TLT, EURUSD, BTC-USD). There is no live fetch, no API key handling, and no EODHD SDK usage. The fallback URL is Hilpisch's own server, not EODHD directly.

---

### d. Black-Litterman: does it appear, and are the view generators deterministic/portable?

**Yes, BL appears** in `ch06_black_litterman.py`. The core formula is clean and portable (see verbatim above in the ch06 entry).

**The view generators are hardcoded and not deterministic in a reusable sense.** Views are set at module scope as fixed literals:

```python
P = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, 0],
    ]
)
Q = np.array([0.01, 0.005])
OMEGA = np.diag([0.0004, 0.0009])
```

There is no `ViewGenerator` class, no function that derives views from signals or market data, and no confidence scaling logic. The two views encode:
- View 1: AAPL has an absolute expected excess return of +1 %/year.
- View 2: NVDA outperforms GLD by +0.5 %/year.
- Omega is a fixed diagonal confidence matrix.

**Verdict for session-1 BL:** The `black_litterman()` function itself is directly portable — it is a standard Idzorek/He-Litterman formula with clean NumPy inputs. The view scaffolding (P, Q, Ω) must be rebuilt from scratch with real signal-driven generators; Hilpisch's version offers no reusable mechanism for that.

---

### e. What in `code/` is worth porting wholesale into next-gen-aiam?

**High-value ports (low modification needed):**

| File | Item | Why |
|---|---|---|
| `ch07_risk_measures.py` | `simulate_portfolio` + `risk_measures_from_losses` | Clean, tested, self-contained VaR/ES from simulated losses. Direct drop-in for our risk evaluation module. |
| `ch08_risk_decomposition.py` | `vol_and_rcov` | Marginal risk contributions in 5 lines of NumPy. Needed for risk-parity and risk-budgeting strategies. |
| `ch09_active_risk_management.py` | `apply_vol_target` + `apply_drawdown_overlay` + `summary_stats` | Vol-targeting and drawdown-overlay overlays; clean, no external deps. Worth porting as overlay utilities. |
| `ch11_performance_backtesting.py` | `performance_stats` | This function is copy-pasted across 7 files. Port once as the canonical `aiam.evaluation.performance_stats`. |
| `ch06_black_litterman.py` | `black_litterman()` | The formula is correct and clean. Port it; build the view-generator wrapper on top. |
| `ch10_data_engineering.py` | `build_features` + `data_quality_report` | Feature engineering and data quality utilities are directly useful in `aiam.data`. |

**Low-value / skip:**

- `ch01–ch03`: mostly data loading and plotting boilerplate. Our `aiam.data` module will subsume this with a proper EODHD client.
- `ch15–ch16` (PyTorch models): `MLP`, `AutoEncoder`, `ReturnLSTM`, `SimpleTransformer` are minimal teaching illustrations. They will be superseded by our DL strategy implementations.
- `ch17–ch18` (RL environments): `PortfolioEnv` / `SimpleEnv` are toy environments. Our RL strategy will need a properly designed environment; these are starting points for design reference only.
- `ch22–ch23` (LLM scaffolding): `MockLLMClient`, `SimpleAgent`, `render_prompt` are stubs. Not portable in their current form.
- All `appx_*`: reference material only; nothing to port.

**`performance_stats` duplication note:** The function appears verbatim in at least 7 files (ch11, ch12, ch13, ch14, ch23, and implicitly ch01). This is the single most important consolidation opportunity — define it once in `aiam.evaluation` and import everywhere.
