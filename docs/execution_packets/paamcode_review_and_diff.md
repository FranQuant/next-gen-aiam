# paamcode Review and Comparison to paam_lab

> Produced: 2026-05-12  
> paamcode: `git clone https://github.com/yhilpisch/paamcode.git /tmp/paamcode`  
> paam_lab: `~/Projects/ai_asset_management_lab/src/paam_lab/`  
> Book: *Python and AI for Asset Management* — Dr. Yves J. Hilpisch

---

## 1. READMEs (verbatim)

### `/tmp/paamcode/README.md`

```
# Python and AI for Asset Management — Code & Notebooks

This repository contains the Jupyter notebooks and Python scripts that accompany the
*Python and AI for Asset Management* class and book in the CPF Program. The material is
organised to mirror the structure of the main text:

- Part I — Foundations of Asset Management and Quantitative Methods  
- Part II — Classical Asset Management Theory and Practice  
- Part III — Risk and Active Risk Management  
- Part IV — Machine Learning Foundations and Linear Models  
- Part V — Tree-Based Methods, Deep Models, and Sequence Models  
- Part VI — Unsupervised Learning, LLMs, and Assistants  
- Part VII — From Research to Production: Risk, Governance, and Infrastructure  
- Part VIII — LLMs, Agents, and Modern AI in Asset Management

## Structure

- `notebooks/` — chapter and appendix notebooks (`chXX_*.ipynb`, `appx_*.ipynb`) that
  bring together concepts, code, and plots.
- `code/` — standalone Python modules and helper scripts used for figures, simulations,
  and risk and performance calculations.
- `data/` — source CSV datasets required by the notebooks and scripts (for example,
  `pyaiam_eod.csv`); generated artifacts such as Parquet files are intentionally
  excluded from sync.

## Usage

The notebooks are designed to run in a standard scientific Python environment with:

- Python 3.11+  
- `numpy`, `pandas`, `matplotlib`  
- `scipy`, `scikit-learn`, `statsmodels` (selected examples)  
- `torch`, `torchvision` (deep learning and sequence models)  
- `gymnasium` or similar RL environments where used
```

### `/tmp/paamcode/code/README.md` (summary — full doc lists all 28 .py files by chapter)

Key entries:

- `ch06_black_litterman.py` — Black–Litterman posterior calculations, including views, confidences, and implied equilibrium returns.
- `ch09_active_risk_management.py` — active risk, information ratio, and tracking error calculations, plus simple overlay strategies.
- `ch10_data_engineering.py` — feature engineering utilities for end-of-day data, including rolling statistics and factor panels.
- `ch11_performance_backtesting.py` — vectorised backtest loops, performance metrics, and equity curve generation.

### `/tmp/paamcode/notebooks/README.md` (summary)

25 notebooks (ch01–ch20 + appx_b–appx_g), each mirroring a book chapter. Notebooks integrate narrative, code, and plots.

### `/tmp/paamcode/requirements.txt` (verbatim)

```
# Core scientific stack (used throughout code/ and notebooks/)
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13

# Parquet engine for pandas.to_parquet / read_parquet
pyarrow>=14.0

# ML + statistics (used in code/ and referenced in the book)
scikit-learn>=1.3
statsmodels>=0.14
torch>=2.1

# Interactive + notebook tooling
ipython>=8.0
jupyterlab>=4.0
ipykernel>=6.0
nbformat>=5.9
nbclient>=0.10

# LaTeX minted
Pygments>=2.16
```

---

## 2. Directory Trees

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
├── ch20_llms_agents_value_chain.ipynb
└── README.md
```

---

## 3. Python File Inventory (`code/`)

### `ch01_asset_management_basics.py` — 160 lines

**Docstring:** `Chapter 1 · Asset Management Basics and Problem Landscape`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `IPython.display` (optional)  
**Classes:** none

| Function | Summary |
|---|---|
| `display_frame(frame)` | Display DataFrame via IPython if available, else print wide. |
| `prepare_return_panels(price_frame) -> dict` | Forward-fill, compute log and simple returns; return dict with keys `prices`, `log`, `simple`. |
| `describe_portfolio(log_returns, weights, risk_free=0.02)` | Annualized return, vol, Sharpe, max drawdown for a portfolio. |

---

### `ch02_math_stat_preliminaries.py` — 186 lines

**Docstring:** `Chapter 2 · Mathematical and Statistical Preliminaries`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `argparse`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `write_tex_macros(path, macros)` | Write LaTeX `\newcommand` macros file. |
| `toy_portfolio_mean_vol() -> tuple[float, float]` | Fixed 3-asset toy portfolio return and vol. |
| `parse_args() -> argparse.Namespace` | Parse `--write-stats-tex` CLI arg. |
| `compute_return_panels(price_frame) -> tuple[DataFrame, DataFrame]` | Return (log_returns, simple_returns). |

---

### `ch03_python_infrastructure.py` — 108 lines

**Docstring:** `Chapter 3 · Python Infrastructure for Asset Management Research`  
**Imports:** `platform`, `sys`, `pathlib.Path`, `matplotlib.pyplot`, `numpy`, `pandas`  
**Classes:** none

| Function | Summary |
|---|---|
| `resample_prices(price_frame, freq="W") -> DataFrame` | Resample to lower frequency using last observation. |
| `rolling_features(log_returns, window=21) -> DataFrame` | Rolling vol and momentum stacked DataFrame. |
| `print_structure(root, structure) -> None` | Print project directory skeleton. |

---

### `ch04_mean_variance.py` — 114 lines

**Docstring:** `Chapter 4 · Mean–Variance Portfolio Theory`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `portfolio_stats(weights, mean_vec, cov_mat) -> tuple[float, float]` | Return (annualized return, annualized vol). |

Computes GMV and Max-Sharpe analytically via closed-form `cov_inv`; demonstrates mean shrinkage.

---

### `ch05_capm_factor_models.py` — 77 lines

**Docstring:** `Chapter 5 · CAPM, Multifactor Models, and APT`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `statsmodels.api`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `regression_frame(asset, market="SPY") -> DataFrame` | Two-column excess-return DataFrame for OLS. |

---

### `ch06_black_litterman.py` — 124 lines

**Docstring:** `Chapter 6 · Black–Litterman and Bayesian Portfolio Construction`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `black_litterman(cov, tau, pi_vec, P, Q, omega)` | BL posterior mean and covariance. |

**Key function verbatim:**
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

**Module-level view setup (verbatim):**
```python
RISK_AVERSION = 3.0
TAU = 0.05
pi = RISK_AVERSION * cov_matrix.values @ w_benchmark
P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, -1, 0]])
Q = np.array([0.01, 0.005])
OMEGA = np.diag([0.0004, 0.0009])
```

---

### `ch07_risk_measures.py` — 71 lines

**Docstring:** `Chapter 7 · Coherent and Convex Risk Measures for Portfolios`  
**Imports:** `numpy`  
**Classes:** none

| Function | Summary |
|---|---|
| `simulate_portfolio(mu, Sigma, w, n_sims=50_000, seed=7) -> tuple[ndarray, ndarray]` | Simulate asset and portfolio returns from multivariate normal. |
| `risk_measures_from_losses(port_ret, port_loss, alpha=0.99) -> tuple[float, float, float]` | Compute volatility, VaR, and ES from simulated losses. |
| `main() -> None` | Run Chapter 7 case study and print summary. |

---

### `ch08_risk_decomposition.py` — 74 lines

**Docstring:** `Chapter 8 · Risk Decomposition, Risk Parity, and Risk Budgeting`  
**Imports:** `numpy`  
**Classes:** none

| Function | Summary |
|---|---|
| `vol_and_rcov(Sigma, w) -> tuple[float, ndarray]` | Portfolio volatility and component risk contributions. |
| `main() -> None` | Compare EW, MV, and risk-parity portfolios. |

---

### `ch09_active_risk_management.py` — 134 lines

**Docstring:** `Chapter 9 · Active Portfolio Risk Management Beyond Diversification`  
**Imports:** `numpy`  
**Classes:** none

| Function | Summary |
|---|---|
| `simulate_baseline(n_days=2500, seed=9) -> ndarray` | Simulate baseline daily returns from a normal distribution. |
| `rolling_vol(ret, window) -> ndarray` | Manual-loop rolling volatility estimate. |
| `apply_vol_target(ret, vol_target=0.12, window=20, lam_min=0.0, lam_max=2.0) -> ndarray` | Scale returns by `vol_target / realized_vol`. |
| `equity_curve(ret, start=1.0) -> ndarray` | Cumulative return via cumprod. |
| `max_drawdown(equity) -> float` | Maximum drawdown from an equity curve. |
| `apply_drawdown_overlay(ret, dd_threshold=0.10, scale_in_drawdown=0.5) -> ndarray` | Scale returns when drawdown exceeds threshold. |
| `summary_stats(ret) -> dict[str, float]` | Annualized return, volatility, max drawdown. |
| `main() -> None` | Run Chapter 9 case study. |

---

### `ch10_data_engineering.py` — 99 lines

**Docstring:** `Chapter 10 · Data Engineering and Cleaning for Financial Time Series`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `build_features(price_frame, window=21) -> DataFrame` | Compute vol, momentum, z-score; return stacked panel. |
| `data_quality_report(raw, cleaned) -> DataFrame` | pct_missing before/after and data-currency flag. |

---

### `ch11_performance_backtesting.py` — 95 lines

**Docstring:** `Chapter 11 · Performance Measurement, Backtesting, and Pitfalls`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`  
**Classes:** none

| Function | Summary |
|---|---|
| `performance_stats(returns, risk_free=0.02)` | Annualized return, vol, Sharpe, max drawdown as pd.Series. |
| `generate_signal(data) -> Series` | Sign of 21-day pct change, shifted 1 day. |
| `backtest(prices) -> Series` | Apply momentum signal to a single asset. |
| `plot_performance(returns, title)` | Equity curve and rolling Sharpe. |
| `drawdown_series(returns) -> Series` | Drawdown time series. |

---

### `ch12_ml_workflow.py` — 103 lines

**Docstring:** `Chapter 12 · Machine Learning Foundations and Workflow`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.linear_model.Ridge`, `sklearn.model_selection.TimeSeriesSplit`, `sklearn.metrics.mean_squared_error`  
**Classes:** none

| Function | Summary |
|---|---|
| `performance_stats(returns, risk_free=0.02) -> Series` | Annualized return, vol, Sharpe, max drawdown. |
| `rolling_split(X, y, n_splits=5)` | Generator: (X_train, X_val, y_train, y_val) via TimeSeriesSplit. |
| `run_ridge_workflow(X, y, alpha=10.0)` | Fit Ridge via rolling splits; return (OOS predictions, avg fold MSE). |

---

### `ch13_linear_glm.py` — 99 lines

**Docstring:** `Chapter 13 · Linear and Generalized Linear Models for Return Prediction`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.pipeline.Pipeline`, `sklearn.linear_model.{Ridge, Lasso, LogisticRegression}`, `sklearn.metrics.{mean_squared_error, roc_auc_score}`  
**Classes:** none

| Function | Summary |
|---|---|
| `performance_stats(returns, risk_free=0.02)` | Same pattern as ch11. |

---

### `ch14_trees_ensembles.py` — 101 lines

**Docstring:** `Chapter 14 · Tree-Based Methods and Ensembles`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.tree.DecisionTreeRegressor`, `sklearn.ensemble.{RandomForestRegressor, GradientBoostingRegressor}`, `sklearn.metrics.mean_squared_error`  
**Classes:** none

| Function | Summary |
|---|---|
| `performance_stats(returns, risk_free=0.02)` | Same pattern as ch11. |

---

### `ch15_deep_learning.py` — 120 lines

**Docstring:** `Chapter 15 · Deep Learning for Cross-Sectional and Panel Data`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset, DataLoader}`

| Class | `__init__` | Methods |
|---|---|---|
| `MLP(nn.Module)` | `(self, hidden=64)` — 1→hidden→ReLU→Dropout(0.2)→1 | `forward(self, x)` |
| `AutoEncoder(nn.Module)` | `(self, latent=3)` — encoder: 1→8→latent; decoder: latent→8→1 | `forward(self, x) -> (latent, recon)` |

| Function | Summary |
|---|---|
| `train(model, epochs=5)` | Training loop using module-level loaders/optimizer. |

---

### `ch16_sequence_models.py` — 113 lines

**Docstring:** `Chapter 16 · Sequence Models and Temporal Deep Learning`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset, DataLoader}`

| Class | `__init__` | Methods |
|---|---|---|
| `ReturnLSTM(nn.Module)` | `(self, hidden=32)` — LSTM(1,hidden) + Linear(hidden,1) | `forward(self, x)` |
| `SimpleTransformer(nn.Module)` | `(self, d_model=32, nhead=4, num_layers=2)` — Linear(1,d_model)+TransformerEncoder+Linear | `forward(self, x)` |

| Function | Summary |
|---|---|
| `train_seq(model, epochs=5)` | Training loop for sequence models. |

---

### `ch17_rl_foundations.py` — 79 lines

**Docstring:** `Chapter 17 · Reinforcement Learning Foundations`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `dataclasses.dataclass`

| Class | `__init__` | Methods |
|---|---|---|
| `PortfolioEnv` (dataclass) | `prices: pd.DataFrame, cost: float = 0.0005` | `reset(self)`, `state(self) -> ndarray`, `step(self, action) -> (state, reward, done)` |

---

### `ch18_rl_algorithms.py` — 148 lines

**Docstring:** `Chapter 18 · RL Algorithms for Asset Management`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `dataclasses.dataclass`

| Class | `__init__` | Methods |
|---|---|---|
| `SimpleEnv` (dataclass) | `prices: pd.Series, cost: float = 0.0005` | `reset(self)`, `state(self) -> ndarray[momentum, position]`, `step(self, action) -> (state, reward, done)` |

| Function | Summary |
|---|---|
| `discretize(state)` | Bin continuous state for tabular Q-learning. |
| `run_policy(policy_vals, episodes=20)` | Evaluate a Q-table policy; return avg episode reward. |
| `softmax_policy(state)` | Linear softmax policy parameterized by module-level `theta`. |

---

### `ch19_unsupervised_representation.py` — 85 lines

**Docstring:** `Chapter 19 · Unsupervised Learning and Representation Learning`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.decomposition.PCA`, `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score`  
**Classes/Functions:** none (notebook-style cells — KMeans + PCA on asset feature panel)

---

### `ch20_model_risk_explainability.py` — 88 lines

**Docstring:** `Chapter 20 · Model Risk Management and Explainability`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.inspection.permutation_importance`  
**Classes:** none

| Function | Summary |
|---|---|
| `what_if(row, adjustments) -> Series` | Predict RF output for feature perturbations. |

---

### `ch21_tech_stack_deployment.py` — 88 lines

**Docstring:** `Chapter 21 · Technology Stack and Deployment Patterns`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `datetime`, `logging`  
**Classes:** none

| Function | Summary |
|---|---|
| `ingest(path) -> DataFrame` | Load CSV with Date index. |
| `engineer(df) -> DataFrame` | Rolling vol + momentum features. |
| `score(panel) -> DataFrame` | Placeholder scorer (groupby mean). |
| `daily_pipeline() -> None` | Chain ingest → engineer → score → write Parquet. |
| `send_alert(message) -> None` | Log a WARNING-level alert. |

---

### `ch22_llms_assistants.py` — 82 lines

**Docstring:** `Chapter 22 · LLMs as Research and Coding Assistants`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `textwrap`

| Class | `__init__` | Methods |
|---|---|---|
| `MockLLMClient` | `(self)` | `complete(self, prompt: str) -> str` — returns mock stub |

| Function | Summary |
|---|---|
| `render_prompt(name, **kwargs) -> str` | Format named prompt template from `PROMPTS` dict. |

---

### `ch23_llms_agents_value_chain.py` — 90 lines

**Docstring:** `Chapter 23 · LLMs and Agents in the Asset Management Value Chain`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `json`, `pprint`

| Class | `__init__` | Methods |
|---|---|---|
| `SimpleAgent` | `(self)` sets `self.history = []` | `run(self, task, **kwargs) -> dict` — dispatches on `task` string |

| Function | Summary |
|---|---|
| `performance_stats(returns, risk_free=0.02) -> Series` | Same pattern as ch11. |
| `load_prices() -> DataFrame` | Load from DATA_PATH. |
| `simple_strategy(asset="AAPL") -> Series` | 20-day momentum strategy returns. |
| `strategy_report(asset="AAPL") -> dict` | Run simple_strategy and return performance_stats as dict. |

---

### `appx_b_numpy_pandas.py` — 69 lines

**Docstring:** `Appendix B · NumPy and pandas Reference`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`  
**Classes/Functions:** none (reference cells)

---

### `appx_c_sklearn_cheatsheet.py` — 73 lines

**Docstring:** `Appendix C · scikit-learn Cheat Sheet`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sklearn.linear_model.{Ridge,LogisticRegression}`, `sklearn.pipeline.Pipeline`, `sklearn.model_selection.{TimeSeriesSplit,GridSearchCV}`, `sklearn.metrics.{mean_squared_error,roc_auc_score}`  
**Classes/Functions:** none

---

### `appx_d_pytorch_finance.py` — 53 lines

**Docstring:** `Appendix D · PyTorch for Finance`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `torch`, `torch.nn`, `torch.utils.data.{TensorDataset,DataLoader}`  
**Classes/Functions:** none

---

### `appx_f_practical_tools.py` — 57 lines

**Docstring:** `Appendix F · Practical Tools`  
**Imports:** `numpy`, `pandas`, `matplotlib.pyplot`, `pathlib.Path`, `logging`, `time`  
**Classes:** none

| Function | Summary |
|---|---|
| `make_logger(name="pyaiam") -> logging.Logger` | Create and return a Logger with basicConfig. |
| `timed(func)` | Decorator: time the wrapped function and log elapsed. |
| `demo_sleep()` | Demo target for the `@timed` decorator. |

---

### `appx_g_repo_colab.py` — 28 lines

**Docstring:** `Appendix G · Repository and Colab Guide`  
**Imports:** `os`, `sys`, `pathlib.Path`  
**Classes/Functions:** none (env-check cells)

---

## 4. Notebooks Inventory

| Filename | Subtitle (## heading in first markdown cell) |
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

## 5. Comparison: paam_lab vs paamcode

For each file in `~/Projects/ai_asset_management_lab/src/paam_lab/`:

---

### `paam_lab/construction.py`

**Closest paamcode analog:** `ch04_mean_variance.py` (both implement portfolio construction)

**Diff `diff -u ch04_mean_variance.py construction.py | head -200`:**

The diff replaces 114 lines of notebook-style MV script with 420 lines of library code. Every function is rewritten from scratch. Selected diff context:

```diff
-def portfolio_stats(weights: np.ndarray, mean_vec: np.ndarray, cov_mat: np.ndarray) -> tuple[float, float]:
-    port_ret = weights @ mean_vec
-    port_vol = np.sqrt(weights @ cov_mat @ weights)
-    return port_ret, port_vol
+def port_vol(w, Sigma) -> float:
+    """Portfolio annualised volatility. Canonical form from 08g."""
+    w = _to_array(w)
+    return float(np.sqrt(max(w @ _to_array(Sigma) @ w, 0.0)))
+
+def gmv_weights(Sigma) -> np.ndarray:
+    """Long-only global minimum variance via SLSQP. Canonical form from 08g."""
+    ...
+    res = minimize(lambda w: float(w @ S @ w), ..., method='SLSQP', ...)
```

paamcode uses closed-form `cov_inv` GMV (no long-only constraint). paam_lab uses SLSQP throughout. paam_lab adds: `msr_weights`, `emv_weights`, `mdp_weights`, `rp_weights`, `solve_risk_budget`, `bl_posterior_returns`, `bl_weights`, `hrp_weights`, `harp_weights`, `hap_weights`, `hcp_weights`, `corr_dist_matrix` — none of which exist in paamcode.

**Verdict: paam_lab original.** The concept (portfolio construction) is shared; zero code is shared.

---

### `paam_lab/covariance.py`

**Closest paamcode analog:** `ch02_math_stat_preliminaries.py` (both compute covariance matrices)

**Diff summary:** paamcode is 186 lines of statistical visualization; covariance.py is 45 lines of clean library code with `empirical_cov`, `ledoit_wolf_cov`, `oas_cov`, `rmf_cov`. Zero shared code. paamcode has no shrinkage or RMF implementation.

**Verdict: paam_lab original.** `empirical_cov` is the only conceptual overlap (both call `.cov()`).

---

### `paam_lab/data_quality.py`

**Closest paamcode analog:** `ch10_data_engineering.py` (both do data cleaning)

**Diff summary:** paamcode has 99 lines with `build_features` and a 5-line `data_quality_report`. paam_lab has 325 lines with 9 functions. Selected diff context:

```diff
-def data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
-    return pd.DataFrame(
-        {
-            "pct_missing_before": raw.isna().mean(),
-            "pct_missing_after": cleaned.isna().mean(),
-            "up_to_date": cleaned.iloc[-1].notna(),
-        }
-    )
+def data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
+    """Consolidated before/after missingness and data-currency report indexed by asset."""
+    n_raw = max(len(raw), 1)
+    n_clean = max(len(cleaned), 1)
+    raw_aligned = raw.reindex(columns=cleaned.columns)
+    return pd.DataFrame(...)
+
+def validate_panel_structure(panel, name="panel") -> pd.DataFrame: ...
+def summarize_missingness(panel) -> pd.DataFrame: ...
+def find_duplicate_dates(panel) -> pd.DataFrame: ...
+def find_stale_segments(panel, min_length=3) -> pd.DataFrame: ...
+def flag_extreme_moves(returns, threshold=0.10) -> pd.DataFrame: ...
+def flag_split_like_moves(returns, threshold=0.40) -> pd.DataFrame: ...
+def clean_price_panel(panel) -> tuple[DataFrame, DataFrame]: ...
+def plot_flagged_points(prices, flags, asset, ...) -> Figure: ...
```

**Verdict: substantial rewrite.** `data_quality_report` shares the same concept (3 columns); the remaining 8 functions are paam_lab original.

---

### `paam_lab/features.py`

**Closest paamcode analog:** `ch10_data_engineering.py` (both build feature panels)

**Diff summary:**

```diff
-def build_features(price_frame: pd.DataFrame, window: int = 21) -> pd.DataFrame:
-    log_ret = np.log(price_frame / price_frame.shift(1))
-    vol = log_ret.rolling(window).std() * np.sqrt(252)
-    momentum = price_frame.pct_change(window)
-    ...
+def build_feature_panel(clean_prices: pd.DataFrame) -> pd.DataFrame:
+    log_returns = np.log(clean_prices / clean_prices.shift(1))
+    lagged = log_returns.shift(1)           # ← explicit look-ahead guard
+    features = pd.concat({
+        "log_ret_1d": lagged,
+        "mom_5d": lagged.rolling(5).sum(),  # ← log-sum, not pct_change
+        "mom_20d": lagged.rolling(20).sum(),
+        "vol_20d": lagged.rolling(20).std() * sqrt(252),
+        "z_20d": ...,
+    }, axis=1)
+    cs_rank = features["mom_20d"].rank(axis=1, pct=True)  # ← cross-sectional rank
```

**Verdict: substantial rewrite.** Three critical differences: (1) explicit `shift(1)` lagging invariant to prevent look-ahead, (2) log-sum momentum instead of `pct_change`, (3) cross-sectional rank and MultiIndex output. The conceptual skeleton (vol + mom + z-score) is shared.

---

### `paam_lab/overlay.py`

**Closest paamcode analog:** `ch09_active_risk_management.py` (both implement vol-targeting overlays)

**Diff summary:**

```diff
-def apply_vol_target(ret: np.ndarray, vol_target=0.12, window=20, lam_min=0.0, lam_max=2.0):
-    sigma_hat = rolling_vol(ret, window=window)
-    lam = np.empty_like(ret)
-    lam[:] = 1.0
-    mask = ~np.isnan(sigma_hat) & (sigma_hat > 0.0)
-    lam[mask] = vol_target / (sigma_hat[mask] * np.sqrt(252.0))
-    lam = np.clip(lam, lam_min, lam_max)
-    return lam * ret
+def vol_target_multiplier(sigma_hat, target_vol=0.12, lambda_min=0.50, lambda_max=1.50):
+    """...Works for both scalar and pd.Series inputs..."""
+    if isinstance(sigma_hat, pd.Series):
+        return (target_vol / sigma_hat).clip(lower=lambda_min, upper=lambda_max)
+    return float(np.clip(target_vol / sigma_hat, lambda_min, lambda_max))
+
+def vmp_overlay_lagged(portfolio_returns, target_vol=0.12, window=21, ...):
+    """VMP overlay with explicit one-day lag (Moreira & Muir 2017)."""
+    lam_lagged = lam.shift(1)               # ← explicit look-ahead guard
+    scaled = portfolio_returns * lam_lagged
```

paam_lab adds: `drawdown_multiplier` (3-band step function vs binary threshold), `lagged_exposure`, `apply_overlay` (explicit lagging API), `vmp_overlay_lagged` (Moreira & Muir 2017). paamcode operates on numpy arrays; paam_lab on pandas Series.

**Verdict: substantial rewrite.** Core vol-targeting concept shared; paam_lab adds look-ahead guard, pandas interface, drawdown bands, and Moreira-Muir form.

---

### `paam_lab/evaluation/common.py`

**Closest paamcode analog:** `ch11_performance_backtesting.py` (both compute performance stats)

**Diff summary:**

```diff
-def performance_stats(returns: pd.Series, risk_free: float = 0.02):
-    ann_ret = returns.mean() * 252           # ← arithmetic mean (not geometric)
-    ann_vol = returns.std() * np.sqrt(252)   # ← ddof=1 by default
-    sharpe = (ann_ret - risk_free) / ann_vol
-    ...
+def performance_stats(simple_returns):
+    wealth = (1.0 + simple_returns).cumprod()
+    annualized_return = wealth.iloc[-1] ** (TRADING_DAYS / len(simple_returns)) - 1.0
+    annualized_volatility = simple_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
+    sharpe = annualized_return / annualized_volatility  # ← no rf deduction
+    ...
+    return pd.Series({"final wealth": ..., "annualized return": ..., ..., "observations": ...})
```

paam_lab adds: `drawdown_from_wealth`, `turnover_from_weights`, `transaction_cost_from_turnover`, `max_drawdown`, `hit_ratio`, `equity_from_returns`, `drawdown_from_equity`, `rank_ic`, `summarize_prediction_errors`, `summarize_strategy`. All original.

**Verdict: substantial rewrite.** Same concept (`performance_stats`), but different annualization (geometric vs arithmetic), ddof, rf treatment, and sign convention for max_drawdown. Everything beyond `performance_stats` is paam_lab original.

---

### `paam_lab/evaluation/portfolio.py`

**Closest paamcode analog:** `ch11_performance_backtesting.py` (conceptually — both backtest strategies)

**Diff summary:** paamcode's `backtest` is 6 lines (single-asset momentum). paam_lab's `build_portfolio_proxy` is 60+ lines building a full cross-sectional backtesting framework with transaction costs, benchmark, rank IC time series, and summary table. Zero code shared.

**Verdict: paam_lab original.** Entirely new abstraction layer.

---

### `paam_lab/ml/common.py`

**Closest paamcode analog:** `ch12_ml_workflow.py` (both implement ML data pipeline)

**Diff summary:**

```diff
-def rolling_split(X, y, n_splits=5):
-    tscv = TimeSeriesSplit(n_splits=n_splits)
-    for train_idx, val_idx in tscv.split(X):
-        yield X.iloc[train_idx], X.iloc[val_idx], ...
+def make_chronological_splits(date_index, train_share=0.70, validation_share=0.15):
+    """Three-way split: train / validation / test."""
+    ...
```

paam_lab adds: `fit_standardizer`, `transform_standardized`, `build_design_matrix`, `build_target_panel` (multi-horizon, multi-target-type), `build_prediction_frame`. All original. paamcode uses sklearn `TimeSeriesSplit`; paam_lab uses date-based chronological splits.

**Verdict: substantial rewrite.** Split concept is shared; everything else is paam_lab original.

---

### `paam_lab/ml/linear.py`

**Closest paamcode analog:** `ch13_linear_glm.py` (both implement Ridge regression)

**Diff summary:**

```diff
-# paamcode: uses sklearn Pipeline + StandardScaler + Ridge
-ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=5.0))])
-ridge.fit(X_train, y_train)

+# paam_lab: closed-form via linear algebra
+def fit_ridge_closed_form(X, y, alpha=1.0):
+    X_aug = np.column_stack([np.ones(len(X)), X])
+    penalty = np.eye(X_aug.shape[1]); penalty[0, 0] = 0.0
+    beta = np.linalg.solve(X_aug.T @ X_aug + alpha * penalty, X_aug.T @ y)
+    return beta.ravel()
```

**Verdict: paam_lab original.** Completely different implementation (closed-form vs sklearn). The closed-form avoids sklearn dependency and exposes the math directly.

---

### `paam_lab/bootstrap.py`

**Closest paamcode analog:** No analog found.

paamcode has no package infrastructure — every script is standalone. `bootstrap.py` provides `discover_repo_root` and `ensure_src_on_path` for notebook sessions to locate the `src/` tree. Entirely original.

**Verdict: paam_lab original.**

---

### `paam_lab/io/handoff.py`

**Closest paamcode analog:** No analog found.

Manages the ch10 artifact handoff: resolves paths, loads manifests, checks artifact existence, summarizes DataFrames. No equivalent in paamcode.

**Verdict: paam_lab original.**

---

## 6. Synthesis

### a. Does paamcode define any class hierarchy or ABC for strategies?

**No.** Zero strategy abstractions exist. All strategy logic is standalone module-level functions. For RL, two environment dataclasses (`PortfolioEnv`, `SimpleEnv`) implement a `reset`/`state`/`step` interface, but there is no Strategy superclass, no ABC, and no polymorphism. There is nothing to inherit from.

---

### b. Does paamcode define a Panel or Universe class?

**No.** Assets are plain Python lists. The closest structural object is a `mandate` dict (asset list + constraints + risk_budget) in ch01, but it is just a Python dict — not a class, not importable, not reusable across scripts.

---

### c. EODHD client pattern in paamcode (if any)?

**Not present.** Every script repeats this exact pattern:

```python
DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
```

There is no EODHD API key, no HTTP client, no authentication logic. Data is pre-downloaded as a static 8-asset CSV (`pyaiam_eod.csv`) served from Hilpisch's own domain. The fallback URL is not EODHD's API endpoint.

---

### d. Black-Litterman in paamcode — deterministic view generators we could borrow?

**BL is present** (`ch06_black_litterman.py`). The `black_litterman()` function is clean and portable. However, the view generators are **hardcoded literals**:

```python
P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, -1, 0]])
Q = np.array([0.01, 0.005])
OMEGA = np.diag([0.0004, 0.0009])
```

No `ViewGenerator` class, no function that derives P/Q/Ω from signals or data. There is nothing to borrow for view generation; we must build that from scratch.

**Contrast with paam_lab:** `bl_posterior_returns()` uses the cleaner formula form  
`mu_BL = pi + τΣP^T(PτΣP^T + Ω)^{-1}(Q - Pπ)` while paamcode's `black_litterman()` inverts `τΣ` and uses the matrix-inverse form. Both are mathematically equivalent; paam_lab's form is numerically more stable for medium-sized universes.

---

### e. What is in `paamcode/code/` but NOT in `paam_lab/src/paam_lab/`?

Items worth porting or adapting for next-gen-aiam:

| paamcode item | Port target in next-gen-aiam | Notes |
|---|---|---|
| `ch07`: `simulate_portfolio` + `risk_measures_from_losses` | `aiam.evaluation.risk` | Clean VaR/ES from simulation; no deps. |
| `ch08`: `vol_and_rcov` | `aiam.construction` | 5-line Euler risk contribution. Drop-in. |
| `ch06`: `black_litterman()` body | `aiam.construction.bl` | Formula is portable; build view-generator wrapper around it. |
| `ch15`: `MLP`, `AutoEncoder` | `aiam.strategies.dl` | Teaching architectures; useful as baseline DL strategies. |
| `ch16`: `ReturnLSTM`, `SimpleTransformer` | `aiam.strategies.dl` | Same; baseline sequence models. |
| `ch17`/`ch18`: `PortfolioEnv`, `SimpleEnv`, Q-learning, policy-gradient | `aiam.strategies.rl` | Reference designs for RL strategy environments. |

**paamcode-only patterns worth noting (do NOT port directly):**

- `performance_stats` (7× duplicated in paamcode) — paam_lab's version is strictly better; do not backport paamcode's.
- All plotting/visualization code — not a library concern; notebooks only.

---

### f. What is in `paam_lab/src/paam_lab/` but NOT in `paamcode/code/`?

Everything below is **user-original work**. These are the outputs of prior notebook extraction sessions and should be ported wholesale into next-gen-aiam:

**`construction.py` (paam_lab original):**
- `gmv_weights` (SLSQP, long-only)
- `msr_weights` (SLSQP, long-only)
- `emv_weights`, `mdp_weights`
- `rp_weights` / `solve_risk_budget` (risk parity, iterative)
- `bl_posterior_returns` + `bl_weights` (decomposed BL interface)
- `hrp_weights`, `harp_weights`, `hap_weights`, `hcp_weights` (hierarchical methods from Lopez de Prado / Hlavaty & Smith / Raffinot)
- `corr_dist_matrix`, `_quasi_diag`, `_cluster_var`, `_bisect`, `_harp_bisect`

**`covariance.py` (paam_lab original):**
- `ledoit_wolf_cov` (sklearn LedoitWolf)
- `oas_cov` (sklearn OAS)
- `rmf_cov` (Random Matrix Filtering, Marchenko-Pastur)

**`data_quality.py` (mostly original):**
- `validate_panel_structure`, `summarize_missingness`
- `find_duplicate_dates`, `find_stale_segments`
- `flag_extreme_moves`, `flag_split_like_moves`
- `clean_price_panel` (dedup + ffill with audit)
- `plot_flagged_points`

**`features.py` (substantial rewrite):**
- `build_feature_panel` — with look-ahead guard (`.shift(1)` lagging invariant), log-sum momentum, cross-sectional rank

**`overlay.py` (substantial rewrite):**
- `vmp_overlay_lagged` (Moreira & Muir 2017)
- `drawdown_multiplier` (3-band)
- `lagged_exposure`, `apply_overlay`

**`evaluation/common.py` (substantial rewrite):**
- `performance_stats` with geometric annualization and no-rf Sharpe
- `hit_ratio`, `rank_ic`, `summarize_prediction_errors`, `summarize_strategy`
- `equity_from_returns`, `drawdown_from_equity`, `drawdown_from_wealth`
- `turnover_from_weights`, `transaction_cost_from_turnover`

**`evaluation/portfolio.py` (paam_lab original):**
- `build_portfolio_proxy` (full cross-sectional backtest with TC and rank IC)
- `build_rank_proxy_bundle`
- `scores_to_rank_long_only_weights`

**`ml/common.py` (substantial rewrite):**
- `make_chronological_splits` (3-way date-based split)
- `build_target_panel` (multi-horizon, multi-target-type)
- `build_prediction_frame`
- `fit_standardizer`, `build_design_matrix`

**`ml/linear.py` (paam_lab original):**
- `fit_ridge_closed_form` (closed-form `(X^TX + αI')^{-1}X^Ty`)
- `predict_ridge_closed_form`

**`bootstrap.py` (paam_lab original):**
- `discover_repo_root`, `ensure_src_on_path`

**`io/handoff.py` (paam_lab original):**
- Ch10 artifact manifest management
