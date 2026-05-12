# Repo Review for Claude Chat

Structured read-only inventory of `~/Projects/ai_asset_management_lab` (OLD) and
`~/Projects/next-gen-aiam` (NEW). Input for an architecture conversation. No code was changed; the only commands run were file reads and `pytest --collect-only`.

---

## PART 1: OLD repo inventory

### OLD CLAUDE.md

```markdown
# CLAUDE.md

## Project
AI Asset Management Lab — research notebook project based on
*Python and AI for Asset Management*. 34 notebooks across 23 chapters,
including book-aligned chapters (01–23) and lab-extended notebooks (★).

## Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

## Structure
- `notebooks/` — one notebook per chapter, fully self-contained
- `src/paam_lab/` — shared helper package (data_quality, features, construction, covariance, overlay, bootstrap, evaluation/, io/, ml/); installed editable via pyproject.toml
- `data/` — local data files only; primary panel `pyaiam_eod.csv`, derived outputs in `data/derived/`
- `docs/execution_packets/` — chapter contracts and execution packets (read before modifying any chapter)
- `docs/PENDING.md` — pending work, deferred items, and chapter contract gaps

## Non-negotiable rules
- Never import across notebooks — each is fully self-contained
- No external data downloads — local files only
- No new dependencies without explicit instruction
- Do not run, execute, or commit notebooks — write only
- Do not run any git commands of any kind

## Notebook editing

Never use sed, perl, or regex to modify `.ipynb` files — the JSON
structure will break. Use `tools/patch_cell.py` exclusively.

Running `tools/patch_cell.py` is permitted — it is the canonical
write-only edit path for notebooks. It modifies cell source while
preserving outputs and execution_count. No other code execution is
allowed; the write-only rule still applies to notebook kernels,
pytest, nbconvert, and arbitrary scripts.

Patch mode (replace cell source, preserve outputs and execution_count):
    python tools/patch_cell.py <nb_path> <cell_index> <source_file>

Delete mode (remove a cell entirely):
    python tools/patch_cell.py --delete <nb_path> <cell_index>

## Skills scope

This project keeps all globally-installed skills (`~/.claude/skills/`) and
project-local stack skills (`.claude/skills/`) intact. Behavior is shaped
here, not by removing skills.

The non-negotiables above (write-only, no execution, no git) override any
skill default that implies otherwise. Specifically:

- **Plans are documents, not actions.** `writing-plans` and `executing-plans`
  produce written cells in notebooks or markdown in `docs/`. An "executed
  plan" means the cells are written; running them is the user's job.
- **TDD framing applies to writing tests** in `tests/` alongside helpers in
  `src/paam_lab/`. It does not authorize running pytest, nbconvert, or
  notebook kernels.
- **Verification happens by re-reading** the cell, checking against the
  execution packet, and asserting against locked numbers in the chapter —
  not by executing code.
- **Git skills** (`using-git-worktrees`, `finishing-a-development-branch`,
  and any commit-flavored behavior) do not apply. All git operations are
  user-initiated, outside Claude Code sessions. Claude does not co-author
  commits.
- **Prototyping skills** (`prototype`) do not apply — this is a research
  notebook project, not a product surface.
- **Duplicate skills** (`tdd` from mattpocock alongside `test-driven-development`
  from obra) are tolerated; both shape framing without authorizing execution.
- **Tavily skills** are installed for potential future research-fetch
  workflows but are not active in current chapters. If a chapter needs
  external research, it gets a contract amendment first, not an ad-hoc
  Tavily call.

Project-local skills (`.claude/skills/`) — `pandas-pro`, `scikit-learn`,
`machine-learning`, `python-testing-patterns`, `senior-data-scientist` —
are reference material for the stack and do not override these rules.

## Data loading pattern (all notebooks)
from pathlib import Path
candidate_paths = [
    Path.cwd() / "data" / "pyaiam_eod.csv",
    Path.cwd().parent / "data" / "pyaiam_eod.csv",
    Path.cwd() / "data" / "eod_tech.csv",
    Path.cwd().parent / "data" / "eod_tech.csv",
]
data_path = next((p for p in candidate_paths if p.exists()), None)

Preferred basket: ['AAPL', 'NVDA', 'JPM', 'SPY']
Fallback basket: ['AAPL', 'NVDA', 'MSFT', 'GOOG']

## Notebook conventions
- np.set_printoptions(precision=4, suppress=True) in every notebook
- Plotting: dpi=120, grid.alpha=0.25, spines top/right off, tight_layout()
- Annualize: returns ×252, vol ×√252, covariance ×252
- Risk-free rate: rf = 0.03
- Display tables with display(df.round(4))
- Assert solver success and abs(weights.sum()-1) < 1e-6 after every optimization

## Notebook style

Conventions established through Ch17–Ch18 review. New chapters should land these from the start; existing chapters drift toward them on next major edit.

### Cell granularity

- One conceptual unit per cell. If a cell's purpose cannot be described in one short sentence, split it.
- Splits happen between complete function/class definitions, never mid-body.
- Each code-cell cluster gets a `###` markdown header above it, named by purpose. The header is one line, no intro paragraph.
- Examples of acceptable clusters: "Environment mechanics — state, action, reward"; "DQN building blocks — replay buffer, Q-network, optimization step"; "Action selectors". Heterogeneous definitions (env + training + rollouts in one cell) get split.

### Code-first, prose-second

- If a finding can be computed, compute it. Delta columns vs predecessor baselines, ratios, differences, deltas all belong in the code, not in prose.
- If a finding can be visualized, plot it. Training trajectories, action-mix charts, policy comparisons, regime breakdowns belong as figures.
- Markdown earns its place only for: chapter framing; one-sentence headline-naming per major output; caveats that are not visible in the data.
- Findings cells are short: one paragraph headline + one paragraph caveat. If more is needed, add a chart, not more prose.
- No "Reading", "Discussion", or "Interpretation" sub-sections. No bullet lists that re-state numbers already shown in nearby tables or plots.

### Section structure

- `##` for major chapter sections, matching the execution packet's spine.
- `###` for sub-sections within a code group.
- Headings stand alone — no intro paragraph after a heading unless the section genuinely cannot be understood from the code that follows.

### Math notation

- Mathematical content uses LaTeX. Jupyter renders MathJax inline (`$x_t$`) and display (`$$ \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, (G_t - b_t) $$`) without setup.
- Algorithmic steps that include math go in numbered lists with inline LaTeX, not in monospace pseudo-code blocks.
- Reserve fenced code blocks for executable code or shell commands. Pseudo-code that mixes algorithmic structure with math expressions belongs in prose + LaTeX.

### Layered editing discipline

When working on an existing chapter, separate edits by layer rather than mixing them in one session:

- **Layer 1** — structural fixes. Blockers, predecessor continuity, missing setup cells. Mechanical, write-only. Run the notebook after Layer 1 to produce real numbers.
- **Layer 2** — interpretive content. Findings, takeaways, bridge-to-next-chapter. Drafted only after execution, against actual numbers.
- **Layer 3** — visualization additions and prose trims. Shifts the code:prose ratio toward visualization-carries-the-weight. Should rarely be needed if visualizations were planned in the contract.
- **Layer 4** — cell granularity refactor. Splits monolithic cells into single-concern sub-cells with `###` headers. Should rarely be needed if cells were granular from the start.

A new chapter should meet Layer 1 and Layer 4 standards from the start. Layer 2 follows execution. Layer 3 is rarely needed if the contract anticipated the visualizations.

## Workflow conventions

The book chapters from Ch10 onward follow a governance-first workflow that
the regime/lab extensions do not require. New book chapters must follow
this pattern; lab extensions remain free-form.

- Each chapter has a contract (`docs/execution_packets/chXX_contract.md`)
  and an execution packet (`docs/execution_packets/chXX_execution_packet.md`).
  Contract states scope and non-negotiables; packet states the spine of cells
  and explicit avoid-list.
- Hyperparameter selection (α for regularizers, depth/n_estimators/learning_rate
  for trees) is validation-driven on a small log grid — not hardcoded, not
  exhaustive search. Per-regularizer or per-family grids are bounded by the
  packet's "no sweep explosion" rule.
- Each chapter closes with an honest-finding bullet acknowledging where the
  selected model fails to beat the relevant benchmark, where applicable.
- "Done" includes a code-quality review pass: dead imports, alias parades,
  function shadows between paam_lab and notebook-local helpers, and trivial
  one-line wrapper functions are surfaced and resolved before the chapter
  is declared complete. Numerical correctness alone is not "done."
- Chapter notebooks reuse paam_lab helpers wherever possible; chapter-local
  helpers live in a single helpers cell only when they're notebook-specific
  glue (e.g., model-selection logic that is not reusable across chapters).
- Current pending work, deferred items, and chapter contract gaps live in
  `docs/PENDING.md` — consult at session start when picking up work.

## Regime-switching heuristic

19c findings — regime-to-method mapping (input to 19d):
- R0 (High & Accelerating) → EW
- R1 (High & Decelerating) → MDP
- R5 (Low & Contracting)   → MSR (least bad in crisis)
- R7 (Low & Declining)     → MDP
- Default (thin regimes)   → MDP (most robust overall)

## Reference layer

Deep reference — fetch when needed, not at session start:

- **Notebook inventory** (full list, LOCKED status, lab-extended notes, rename history):
  `docs/notebook_inventory.md` — consult when picking up any chapter work.
- **Citation map** (authoritative citations per notebook):
  `docs/citation_map.md` — consult when editing 08d, 08e, 08f, or 09b.
- **Data artifacts and regime engine notes**:
  `docs/data_artifacts.md` — consult when tracing data flow across 19b/c/d.
- **Pending work, deferred items, chapter contract gaps**:
  `docs/PENDING.md` — see also Workflow conventions above.
```

### OLD src tree

```
/Users/frasagui/Projects/ai_asset_management_lab/src/paam_lab/
├── __init__.py
├── __pycache__/                  (compiled bytecode — omitted)
├── bootstrap.py
├── construction.py
├── covariance.py
├── data_quality.py
├── evaluation/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── common.py
│   └── portfolio.py
├── features.py
├── io/
│   ├── __init__.py
│   ├── __pycache__/
│   └── handoff.py
├── ml/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── common.py
│   └── linear.py
└── overlay.py

8 directories, 42 files (incl. .pyc)
```

`src/paam_lab/__init__.py` is the docstring `"""PAAM Lab shared utilities."""`
only — no re-exports, no `__version__`. Subpackage `__init__.py` files
re-export the public API of their `common.py` modules (see source files
below).

### Source files

#### `construction.py` — 420 lines

External-library imports:
- `numpy as np`
- `pandas as pd`
- `scipy.cluster.hierarchy.fcluster, linkage`
- `scipy.optimize.minimize`
- `scipy.spatial.distance.squareform`

No `sklearn`, `hmmlearn`, `statsmodels`, `cvxpy`, `riskfolio` — pure
NumPy/pandas/SciPy.

Public functions / classes (signatures + actual docstrings, quoted verbatim):

- `def _to_array(x) -> np.ndarray` — `"Coerce to float numpy array. Notebooks alias this as _arr or _as_array."` (private helper)
- `def port_vol(w, Sigma) -> float` — `"Portfolio annualised volatility. Canonical form from 08g."`
- `def port_ret(w, mu) -> float` — `"Portfolio annualised return."`
- `def sharpe(w, mu, Sigma, rf: float = 0.03) -> float` — `"Annualised Sharpe ratio. Canonical form from 08d."`
- `def pct_rc(w, Sigma) -> np.ndarray` — `"Percent risk contribution vector via Euler decomposition. From 08d."`
- `def div_ratio(w, vols, Sigma) -> float` — `"Diversification ratio: weighted-avg individual vol / portfolio vol. From 08d."`
- `def ew_weights(n: int) -> np.ndarray` — `"Equal-weight portfolio."`
- `def gmv_weights(Sigma) -> np.ndarray` — `"Long-only global minimum variance via SLSQP. Canonical form from 08g."`
- `def msr_weights(mu, Sigma, rf: float = 0.03) -> np.ndarray` — `"Maximum Sharpe ratio portfolio (long-only) via SLSQP. Canonical form from 08g."`
- `def emv_weights(vols) -> np.ndarray` — `"Equal marginal volatility: w_i proportional to 1/sigma_i. From 08d."`
- `def mdp_weights(vols, Sigma) -> np.ndarray` — `"Maximum diversification portfolio via SLSQP. From 08d/08g."`
- `def project_to_simplex(v) -> np.ndarray` — `"Project v onto the probability simplex (isotonic-sort method). From 09."`
- `def solve_risk_budget(Sigma, budget=None, max_iter: int = 5000, tol: float = 1e-12, damping: float = 0.5) -> np.ndarray` — `"Risk-budgeting via iterative proportional scaling. budget=None uses equal risk budget (risk parity). Parameter renamed from target_shares to budget per §6.4 of the extraction contract. From 09."`
- `def rp_weights(Sigma, **kwargs) -> np.ndarray` — `"Risk parity: solve_risk_budget with equal budget. One-line wrapper."`
- `def bl_posterior_returns(pi, Sigma, P, Q, Omega, tau: float = 0.05) -> np.ndarray` — `"BL posterior expected returns from Hilpisch §6.4. Formula: mu_BL = pi + tau*Sigma*P^T * (P*tau*Sigma*P^T + Omega)^{-1} * (Q - P*pi). Extracted from the 08h monolithic bl_weights; view construction (P, Q, Omega) stays in the calling notebook."`
- `def bl_weights(mu, Sigma, delta: float = 2.5, long_only: bool = False) -> np.ndarray` — `"Thin MV wrapper maximising w@mu - 0.5*delta*w@Sigma@w with sum(w)=1. Canonical decomposed form per §6.2 of the extraction contract..."`
- `def _quasi_diag(link) -> list` — `"Recover leaf ordering from linkage by recursive tree traversal."` (private)
- `def _cluster_var(cov) -> float` — `"Inverse-variance-weighted cluster variance."` (private)
- `def _bisect(cov, sort_ix) -> np.ndarray` — `"Allocate weights by inverse cluster variance across recursive splits..."` (private)
- `def _logistic_scale(alpha) -> np.ndarray` — `"Z-score to [-10,+10], apply logistic, renorm to [0,1]."` (private)
- `def _harp_bisect(cov, alpha_sorted, sort_ix, lam) -> np.ndarray` — `"Recursive bisection blending HRP (inverse-var) and HAP (logistic-alpha) splits..."` (private)
- `def corr_dist_matrix(returns) -> np.ndarray` — `"Correlation distance matrix: sqrt(0.5*(1 - rho_ij)). Public canonical name superseding the private _corr_dist used in 08g/08h (§6.5 of the extraction contract). From Lopez de Prado (2016)."`
- `def hrp_weights(returns) -> pd.Series` — `"HRP portfolio weights via single-linkage seriation and recursive bisection. Canonical form from 08e..."`
- `def harp_weights(returns, lam: float = 0.5) -> pd.Series` — `"HARP weights blending HRP risk-parity and logistic-alpha HAP splits. lam=1.0 is pure HRP; lam=0.0 is pure HAP..."`
- `def hap_weights(returns) -> pd.Series` — `"HAP weights: alpha-only hierarchical allocation (HARP at lam=0.0). Named wrapper for API symmetry with HRP/HARP/HCP..."`
- `def hcp_weights(returns, method: str = 'ward', n_clusters: int | None = None) -> pd.Series` — `"HCP: equal capital per cluster, equal weight within cluster. n_clusters defaults to max(2, N//3) per 08e canonical form..."`

Two distinct input shapes coexist in this file:
- **Vector/matrix shape** (`np.ndarray` or `pd.Series`/`pd.DataFrame` coerced via `_to_array`): `gmv_weights(Sigma)`, `msr_weights(mu, Sigma)`, `mdp_weights(vols, Sigma)`, `bl_weights(mu, Sigma, ...)`, `solve_risk_budget(Sigma, budget)`, etc. Returns `np.ndarray` of weights without an index.
- **Returns-DataFrame shape**: hierarchical constructors `hrp_weights(returns)`, `harp_weights(returns, lam)`, `hap_weights(returns)`, `hcp_weights(returns, method, n_clusters)`. Returns `pd.Series` indexed by asset.

There is no time index, no "as-of" date concept anywhere; callers in the
notebooks slice a returns window before calling these functions.

#### `covariance.py` — 45 lines

External-library imports:
- `numpy as np`
- `pandas as pd`
- `sklearn.covariance.LedoitWolf, OAS`

Public functions:

- `def empirical_cov(returns: pd.DataFrame) -> np.ndarray` — `"Standard sample covariance, annualized by 252."`
- `def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray` — `"Ledoit-Wolf analytical shrinkage toward identity-scaled target."`
- `def oas_cov(returns: pd.DataFrame) -> np.ndarray` — `"Oracle Approximating Shrinkage (OAS) covariance estimator."`
- `def rmf_cov(returns: pd.DataFrame) -> np.ndarray` — `"Random Matrix Filtering: collapse Marchenko-Pastur noise eigenvalues to their mean."`

All return `np.ndarray` (annualised × 252) with no asset-name index. The
single shape is `(returns: pd.DataFrame) -> np.ndarray`.

#### `overlay.py` — 118 lines

External-library imports:
- `numpy as np`
- `pandas as pd`

Module constant: `ANNUALIZATION = 252`

Public functions:

- `def rolling_annualized_vol(returns: pd.Series, window: int = 21) -> pd.Series` — `"Rolling annualised volatility using ddof=0 (population std). Source: 09 canonical form. ddof=0 matches the notebook convention. Note: vmp_overlay_lagged uses ddof=1 (pandas default) per 09b."`
- `def vol_target_multiplier(sigma_hat, target_vol: float = 0.12, lambda_min: float = 0.50, lambda_max: float = 1.50)` — `"Scale factor λ = target_vol / σ̂, clipped to [lambda_min, lambda_max]. Works for both scalar and pd.Series inputs..."`
- `def drawdown_multiplier(drawdown: pd.Series, band1: float = 0.05, band2: float = 0.10, full: float = 1.0, reduced: float = 0.75, minimal: float = 0.50) -> pd.Series` — `"Three-band step function: full → reduced → minimal as drawdown deepens..."`
- `def lagged_exposure(exposure_signal: pd.Series) -> pd.Series` — `"Shift exposure by 1 day; fill leading NaN with 1.0 (fully invested). The fill-to-1.0 convention means the first day of an overlay series is always at full exposure — no look-ahead on day 1."`
- `def apply_overlay(base_returns: pd.Series, exposure_signal: pd.Series) -> tuple[pd.Series, pd.Series]` — `"Apply a (lagged) exposure signal to base returns. Returns (strategy_returns, applied_exposure). The exposure is lagged inside this function — callers should NOT pre-lag the signal."`
- `def vmp_overlay_lagged(portfolio_returns: pd.Series, target_vol: float = 0.12, window: int = 21, lam_min: float = 0.25, lam_max: float = 1.5) -> tuple[pd.Series, pd.Series]` — `"VMP overlay with explicit one-day lag to avoid look-ahead bias. Canonical form from 09b §4 (Moreira & Muir 2017, Journal of Finance 72(4), 1611–1644). Returns (scaled_returns, exposure_multiplier). λ_t = clip(σ* / σ̂_{t-1}, lam_min, lam_max). scaled_returns_t = portfolio_returns_t × λ_t. Note: uses ddof=1 (pandas default std) per 09b source, consistent with the published Moreira & Muir formula."`

#### `evaluation/common.py` — 155 lines  **⚠ Sharpe bug here**

External-library imports:
- `numpy as np`
- `pandas as pd`

Module constant: `TRADING_DAYS = 252`

Public functions:

- `def drawdown_from_wealth(wealth)` — no docstring; returns `wealth / wealth.cummax() - 1.0`, a **non-positive** Series.
- `def performance_stats(simple_returns)` — no docstring. Returns Series with keys `{"final wealth", "annualized return", "annualized volatility", "Sharpe", "max drawdown", "observations"}`. **This is the source of the Sharpe bug — see Part 3.C.**
- `def turnover_from_weights(weights)` — no docstring.
- `def transaction_cost_from_turnover(turnover, cost_bps=10)` — no docstring.
- `def max_drawdown(returns: pd.Series) -> float` — `"Maximum drawdown as a positive magnitude in [0, 1]. Canonical convention: positive float, matching performance_stats and 09b. The 08g/08h inline form returns a negative float — callers that negated before display must remove the negation after this extraction."`
- `def hit_ratio(returns: pd.Series) -> float` — `"Fraction of calendar months with positive portfolio return. Uses resample('ME') on pandas ≥ 2.2; falls back to 'M' on older installs..."`
- `def equity_from_returns(returns: pd.Series) -> pd.Series` — `"Cumulative wealth series from simple daily returns. Starts at 1+r[0]."`
- `def drawdown_from_equity(equity: pd.Series) -> pd.Series` — `"Peak-to-current drawdown from an equity curve. Returns a non-negative Series: 0 at each peak, positive during drawdown. Contrast with drawdown_from_wealth which returns a non-positive Series."`
- `def rank_ic(x, y)` — `"Spearman rank correlation between a prediction vector and a realized vector, with NaN handling. Used as a cross-sectional signal-quality metric. Returns np.nan when fewer than two valid paired observations are available."`
- `def summarize_prediction_errors(frame)` — no docstring.
- `def summarize_strategy(returns: pd.Series, exposure: pd.Series | None = None) -> pd.Series` — `"Strategy summary including average exposure. Richer than performance_stats: includes average exposure and fraction of de-risked days. Does not replace or modify performance_stats. max drawdown is returned as a positive magnitude (canonical convention)."`

The buggy Sharpe lines are quoted in Part 3.C below.

#### `io/handoff.py` — 179 lines

External-library imports:
- `json`, `pathlib.Path`, `typing.Any`
- `pandas as pd`
- `paam_lab.bootstrap.discover_repo_root` (internal)

Public functions:

- `def resolve_ch10_handoff_root(repo_root=None, start=None) -> Path` — no docstring.
- `def resolve_ch10_manifest_path(repo_root=None, artifact_root=None, start=None) -> Path` — no docstring.
- `def load_ch10_manifest(manifest_path=None, repo_root=None, artifact_root=None, start=None) -> dict[str, Any]` — no docstring.
- `def check_ch10_handoff_artifacts(manifest, artifact_root=None, required_keys=REQUIRED_HANDOFF_KEYS) -> pd.DataFrame` — no docstring.
- `def summarize_ch10_handoff_artifacts(empirical_clean, empirical_returns_ready, empirical_features_long) -> pd.DataFrame` — no docstring.
- `def load_ch10_handoff(repo_root=None, artifact_root=None, manifest_path=None, preview_rows=3, strict=True) -> dict[str, Any]` — no docstring.
- `def load_feature_panel_artifact(repo_root=None, artifact_root=None, manifest_path=None, preview_rows=3, strict=True) -> dict[str, Any]` — no docstring; calls `load_ch10_handoff` with the same arguments.

Module constants: `CH10_HANDOFF_DIR = Path("data/derived/ch10_handoff")`, `CH10_MANIFEST_NAME = "manifest.json"`, `REQUIRED_HANDOFF_KEYS = ("empirical_clean", "empirical_returns_ready", "empirical_features_ready")`.

The loader's terminal return dict contains: `repo_root, artifact_root, manifest_path, manifest, artifact_names, asset_list, feature_list, artifact_check_frame, artifact_summary, row_count, date_range, interface_note, empirical_clean, empirical_returns_ready, empirical_features_long, empirical_features_ready, feature_panel, empirical_features_panel, feature_preview`.

The wide `feature_panel` returned has a two-level column MultiIndex `(feature, asset)` and a `Date` row index.

#### `data_quality.py` — 325 lines (exists)

External-library imports:
- `numpy as np`
- `pandas as pd`
- `matplotlib.pyplot` (lazily imported inside `plot_flagged_points`)

Public functions:

- `def validate_panel_structure(panel: pd.DataFrame, name: str = "panel") -> pd.DataFrame` — `"Return a one-row summary of panel shape, index type, and data quality counts."`
- `def summarize_missingness(panel: pd.DataFrame) -> pd.DataFrame` — `"Return per-asset missingness statistics sorted by missing count descending."`
- `def find_duplicate_dates(panel: pd.DataFrame) -> pd.DataFrame` — `"Return all rows whose date index value appears more than once."`
- `def find_stale_segments(panel: pd.DataFrame, min_length: int = 3) -> pd.DataFrame` — `"Find flat-price (stale) runs of length >= min_length in each asset series."`
- `def flag_extreme_moves(returns: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame` — `"Flag log-return observations where abs(log_return) >= threshold."`
- `def flag_split_like_moves(returns: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame` — `"Flag log-return observations where abs(log_return) >= threshold as split-like candidates."`
- `def clean_price_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` — `"Remove duplicate-date rows (keep last) then forward-fill one isolated gap per asset."`
- `def data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame` — `"Consolidated before/after missingness and data-currency report indexed by asset."`
- `def plot_flagged_points(prices: pd.DataFrame, flags: pd.DataFrame, asset: str, *, ax=None, title=None) -> matplotlib.figure.Figure` — `"Overlay flagged dates as red scatter points on an asset price series line plot..."`

The canonical panel shape across this module is a **wide DataFrame**: rows are dates (`DatetimeIndex`), columns are assets, values are prices (or returns).

#### `features.py` — 76 lines (exists)

External-library imports:
- `numpy as np`
- `pandas as pd`

Module constant: `TRADING_DAYS = 252`

Public functions:

- `def build_feature_panel(clean_prices: pd.DataFrame) -> pd.DataFrame` — `"Build a uniformly-lagged, cross-sectional feature panel from clean prices. **Lagging invariant**: all features are computed on lagged = log_returns.shift(1), where log_returns = log(price_t / price_{t-1}). This one-day lag ensures that every feature value at date T uses only information available through T-1. Consequently the feature named log_ret_1d is the prior day's return, not the current day's return..."` (long; full docstring quoted above in source listing).

Returns a wide DataFrame with a **two-level MultiIndex on columns** `(feature, asset)` where feature ∈ `{log_ret_1d, mom_5d, mom_20d, vol_20d, z_20d, cs_rank_mom_20d}`.

#### `bootstrap.py` — 33 lines (exists)

External-library imports: `sys`, `pathlib.Path` — standard library only.

Public functions:

- `def discover_repo_root(start: Path | None = None, marker: str = "README.md") -> Path` — `"Discover the repo root by walking upward until the marker file is found. Falls back to the provided start directory when the marker cannot be found."`
- `def ensure_src_on_path(repo_root: Path | None = None, start: Path | None = None) -> Path` — `"Ensure the local src/ tree is importable in notebook sessions."`

Note: there is no top-level `paam_lab.bootstrap` function for loading data — this module is purely about path discovery for the notebook environment. Returns-panel construction lives in the notebooks themselves.

#### Also present (not asked for, listed for completeness)

- `evaluation/portfolio.py` (163 lines) — `build_portfolio_proxy`, `build_rank_proxy_bundle`, `scores_to_rank_long_only_weights`. Consumes the ML prediction frames and produces wealth/turnover/summary bundles. Imports `performance_stats`, `drawdown_from_wealth`, `rank_ic`, `transaction_cost_from_turnover`, `turnover_from_weights` from `.common`. **Direct caller of the buggy `performance_stats`.**
- `ml/common.py` (259 lines) — chronological splits, standardiser, build_design_matrix, build_target_panel (with `horizon`, `target_type`), build_prediction_frame, build_oos_prediction_frame, build_prediction_metrics_frame, build_regression_metrics_frame. Imports `paam_lab.evaluation.common.rank_ic`. No sklearn — purely pandas/numpy.
- `ml/linear.py` (24 lines) — `fit_ridge_closed_form(X, y, alpha=1.0)`, `predict_ridge_closed_form(X, beta)`. No sklearn — closed-form numpy.

### Tests

Test files in `~/Projects/ai_asset_management_lab/tests/`:

```
/Users/frasagui/Projects/ai_asset_management_lab/tests/
├── __pycache__/
├── conftest.py
├── test_construction.py
├── test_covariance.py
├── test_data_quality.py
├── test_evaluation_common.py
├── test_features.py
└── test_overlay.py
```

No subdirectories. No regime-engine tests, no ml/linear tests, no
evaluation/portfolio tests, no io/handoff tests — only the seven files
above. One-sentence summary per file (read in full, not inferred from
name):

- **`conftest.py`** — Inserts `src/` on `sys.path` and defines two pytest fixtures, `corrupted_toy_panel` (10 business days, 3 assets, deliberately injected NaN / duplicate-date / stale-run / split-like jump) and `clean_toy_panel` (same shape, no defects), used by the data-quality tests.
- **`test_construction.py`** — 56 test cases plus three private-helper TestQuasiDiag/TestClusterVar/TestBisect classes, covering every public weight-builder (`port_vol`, `port_ret`, `sharpe`, `pct_rc`, `div_ratio`, `ew/gmv/msr/emv/mdp/rp/hrp/harp/hap/hcp/bl_weights`, `bl_posterior_returns`, `solve_risk_budget`, `project_to_simplex`, `corr_dist_matrix`) against numerical goldens locked from the 08d/08e/08f/08g/08h notebooks on `data/pyaiam_eod.csv` and `data/eod_tech.csv`.
- **`test_covariance.py`** — Asserts shape, symmetry, positive-definiteness, and four Frobenius / `[0,0]` / `[0,1]` golden values for each of `empirical_cov`, `ledoit_wolf_cov`, `oas_cov`, `rmf_cov` on the 4-asset pyaiam basket; includes a synthetic high-q test that activates the Marchenko-Pastur filter in `rmf_cov`.
- **`test_data_quality.py`** — 27 cases covering `validate_panel_structure`, `summarize_missingness`, `find_duplicate_dates`, `find_stale_segments`, `flag_extreme_moves`, `flag_split_like_moves`, `clean_price_panel`, `data_quality_report`, and `plot_flagged_points` (returns a `matplotlib.figure.Figure`, correct scatter count, override-axes), all against the corrupted/clean toy panel fixtures.
- **`test_evaluation_common.py`** — Tests for `max_drawdown`, `hit_ratio`, `equity_from_returns`, `drawdown_from_equity`, `summarize_strategy`, with goldens locked from the `ch08g_method_returns.parquet` EW series (`max_drawdown ≈ 0.3696`, `hit_ratio ≈ 0.7167`) and the 09 raw-baseline RP series (`Sharpe ≈ 1.0119`, `max drawdown ≈ 0.2368`, `ann return ≈ 0.2359`); also includes regression checks that `drawdown_from_wealth` stays non-positive and `performance_stats["max drawdown"]` stays non-negative. **Does not test the Sharpe bug** — no golden for `performance_stats["Sharpe"]` is asserted.
- **`test_features.py`** — Eight cases: structural (MultiIndex columns, feature set, asset set), the lagging invariant (perturbing the last price does not change the last feature row), `vol_20d` annualisation, `cs_rank_mom_20d` in `[0,1]` and row-mean = 2/3 for three-asset rows, and a round-trip through `build_feature_panel → tidy long → parquet → load_feature_panel_artifact`.
- **`test_overlay.py`** — 24 cases over `rolling_annualized_vol`, `vol_target_multiplier`, `drawdown_multiplier`, `lagged_exposure`, `apply_overlay`, `vmp_overlay_lagged`, including a golden end-to-end test loading `data/derived/ch08g_method_returns.parquet`, applying VMP to the MSR series with `(target=0.12, window=21, lam_min=0.25, lam_max=1.5)`, and asserting `Sharpe ≈ 1.3432`.

`pytest --collect-only` (first 80 lines):

```
============================= test session starts ==============================
platform darwin -- Python 3.14.0, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/frasagui/Projects/ai_asset_management_lab
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 94 items / 2 errors

<Dir ai_asset_management_lab>
  <Dir tests>
    <Module test_data_quality.py>
      <Function test_validate_panel_structure_happy_path>
      <Function test_validate_panel_structure_non_datetime_index>
      <Function test_validate_panel_structure_duplicate_date_count>
      <Function test_validate_panel_structure_empty_panel>
      <Function test_summarize_missingness_asset_ordering>
      <Function test_summarize_missingness_all_nan_asset>
      <Function test_summarize_missingness_no_nan>
      <Function test_find_duplicate_dates_no_duplicates>
      <Function test_find_duplicate_dates_one_date_duplicated>
      <Function test_find_duplicate_dates_multiple_duplicates>
      <Function test_find_stale_segments_no_stale>
      <Function test_find_stale_segments_detects_run>
      <Function test_find_stale_segments_min_length_boundary>
      <Function test_flag_extreme_moves_no_flags>
      <Function test_flag_extreme_moves_at_threshold_boundary>
      <Function test_flag_extreme_moves_multiple_assets>
      <Function test_flag_extreme_moves_nan_rows_ignored>
      <Function test_flag_split_like_moves_no_flags>
      <Function test_flag_split_like_moves_at_threshold>
      <Function test_flag_split_like_moves_review_note>
      <Function test_clean_price_panel_idempotent>
      <Function test_clean_price_panel_removes_dup_fills_gap>
      <Function test_clean_price_panel_returns_tuple>
      <Function test_data_quality_report_columns>
      <Function test_data_quality_report_asset_index>
      <Function test_data_quality_report_up_to_date>
      <Function test_data_quality_report_up_to_date_false>
      <Function test_plot_flagged_points_returns_figure>
      <Function test_plot_flagged_points_scatter_count>
      <Function test_plot_flagged_points_ax_override>
    <Module test_evaluation_common.py>
      <Class TestMaxDrawdown>
        <Function test_positive_convention>
        <Function test_closed_form>
        <Function test_no_drawdown>
        <Function test_golden_ew_from_08g>
        <Function test_sign_flip_from_08g_negative_form>
      <Class TestHitRatio>
        <Function test_fraction_in_unit_interval>
        <Function test_all_positive>
        <Function test_all_negative>
        <Function test_golden_ew_from_08g>
        <Function test_month_end_grouping_does_not_raise>
      <Class TestEquityFromReturns>
        <Function test_first_value>
        <Function test_cumulative_product>
        <Function test_returns_series>
        <Function test_flat_returns_grow_monotonically>
      <Class TestDrawdownFromEquity>
        <Function test_non_negative>
        <Function test_zero_at_peaks>
        <Function test_closed_form>
        <Function test_max_equals_canonical_max_drawdown>
      <Class TestSummarizeStrategy>
        <Function test_returns_series_with_correct_keys>
        <Function test_no_exposure_defaults>
        <Function test_max_drawdown_positive>
        <Function test_golden_rp_baseline_from_09>
        <Function test_with_exposure_reduces_fraction>
      <Class TestExistingFunctions>
        <Function test_drawdown_from_wealth_non_positive>
        <Function test_performance_stats_max_drawdown_positive>
        <Function test_drawdown_from_equity_vs_wealth_complement>
        <Function test_turnover_from_weights_zero_for_fixed>
    <Module test_features.py>
      <Function test_build_feature_panel_multiindex_columns>
      <Function test_build_feature_panel_feature_set>
      <Function test_build_feature_panel_asset_set>
      <Function test_build_feature_panel_lagging_invariant>
      <Function test_build_feature_panel_vol_annualization>
      <Function test_build_feature_panel_cs_rank_range>
      <Function test_build_feature_panel_cs_rank_row_mean>
      <Function test_build_feature_panel_round_trip>
    <Module test_overlay.py>
      <Class TestRollingAnnualizedVol>
        <Function test_constant_series_gives_zero_vol>
        <Function test_first_window_minus_one_is_nan>
        <Function test_annualization_factor>
        <Function test_uses_ddof0>
      <Class TestVolTargetMultiplier>
```

And the tail of the collection output:

```
==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_construction.py __________________
ImportError while importing test module '/Users/frasagui/Projects/ai_asset_management_lab/tests/test_construction.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
... importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_construction.py:24: in <module>
    from sklearn.covariance import LedoitWolf
E   ModuleNotFoundError: No module named 'sklearn'
__________________ ERROR collecting tests/test_covariance.py ___________________
... src/paam_lab/covariance.py:9: in <module>
    from sklearn.covariance import LedoitWolf, OAS
E   ModuleNotFoundError: No module named 'sklearn'
=========================== short test summary info ============================
ERROR tests/test_construction.py
ERROR tests/test_covariance.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
==================== 94 tests collected, 2 errors in 0.13s =====================
```

**Observation:** the collection runs against the system Python 3.14.0 with
pytest installed there, not against the project's `.venv` — `sklearn` is
not on its path. `test_construction.py` and `test_covariance.py` both fail
to collect. The remaining 94 tests collect cleanly without running.

### Regime engine notebooks (19b / 19c / 19d)

Located via `find` (also exposed under `notebooks/.ipynb_checkpoints/`
and under `10b/10c/10d` checkpoints — the `19x_*` files are the live
versions; `10[bcd]_*` checkpoints are earlier renames from the rename
history):

- `~/Projects/ai_asset_management_lab/notebooks/19b_macro_regime_engine.ipynb`
- `~/Projects/ai_asset_management_lab/notebooks/19c_regime_conditional_portfolio_analysis.ipynb`
- `~/Projects/ai_asset_management_lab/notebooks/19d_regime_allocation_signal.ipynb`

`find ... -name "*hmm*" -o -name "*markov*" -o -name "*state*"` returns no
additional files — there are no HMM-named, Markov-named, or state-named
notebooks. The only stateful machine is the one inside 19b. There is a
single additional checkpoint `ch06h_regime_engine-checkpoint.ipynb` (an
even earlier rename) but no live `ch06h` file.

#### `19b_macro_regime_engine.ipynb` (37 cells)

Opening markdown narrative cell (verbatim):

```markdown
# 19b — Macro Regime Engine

A rule-based macro regime detection layer built on top of the portfolio construction library.

Each of 8 macro indicators (GDP, CPI, unemployment, yield curve, VIX, SPX) is classified monthly into one of 8 regimes defined by the sign of its **level** (relative to a 5-year rolling mean), **change**, and **convexity**. A first-order Markov transition matrix then produces probability-weighted expected returns, which are combined across indicators using Sharpe weighting into a forward-looking signal for each asset.

**Sections**
0. Environment setup
1. Data pipeline — FRED + yfinance → `df_macro`
2. Feature engineering — `compute_features`
3. Regime assignment — `get_regime` → `df_regimes`
4. Regime visualisation
5. Transition probability matrices — `get_transition_proba`
6. Expected returns by regime — `get_exp_returns`, `run_regime_model`
7. Sharpe-weighted combination — `combine_exp_returns`
8. Export → `data/regime_signals.parquet`
```

Imports:

```
from IPython.display import display
from datetime import datetime
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import warnings
import yfinance as yf
```

Function/class definitions:

- `def _load_fred(code: str, col_name: str) -> pd.Series` — `"Load a FRED series from cache or pull from the web."`
- `def _load_yf(ticker: str, col_name: str) -> pd.Series` — `"Load a yfinance ticker's Adj Close from cache or pull from the web."`
- `def compute_features(series: pd.Series, lookback: int) -> pd.DataFrame` — `"Compute level, change, and convexity for a macro series. Level: 3-month moving average of the raw series. Change: level_t - level_{t-lookback}. Convexity: (level_t + level_{t-lookback}) / 2 - level_{t - lookback//2}..."`
- `def get_regime(row, col_lvl, col_chg, col_conv, mean_lvl, prev_regime=None)` — `"Determine the regime from level, change and convexity with dead-zones."`
- `def get_transition_proba(regime_series: pd.Series, nb_regimes: int = 8) -> pd.DataFrame` — `"Compute the first-order Markov transition probability matrix."`
- `def get_exp_returns(asset_returns: pd.Series, regime_series: pd.Series, rebalance_date: pd.Timestamp, min_regime_months: int = 6, nb_regimes_total: int = 8, ann_factor: int = 12) -> float` — `"Compute probability-weighted expected return as of rebalance_date. For the current regime i, expected return is: Exp_ret = Σ_j P(j|i) * avg_ret_in_regime_j. If P(stay in regime_i | regime_i) > 0.70 threshold, use the unconditional mean of the current regime directly..."`
- `def run_regime_model(asset_returns: pd.Series, regime_series: pd.Series, rebalance_dates: pd.DatetimeIndex, min_regime_months: int = 6, nb_regimes_total: int = 8, ann_factor: int = 12) -> pd.Series` — walk-forward driver that calls `get_exp_returns` at each rebalance date.
- `def combine_exp_returns(dict_exp_ret: dict, df_assets_log_ret: pd.DataFrame, rolling_window: int = 36) -> pd.DataFrame` — `"Combine expected returns from multiple indicators using Sharpe ratio weights..."`

External library dependencies for the regime engine: **none of the heavy
ML/HMM stack.** The regime model is hand-rolled:
- No `hmmlearn`.
- No `sklearn.mixture` / `GaussianMixture`.
- No `statsmodels` `MarkovSwitching` / `MarkovRegression`.
- Only `numpy`, `pandas`, `pandas_datareader`, `yfinance`, `matplotlib`, `seaborn`.

The "Markov" piece is a closed-form first-order transition matrix
computed by counting `(regime_t → regime_{t+1})` transitions
(`get_transition_proba` body: nested `for t` loop building a count
matrix, then row-normalising). The regime labels are produced by the
deterministic rule-based `get_regime` (8 rules over signs of level /
change / convexity, with two ε dead-zones at ±0.001 and a
`prev_regime` fallback for ambiguous cases). State is **only the prior
month's regime label**, used for fallback when the sign rule is
ambiguous; no continuous HMM state, no E-step, no Viterbi.

Public API consumed downstream:

The exported file `data/regime_signals.parquet` is the single contract
surface. Verified schema by direct read:

```
shape: (316, 18)
index: 2000-01-31 → 2026-04-30   (DatetimeIndex, monthly, no freq attribute)
columns:
  regime_GDP, regime_VIX, regime_SPX, regime_CPI, regime_UNEM,
  regime_YC10, regime_YC2, regime_YCSTEP                        — 8 × Int64
  exp_ret_SPY, exp_ret_TLT, exp_ret_GLD, exp_ret_QQQ,
  exp_ret_AAPL, exp_ret_NVDA, exp_ret_JPM, exp_ret_EURUSD,
  exp_ret_BTC-USD                                                — 9 × float64
  dominant_regime                                                — Int64
```

A second sibling file `data/regime_history.parquet` is also exported
holding the raw per-indicator regime time series.

Downstream call to get a regime label for a given date: load
`regime_signals.parquet`, then `df_signals["dominant_regime"].loc[date]`
(monthly granularity, end-of-month dates). That single column is the
entire public regime API; nothing else from the engine is consumed by
19c or 19d.

#### `19c_regime_conditional_portfolio_analysis.ipynb` (23 cells)

Opening markdown:

```markdown
# 19c — Regime-Conditional Portfolio Analysis

**Purpose:** Pure analysis notebook — reads the macro regime engine output from `19b`
and measures how six portfolio construction methods perform conditionally on each
dominant macro regime. No live allocation signal is produced here.

**Inputs:**
- `data/regime_signals.parquet` — monthly regime labels (output of 19b)
- `data/pyaiam_eod.csv` — daily price panel (AAPL, NVDA, JPM, SPY, GLD, TLT, EURUSD, BTC-USD)

**Output:** `data/derived/regime_portfolio_performance.parquet`

---
| Section | Content |
|---|---|
| 0 | Imports and setup |
| 1 | Load data |
| 2 | Portfolio method implementations |
| 3 | Walk-forward backtest |
| 4 | Regime-conditional performance |
| 5 | Sharpe heatmap (methods × regimes) |
| 6 | Summary table (best/worst per regime) |
| 7 | Cumulative return plots by regime |
| 8 | Export |
```

Imports:

```
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import yfinance as yf
```

Function definitions (all notebook-local — they re-implement what
`paam_lab.construction` already provides, instead of importing it):

- `def weight_ew(returns)` — `"Equal weight: 1/N for each asset."`
- `def weight_gmv(returns)` — `"Global minimum variance portfolio."`
- `def weight_msr(returns, rf=0.0)` — `"Maximum Sharpe ratio portfolio."`
- `def weight_mdp(returns)` — `"Most diversified portfolio: maximise diversification ratio."`
- `def weight_rp(returns)` — `"Risk parity: equal risk contribution from each asset."`
- `def weight_hrp(returns)` — `"Hierarchical Risk Parity (Lopez de Prado 2016, SSRN 2708678)."`
- `def run_backtest(log_ret_monthly, method_name, lookback=36)` — `"Walk-forward monthly backtest for one portfolio method."`
- `def perf_metrics(ret_series, rf_monthly=rf)` — `"Annualised performance metrics for a monthly return series. Returns dict: Ann Ret, Ann Vol, Sharpe, Max DD, N months."`

All six `weight_*` functions take a **returns DataFrame** (monthly
window, T×N) and return either a 1-D NumPy weight vector or a Series.
Internal `METHOD_FNS = {"EW", "GMV", "MSR", "MDP", "RP", "HRP"}`.

External library dependencies: scipy (`linkage`, `leaves_list`,
`minimize`, `squareform`), numpy, pandas, seaborn, matplotlib,
yfinance. No paam_lab imports anywhere.

Public API consumed downstream by 19d: `data/derived/regime_portfolio_performance.parquet`
(and the in-notebook `SWITCHING_RULE` dict, which is **transcribed** into
19d rather than imported).

#### `19d_regime_allocation_signal.ipynb` (24 cells)

Opening markdown:

```markdown
# 19d — Regime Allocation Signal

**Live allocation signal** that reads the dominant macro regime from `regime_signals.parquet`,
applies the regime-switching heuristic from 19c, and produces:
- Current portfolio weights (Section 9)
- Walk-forward backtest vs. 6 fixed methods + 60/40 benchmark
- Performance table and cumulative return chart

This notebook is a **pure analysis notebook** — it reads regime labels; it never regenerates them.
```

Imports:

```
from IPython.display import display
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings, yfinance as yf
```

Function definitions (same six weight_* + run_backtest as 19c, plus):

- `def regime_to_method(regime: int) -> str` — `"Return recommended portfolio method for a given dominant regime."` — returns the value from a hard-coded `SWITCHING_RULE` dict `{0: "EW", 1: "MDP", 2: "MDP", 3: "MDP", 4: "MDP", 5: "MSR", 6: "MDP", 7: "MDP"}`.
- `def run_switching_backtest(log_ret, dominant_regime, lookback=24)` — `"Walk-forward backtest of the regime-switching strategy. At each month t: 1. Look up dominant_regime[t-1] (prior month regime, known at t). 2. Select method via SWITCHING_RULE. 3. Fit weights on log_ret[t-lookback : t]. 4. Apply weights to log_ret[t]. Returns pd.Series of monthly log returns, pd.DataFrame of weights by date."`
- `def run_backtest(log_ret_monthly, method_name, lookback=24)` — identical to 19c with default `lookback=24`.
- `def perf_metrics(ret_series, rf_monthly=RF_MONTHLY)` — `"Ann Ret, Ann Vol, Sharpe, Max DD for a monthly log-return series."`

External library dependencies: same as 19c — scipy + matplotlib + numpy
+ pandas + yfinance. No paam_lab imports. No hmmlearn / sklearn /
statsmodels / cvxpy / riskfolio.

**Downstream API to get a regime label for a date** (cleanly stated): load
`regime_signals.parquet`, take the `dominant_regime` column (Int64, monthly
end-of-month index), `.loc[some_date]` to get an int in `[0, 7]`. That's
the entire interface 19c and 19d use against the engine.

### Macro cache

`~/Projects/ai_asset_management_lab/data/macro_cache/` exists. 17 parquet
files:

```
-rw-r--r--   1 frasagui  staff   8.7K  CPI_MoM.parquet
-rw-r--r--   1 frasagui  staff   4.6K  GDP_QoQ.parquet
-rw-r--r--   1 frasagui  staff   149K  SPX.parquet
-rw-r--r--   1 frasagui  staff   6.7K  UNEM.parquet
-rw-r--r--   1 frasagui  staff   112K  VIX.parquet
-rw-r--r--   1 frasagui  staff   7.8K  YC_10Y.parquet
-rw-r--r--   1 frasagui  staff   7.8K  YC_2Y.parquet
-rw-r--r--   1 frasagui  staff    83K  ext_GLD.parquet
-rw-r--r--   1 frasagui  staff   106K  ext_QQQ.parquet
-rw-r--r--   1 frasagui  staff   107K  ext_SPY.parquet
-rw-r--r--   1 frasagui  staff    97K  ext_TLT.parquet
-rw-r--r--   1 frasagui  staff    92K  port_EEM.parquet
-rw-r--r--   1 frasagui  staff    83K  port_GLD.parquet
-rw-r--r--   1 frasagui  staff    76K  port_HYG.parquet
-rw-r--r--   1 frasagui  staff    95K  port_QQQ.parquet
-rw-r--r--   1 frasagui  staff    95K  port_SPY.parquet
-rw-r--r--   1 frasagui  staff    95K  port_TLT.parquet
```

Grouped by prefix:

- **FRED / macro (no prefix):** `CPI_MoM` (8.7K), `GDP_QoQ` (4.6K), `SPX` (149K), `UNEM` (6.7K), `VIX` (112K) — five raw FRED-style monthly indicator series.
- **Yield-curve (`YC_`):** `YC_10Y` (7.8K), `YC_2Y` (7.8K) — used by 19b to derive `regime_YC10`, `regime_YC2`, `regime_YCSTEP`.
- **Extended yfinance ETFs (`ext_`):** `ext_GLD` (83K), `ext_QQQ` (106K), `ext_SPY` (107K), `ext_TLT` (97K) — long-history daily ETF panel back to 2000 for 19b's full-history regime conditioning.
- **Portfolio assets (`port_`):** `port_EEM` (92K), `port_GLD` (83K), `port_HYG` (76K), `port_QQQ` (95K), `port_SPY` (95K), `port_TLT` (95K) — six liquid ETFs back to 2003 used by 19c/19d as the backtest universe.

Additional non-cache parquet artefacts under `data/`:

```
/Users/frasagui/Projects/ai_asset_management_lab/data/regime_history.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/regime_signals.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/regime_portfolio_performance.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/switching_strategy_weights.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch08g_method_returns.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/switching_strategy_returns.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch10_handoff/empirical_returns_ready.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch10_handoff/empirical_features_ready.parquet
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch10_handoff/empirical_clean.parquet
```

### Handoff manifest

`find -name "manifest.json" -o -name "feature_panel.parquet" -o
-name "asset_list.*"`:

```
/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch10_handoff/manifest.json
```

Contents:

```json
{
  "artifact_names": {
    "empirical_clean": "empirical_clean.parquet",
    "empirical_features_ready": "empirical_features_ready.parquet",
    "empirical_returns_ready": "empirical_returns_ready.parquet",
    "manifest": "manifest.json"
  },
  "artifact_root": "/Users/frasagui/Projects/ai_asset_management_lab/data/derived/ch10_handoff",
  "asset_coverage": {
    "AAPL": 1.0, "BTC-USD": 1.0, "EURUSD": 0.9992047713717693,
    "GLD": 1.0, "JPM": 1.0, "NVDA": 1.0, "SPY": 1.0, "TLT": 1.0
  },
  "asset_list": [
    "AAPL", "NVDA", "JPM", "SPY", "GLD", "TLT", "EURUSD", "BTC-USD"
  ],
  "date_ranges": {
    "empirical_clean": { "start": "2015-11-30", "end": "2025-11-28" },
    "empirical_features_ready": { "start": "2015-12-02", "end": "2025-11-28" },
    "empirical_returns_ready": { "start": "2015-12-01", "end": "2025-11-28" }
  },
  "excluded_assets": [],
  "feature_list": [
    "log_ret_1d", "mom_5d", "mom_20d", "vol_20d", "z_20d", "cs_rank_mom_20d"
  ],
  "feature_storage_format": "tidy_long",
  "source_path": "/Users/frasagui/Projects/ai_asset_management_lab/data/pyaiam_eod.csv"
}
```

No `feature_panel.parquet` file exists — the feature panel artefact is
named `empirical_features_ready.parquet` and uses tidy-long storage
(`Date`, `asset`, plus six feature columns); the wide-MultiIndex panel
is reconstructed in-memory at load time by `load_ch10_handoff`. There
is no standalone `asset_list.*` file — the asset list is embedded in
the manifest.

---

## PART 2: NEW repo state

### `~/Projects/next-gen-aiam/CLAUDE.md`

```markdown
# CLAUDE.md

## Project
next-gen-aiam — comparative model harness for AI-driven asset management.
Common data panel (EODHD), uniform `Strategy` interface, side-by-side
comparison of classical methods, regime models, ML, DL, and RL on the
same data.

## Status
Scaffolding (May 2026). Environment, package skeleton, and git in place.
Architectural commitments locked in the handoff doc; v0 build begins next
session.

## Environment

    cd ~/Projects/next-gen-aiam
    source .venv/bin/activate

Reinstall if needed:

    pip install -r requirements.txt
    pip install -e ".[dev]"

Python: 3.14.0 on this machine.

## What exists today
- `src/aiam/` — empty package (just `__init__.py` with version)
- `pyproject.toml`, `requirements.txt` — installable, dev tooling
- `.gitignore`, `README.md`

## What's planned (v0, next session)
- `src/aiam/data/` — Panel, Universe, EODHD client + per-data-type modules
- `src/aiam/strategies/` — Strategy ABC + classical, regime, overlay subpackages
- `src/aiam/evaluation/` — performance_stats with corrected Sharpe
- `src/aiam/harness.py` — run_horse_race
- `tests/` — port from old repo
- `notebooks/01_data_and_universe.ipynb`, `notebooks/05_horse_race.ipynb`

## Non-negotiable rules
- Never commit secrets (EODHD API key reads from environment variable only)
- Never commit data files (`data/cache/` is gitignored)
- `nbstripout` to be configured before notebooks land in git

## Reference repos
- Old lab: `~/Projects/ai_asset_management_lab` — source of `paam_lab` code to port forward
- Hilpisch: github.com/yhilpisch/paamcode — book's official code

## Conventions
Full notebook/code conventions land next session with the Panel and Strategy
ABC. Until then: write defensively; ask before adding top-level files or new
top-level dependencies.
```

### `~/Projects/next-gen-aiam/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aiam"
version = "0.0.1"
description = "Comparative model harness for AI-driven asset management"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "frasagui" }]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "nbstripout>=0.7",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### `~/Projects/next-gen-aiam/.gitattributes`

```
*.ipynb filter=nbstripout
*.zpln filter=nbstripout
*.ipynb diff=ipynb
```

(Exists despite the CLAUDE.md note that nbstripout has yet to be configured —
the filter attribute is in place, but the corresponding git config
`filter.nbstripout.clean`/`smudge` may not be installed locally.)

### `~/Projects/next-gen-aiam/.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/
build/
dist/
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# IDE / OS
.vscode/
.idea/
.DS_Store

# Data cache — locally fetched EODHD parquet
data/cache/

# Secrets / env
.env
.envrc

# Logs
*.log
```

### `~/Projects/next-gen-aiam/requirements.txt`

```
# Core scientific stack
numpy>=1.26
pandas>=2.2
pyarrow>=15.0       # parquet I/O (Panel cache)

# Plotting
matplotlib>=3.8

# Notebook environment
jupyter>=1.0
ipykernel>=6.29

# HTTP (EODHD client, web fetches)
requests>=2.31
```

No scipy, no scikit-learn, no hmmlearn / statsmodels / cvxpy / riskfolio,
no yfinance / pandas_datareader. The OLD `paam_lab.construction` and
`paam_lab.covariance` modules will require adding **at minimum** `scipy`
(for `minimize`, `linkage`, `squareform`, `fcluster`) and `scikit-learn`
(for `LedoitWolf`, `OAS`) to ports cleanly.

### Tree

```
/Users/frasagui/Projects/next-gen-aiam
/Users/frasagui/Projects/next-gen-aiam/.claude
/Users/frasagui/Projects/next-gen-aiam/.claude/settings.local.json
/Users/frasagui/Projects/next-gen-aiam/.git
/Users/frasagui/Projects/next-gen-aiam/.gitattributes
/Users/frasagui/Projects/next-gen-aiam/.gitignore
/Users/frasagui/Projects/next-gen-aiam/.venv
/Users/frasagui/Projects/next-gen-aiam/CLAUDE.md
/Users/frasagui/Projects/next-gen-aiam/docs
/Users/frasagui/Projects/next-gen-aiam/docs/execution_packets
/Users/frasagui/Projects/next-gen-aiam/pyproject.toml
/Users/frasagui/Projects/next-gen-aiam/README.md
/Users/frasagui/Projects/next-gen-aiam/requirements.txt
/Users/frasagui/Projects/next-gen-aiam/src
/Users/frasagui/Projects/next-gen-aiam/src/aiam
/Users/frasagui/Projects/next-gen-aiam/src/aiam/__init__.py
```

No `tests/`, no `notebooks/`, no `data/`. `docs/execution_packets/` exists
but is empty (this report is the first thing landing in it).

### `src/aiam/__init__.py`

```python
__version__ = "0.0.1"
```

That is the whole file.

---

## PART 3: Architecture questions

### A. Strategy ABC shape

**Existing function shapes in paam_lab.construction.** Two distinct
patterns coexist:

1. **Inputs are pre-computed statistics, output is a weight vector.**
   `gmv_weights(Sigma)` (construction.py:64), `msr_weights(mu, Sigma,
   rf=0.03)` (construction.py:81), `mdp_weights(vols, Sigma)`
   (construction.py:111), `bl_weights(mu, Sigma, delta=2.5,
   long_only=False)` (construction.py:206), `bl_posterior_returns(pi,
   Sigma, P, Q, Omega, tau=0.05)` (construction.py:186),
   `solve_risk_budget(Sigma, budget=None, ...)` (construction.py:151),
   `rp_weights(Sigma, **kwargs)` (construction.py:179). The output is
   a bare `np.ndarray` with no asset names. Estimation of `mu`, `Sigma`,
   `vols` is the caller's job, done in the calling notebook from a sliced
   return window.

2. **Input is a returns DataFrame window, output is a weight Series.**
   `hrp_weights(returns)` (construction.py:347), `harp_weights(returns,
   lam=0.5)` (construction.py:366), `hap_weights(returns)`
   (construction.py:389), `hcp_weights(returns, method='ward',
   n_clusters=None)` (construction.py:398). Returns is `pd.DataFrame`
   indexed by date, columns by asset. Output is `pd.Series` indexed by
   asset.

**Neither pattern has an "as-of date" parameter.** Time is handled
externally by the caller (e.g., 19c/19d's `run_backtest` and
`run_switching_backtest`) which slice the returns window for each
rebalance date and call the weight function.

**State across time.** No paam_lab function carries state. The regime
engine's *fit* is encapsulated in `get_transition_proba(reg_hist)` (19b
cell 22) which rebuilds the transition matrix from the full available
history at each rebalance date; `get_exp_returns(..., rebalance_date,
...)` (19b cell 25) takes a `rebalance_date` and slices everything to
`:rebalance_date`. So the regime engine is **stateless at the function
level**, but stateful at the *artefact* level — `regime_signals.parquet`
is computed once and read by 19c/19d. The two layers of the existing
regime workflow are therefore:

- Heavy/expensive offline fit producing a monthly time series of regime
  labels (19b) → persisted to disk.
- Cheap per-step lookup at rebalance time (19c/19d) →
  `dominant_regime.loc[prev_dt]` then weight-fn dispatch via
  `SWITCHING_RULE`.

**Recommendation.** A single-tier ABC of the `predict_weights(asof) ->
pd.Series` form fits the *output* contract for everything in paam_lab
today (every constructor produces a weight vector keyed by asset; if
`np.ndarray` we just need to pair it with the strategy's asset universe).
But it does not cleanly support:

- The regime engine's "fit on history up to `asof`, then look up" split
  (19b's `get_exp_returns` semantics) — this is `predict_weights(asof,
  history)` rather than `predict_weights(asof)`, and you need access to
  the panel inside `predict_weights`.
- Future RL allocators which observe a state, take an action (weights),
  observe a reward, update parameters — the loop is
  `(observation_t, reward_t) → action_t`, and the agent has parameters
  that must persist across `t`.
- Future LLM/agent allocators which may carry plan / scratchpad /
  tool-call history across multiple rebalance dates.

Option (i), `recommend(context) -> Weights`, is **the safe choice
*provided* `context` is a typed object that can carry the panel slice,
the as-of date, the previous holdings, and per-strategy auxiliary inputs
(regime label, transition matrix, fitted-model artefacts).** The risk is
making it too loose — paam_lab's classical constructors should not need
to reach into `context` for anything beyond `panel[:asof]`.

Option (ii), `PointInTimeStrategy + SequentialStrategy`, maps cleanly
onto what exists:

- `PointInTimeStrategy.predict_weights(panel, asof) -> pd.Series` covers
  every classical paam_lab constructor (the strategy slices the panel
  internally, computes its own `mu/Sigma/vols`, and returns weights). It
  also covers the regime engine if we treat `regime_signals.parquet` as
  external precomputed input bolted on at construction time (the strategy
  reads `dominant_regime.loc[asof]` and dispatches to a classical method
  via the switching rule). This is exactly what 19d's
  `run_switching_backtest` already does inline.
- `SequentialStrategy.step(observation_t, reward_{t-1}) -> action_t`
  covers RL and the LLM-agent case. State lives on `self`.
- A shared base ABC defines `assets`, `name`, and a hook
  `fit(panel, train_until=asof)` so both flavours can warm up before
  going live; `PointInTimeStrategy.fit` is a no-op for stateless
  constructors and "rebuild transition matrix" for the regime strategy;
  `SequentialStrategy.fit` is offline pre-training for an RL agent.

Reasoning grounded in the code:
- The `(returns) -> pd.Series` hierarchical constructors at
  `construction.py:347`, `:366`, `:389`, `:398` already operate as
  "give me a window of data, I'll give you weights" — they fit
  `PointInTimeStrategy.predict_weights(panel.loc[:asof].tail(window))`
  with zero re-shaping.
- The `(mu, Sigma)` vector constructors at `construction.py:64`, `:81`,
  `:111`, `:206`, etc. require an estimator step that paam_lab leaves
  to the caller. Wrapping those as `PointInTimeStrategy` subclasses with
  a `covariance_estimator: Callable[[pd.DataFrame], np.ndarray]`
  attribute (defaulting to `ledoit_wolf_cov` from covariance.py:17)
  keeps the existing functions intact while moving the estimator
  selection from "notebook inline code" to "strategy ctor argument."
- The regime workflow in 19d cell 12 (`run_switching_backtest`) is
  effectively a `PointInTimeStrategy` whose `predict_weights` reads the
  pre-computed `dominant_regime` series, picks a method from a dict, and
  delegates to a classical-strategy instance. That maps directly onto a
  `RegimeConditionalStrategy(PointInTimeStrategy)` composed of one
  inner strategy per regime label.
- RL/LLM agents do not exist in OLD code, but the SequentialStrategy
  contract is the standard
  `(obs, reward) → action` loop and trivially supports them.

**Recommended choice: (ii) — a thin two-class hierarchy.** Keeping the
two classes separate avoids overloading a single `recommend(context)`
with an `if self.is_sequential` switch that paam_lab callers would have
to thread through anyway. The hierarchy stays small (one ABC, two
subclasses) and is honest about the fact that classical constructors are
memoryless while RL agents are not.

### B. Panel shape

**What paam_lab passes today.** Three distinct shapes are in flight:

- **Wide DataFrame, prices.** `clean_price_panel(panel: pd.DataFrame) ->
  tuple[pd.DataFrame, pd.DataFrame]` at `data_quality.py:207`. Rows are
  dates (`DatetimeIndex`), columns are assets, values are prices. This is
  the input to feature engineering and the canonical "raw" panel.
- **Wide DataFrame, returns.** Throughout `construction.py` —
  `hrp_weights(returns)` at `construction.py:347` takes a `pd.DataFrame`
  with columns = assets, rows = dates (typically pre-sliced to a fitting
  window). `covariance.empirical_cov(returns: pd.DataFrame) ->
  np.ndarray` at `covariance.py:12` accepts the same shape.
- **Wide DataFrame with two-level MultiIndex columns, features.**
  `build_feature_panel(clean_prices: pd.DataFrame) -> pd.DataFrame` at
  `features.py:18` returns a frame whose `.columns` is a
  `MultiIndex.from_product` of `(feature, asset)`. Six features:
  `log_ret_1d, mom_5d, mom_20d, vol_20d, z_20d, cs_rank_mom_20d`.

A representative signature is the round-trip pair from `io/handoff.py`
and `features.py`:

```python
# features.py:18
def build_feature_panel(clean_prices: pd.DataFrame) -> pd.DataFrame: ...
    # returns: rows = Date, columns = MultiIndex (feature, asset)
```

Persisted to disk in tidy-long form (`Date`, `asset`, plus one column
per feature), then reconstructed wide on load by `load_ch10_handoff`
(`handoff.py:131`):

```python
empirical_features_panel = empirical_features_long.set_index(
    ["Date", "asset"]
)[manifest["feature_list"]].unstack("asset")
```

There is no custom dataclass anywhere; everything is plain DataFrames.
The mixed-frequency case (monthly regime labels in
`regime_signals.parquet`, daily prices in `pyaiam_eod.csv`) is handled
ad-hoc in 19d cell 7 by `resample("M").last()` then
`.intersection(dominant_regime.index)`.

**Smallest-change Panel shape.** The smallest delta that preserves
existing code with minor wrapping and accepts mixed-frequency
extensions:

- A `Panel` thin wrapper that holds **a dict of wide DataFrames keyed by
  data type** (`prices`, `returns`, `features` (wide MultiIndex on
  columns), `macro` (lower frequency), `regimes` (monthly), `news`,
  `fundamentals`, ...), plus a single canonical asset universe and a
  master DatetimeIndex.
- Each value DataFrame has rows = its native index (daily, monthly, …)
  and columns = either the asset universe (for asset-aligned data) or a
  flat string column index (for cross-asset macro data like `VIX`,
  `GDP_QoQ`).
- A method `.slice(asof: pd.Timestamp, *, kind: str, lookback: int |
  None = None, freq: str | None = None) -> pd.DataFrame` returns a wide
  DataFrame of the requested kind, reindexed to whatever frequency the
  caller asks for, with the no-look-ahead invariant enforced (`asof`
  inclusive, anything after it dropped). Existing constructors take
  this output without modification: `gmv_weights(empirical_cov(
  panel.slice(asof, kind='returns', lookback=252)))`.
- Mixed frequency: each frame keeps its native frequency; the slice
  method handles forward-fill, `resample('M').last()`, or
  `.asof(asof)`-style alignment based on the requested target. This is
  what 19d already does inline.

This shape is small enough to add without rewriting paam_lab functions —
they keep their `(returns: pd.DataFrame)` signatures — and big enough to
hold news / fundamentals / macro alongside returns without a schema
break.

The two MultiIndex options (rows MultiIndex `(Date, asset)` vs columns
MultiIndex `(feature, asset)`) both exist in paam_lab already (long
storage on disk, wide compute in memory). The wrapper above keeps both
forms available behind a single API.

### C. Sharpe bug

`evaluation/common.py:14-42` defines `performance_stats`. The buggy
lines are below, quoted verbatim with line numbers:

```python
14	def performance_stats(simple_returns):
15	    simple_returns = pd.Series(simple_returns, dtype="float64").dropna()
16	    if simple_returns.empty:
...
28	    wealth = (1.0 + simple_returns).cumprod()
29	    annualized_return = wealth.iloc[-1] ** (TRADING_DAYS / len(simple_returns)) - 1.0
30	    annualized_volatility = simple_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
31	    sharpe = np.nan if annualized_volatility == 0 else annualized_return / annualized_volatility
```

**The bug is line 31.** `sharpe = annualized_return / annualized_volatility`
omits the risk-free rate entirely. The correct formula, given the rest
of paam_lab uses `rf = 0.03`, is:

```python
RF = 0.03
sharpe = (annualized_return - RF) / annualized_volatility
```

or, parameterised:

```python
def performance_stats(simple_returns, rf: float = 0.03): ...
sharpe = (annualized_return - rf) / annualized_volatility
```

For comparison, `construction.sharpe(w, mu, Sigma, rf=0.03)` at
`construction.py:38-40` does it correctly:

```python
38	def sharpe(w, mu, Sigma, rf: float = 0.03) -> float:
39	    """Annualised Sharpe ratio. Canonical form from 08d."""
40	    return (port_ret(w, mu) - rf) / port_vol(w, Sigma)
```

And `summarize_strategy` at `evaluation/common.py:138`:

```python
138	    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
```

— shares the **same bug** as `performance_stats` (also drops `rf`). The
notebook-local `perf_metrics` in 19c/19d does subtract `rf` (`Sharpe =
(ann_ret - rf_monthly * 12) / (ann_vol + 1e-8)`), so the bug is
exclusively in the paam_lab evaluation layer.

**Blast radius.** Every callsite of the two buggy functions in `src/`:

- `evaluation/portfolio.py:108-110` — `build_portfolio_proxy` calls
  `performance_stats(strategy_gross_simple)`, `performance_stats(
  strategy_net_simple)`, `performance_stats(benchmark_simple)` for the
  ML portfolio-proxy summary.
- `evaluation/portfolio.py:150-151` — `build_rank_proxy_bundle` calls
  `performance_stats(strategy_simple)`,
  `performance_stats(benchmark_simple)`.
- `evaluation/__init__.py:6` — re-exports `performance_stats` (any
  notebook that does `from paam_lab.evaluation import performance_stats`
  is affected).
- `evaluation/common.py:138` — `summarize_strategy` (separate function,
  same bug pattern).

No other `src/paam_lab/` module imports `performance_stats`. Test
exposure: `test_evaluation_common.py` only checks
`stats["max drawdown"] >= 0` (line 233) — it does **not** assert any
golden Sharpe value, so the bug is not caught by the test suite.

Outside `src/`, any notebook that imports `performance_stats` or
`summarize_strategy` reports a Sharpe that is high by `rf /
annualized_volatility` (about 0.18 at `vol = 0.16`, i.e. quoted Sharpes
are inflated by ~0.18 versus the correct value). The two existing
golden tests in `test_overlay.py::test_msr_vmp_sharpe_golden` and
`test_evaluation_common.py::test_golden_rp_baseline_from_09` compute
Sharpe **inline** in the test (`(ann_ret - RF) / ann_vol`), bypassing
`performance_stats` entirely — which is why those goldens are
internally consistent despite the library bug.

When fixing in the port: a single signature change to
`performance_stats(returns, rf=0.03)` plus a one-line edit to
`summarize_strategy` is enough; `evaluation/portfolio.py` callers all
pass a single series and accept defaults.

### D. Regime engine port surface

Wrapping 19b/c/d as a single `RegimeConditionalStrategy(PointInTimeStrategy)`
subclass:

**Inputs at fit time:**
- A wide returns DataFrame for the asset universe (e.g. `port_*` ETFs at
  monthly frequency; the OLD code uses 24- or 36-month lookbacks).
- The eight indicator series from `data/macro_cache/` (`CPI_MoM`,
  `GDP_QoQ`, `SPX`, `UNEM`, `VIX`, `YC_10Y`, `YC_2Y`, plus the derived
  `YCSTEP = YC_10Y - YC_2Y`). FRED+yfinance pulls in 19b are replaced
  by reads from the cache directory.
- A 5-year rolling window for the `mean_lvl` term inside `get_regime`
  (19b cell 14).
- The switching rule mapping `{0: "EW", 1: "MDP", 2: "MDP", 3: "MDP",
  4: "MDP", 5: "MSR", 6: "MDP", 7: "MDP"}` (19d cell 10) — six classical
  paam_lab constructors keyed by regime label.

**State the strategy must carry:**
- The per-indicator monthly regime time series `regime_GDP, …,
  regime_YCSTEP` (eight Int64 series).
- The dominant regime time series (single Int64 series, monthly).
- Per-indicator first-order Markov transition matrices (eight 8×8
  matrices), refreshed at each rebalance from history up to `asof` —
  `get_transition_proba` is fast (single numpy loop), so this can be
  rebuilt online rather than persisted.
- The configured asset universe and the lookback window (24 months in
  19d, 36 months in 19c).
- An inner-strategy table keyed by classical method name (`EW`, `GMV`,
  `MSR`, `MDP`, `RP`, `HRP`) → instance of the corresponding
  `PointInTimeStrategy` subclass.

**Inputs at predict time:**
- `asof: pd.Timestamp` (a month-end).
- The panel (or a slice via `panel.slice(asof, kind='returns',
  lookback=self.lookback, freq='M')`).

**Predict-time logic (verbatim from 19d cells 10–12):**
1. `current_regime = int(self.dominant_regime.loc[prev_month_end(asof)])`
   (use prior-month label to avoid look-ahead — 19d cell 12 lines
   "prev_dt", "regime known at prev month-end").
2. `method = self.switching_rule.get(current_regime, "MDP")`.
3. `window = panel.slice(asof, kind='returns_monthly', lookback=self.lookback)`.
4. `w = self.inner[method].predict_weights(window, asof)`.
5. Return `pd.Series(w, index=window.columns, name=asof)`.

This is essentially a thin shim: the regime engine itself is the
*offline* piece producing `dominant_regime`; the runtime
`RegimeConditionalStrategy` is just a dispatch table.

There are two viable factorings:

- **A. One class, ingests precomputed `dominant_regime`.** The strategy
  is constructed with a `regime_signals: pd.DataFrame` argument and only
  knows how to dispatch. 19b is run separately (offline notebook /
  CLI / pipeline) and produces the signals file. Smallest port surface,
  preserves the OLD 19b/c/d layered design.
- **B. One class that owns the whole pipeline.** `fit(panel,
  macro_indicators)` runs `compute_features` + `get_regime` per
  indicator + `get_transition_proba` + `get_exp_returns` + `combine_exp_returns`
  internally. Larger port surface — `compute_features`, `get_regime`,
  `get_transition_proba`, `get_exp_returns`, `run_regime_model`,
  `combine_exp_returns` all become methods/helpers on the class — but
  removes the offline-step requirement.

Option A is the smaller commit and matches OLD's actual data flow;
option B fits better if the broader `aiam` design wants every strategy to
be runnable in a single `fit()` call.

### E. Test rewrite scope

Grouping by how the existing test files would fare under a refactor to
`aiam.strategies.X().fit(panel).predict_weights(asof)`:

**Light edits — port largely as-is, only re-namespace imports and re-base
fixture paths.**

- `test_covariance.py` — Tests cover the four covariance estimators
  with no strategy interface. Pure `(returns: pd.DataFrame) -> ndarray`
  functions; if `aiam.data.estimators` (or wherever covariance lands)
  keeps the same function signature, these tests survive a search-and-replace
  on the imports + a small change to load the EODHD panel instead of
  `data/pyaiam_eod.csv`.
- `test_features.py` — `build_feature_panel(clean_prices)` is shape-free
  and as-of-free; same story as `test_covariance.py`. The
  `test_build_feature_panel_round_trip` test depends on
  `load_feature_panel_artifact` which will need to be ported alongside
  the function under whatever the new I/O namespace is.
- `test_overlay.py` — `vmp_overlay_lagged`, `rolling_annualized_vol`,
  `drawdown_multiplier`, `lagged_exposure`, `apply_overlay`,
  `vol_target_multiplier` are all pure pandas helpers with no strategy
  interface and survive as utility-function tests. The golden
  end-to-end test reads `data/derived/ch08g_method_returns.parquet` —
  if `ch08g_method_returns.parquet` is replicated (or replaced by an
  equivalent strategy-output artefact) the golden Sharpe number (1.3432)
  may or may not survive depending on whether the new MSR strategy
  produces bit-identical returns on the same input.
- `test_data_quality.py` — All 27 tests run against the toy fixture
  panels in `conftest.py`. Zero strategy interface. Lift-and-shift.
- `test_evaluation_common.py` regression block (TestExistingFunctions,
  TestEquityFromReturns, TestDrawdownFromEquity, TestMaxDrawdown,
  TestHitRatio) — pure return-series helpers. Lift-and-shift, with the
  Sharpe bug fix triggering a new (correctly-valued) golden assertion
  that will need to be added.

**Light edits, but goldens may move.**

- `test_construction.py` (where it covers the simple `(Sigma) -> w` and
  `(mu, Sigma) -> w` constructors) — under the new ABC each of these
  is wrapped as a `PointInTimeStrategy` subclass that internally calls
  the same numerical routine. The unit-level test (the numerical
  routine, called directly) survives if we keep the routine as a public
  function inside the strategies subpackage. If the only public API is
  `strategy.fit(panel).predict_weights(asof)`, then every test in
  TestGmvWeights, TestMsrWeights, TestEmvWeights, TestMdpWeights,
  TestBlPosteriorReturns, TestBlWeights, TestRpWeights, TestHrpWeights,
  TestHarpWeights, TestHapWeights, TestHcpWeights needs a rewrite of
  the form:
  ```python
  strat = aiam.strategies.classical.GMV(covariance_estimator=ledoit_wolf_cov)
  panel = Panel(...)
  w = strat.fit(panel, train_until=asof).predict_weights(asof)
  assert np.isclose(w["SPY"], 1.0, atol=1e-4)
  ```
  Mechanical but not a "light edit."

**Full rewrites.**

- `test_construction.py` private-helper tests (TestToArray, TestQuasiDiag,
  TestClusterVar, TestBisect, TestLogisticScale, TestHarpBisect,
  TestCorrDistMatrix, TestProjectToSimplex, TestSolveRiskBudget,
  TestPortVol, TestPortRet, TestSharpe, TestPctRc, TestDivRatio,
  TestEwWeights) — these test functions that may no longer be public
  (`_to_array`, `_quasi_diag`, `_bisect`, etc.) or may be inlined into a
  strategy class. Each becomes either an internal pytest of the new
  helper module or is dropped if the helper is no longer reachable.
  Specifically:
    - The `_to_array`, `_quasi_diag`, `_cluster_var`, `_bisect`,
      `_logistic_scale`, `_harp_bisect` tests probably get dropped (they
      test private helpers that aren't load-bearing if we trust the
      end-to-end goldens).
    - `port_vol`, `port_ret`, `sharpe`, `pct_rc`, `div_ratio`,
      `project_to_simplex`, `solve_risk_budget` are likely to remain as
      utility functions in a `strategies/_portfolio_math.py`; their tests
      survive with renamed imports.
- The Sharpe golden numbers in TestSharpe (`port_vol(w_ew, Sigma_emp)
  ≈ 0.2535`, `sharpe(w_ew, ann_mu, Sigma_emp, rf=0.03) ≈ 1.1820`) come
  from the OLD `pyaiam_eod.csv` panel. If the NEW EODHD data is used to
  rebuild a comparable basket, these numbers may not match byte-for-byte
  because of adjustments / dividend handling differences between Yahoo
  (OLD `pyaiam_eod.csv`) and EODHD. Treat these as "to be re-locked
  after the first EODHD pull."

**No tests today, needs ground-up coverage.**

- The regime engine (19b/c/d) has zero test coverage. Porting it to a
  `RegimeConditionalStrategy` is a green-field testing exercise: schema
  of `regime_signals.parquet`, no-look-ahead invariant on
  `dominant_regime.loc[prev_dt]`, the per-regime → method dispatch, the
  `get_exp_returns` persistence-shortcut branch.
- `evaluation/portfolio.py` (`build_portfolio_proxy`,
  `build_rank_proxy_bundle`, `scores_to_rank_long_only_weights`) has
  zero test coverage.
- `ml/common.py` (`build_target_panel`, `build_prediction_frame`, the
  metrics-frame builders, the chronological-splits helper) has zero
  test coverage.
- `ml/linear.py` (`fit_ridge_closed_form`, `predict_ridge_closed_form`) has
  zero test coverage.

### F. VMP overlay

`paam_lab.overlay.vmp_overlay_lagged` at `overlay.py:94-118` (cited
verbatim in Part 1, "Source files / overlay.py"):

```python
94	def vmp_overlay_lagged(
95	    portfolio_returns: pd.Series,
96	    target_vol: float = 0.12,
97	    window: int = 21,
98	    lam_min: float = 0.25,
99	    lam_max: float = 1.5,
100	) -> tuple[pd.Series, pd.Series]:
101	    """VMP overlay with explicit one-day lag to avoid look-ahead bias.
102	
103	    Canonical form from 09b §4 (Moreira & Muir 2017, *Journal of Finance*
104	    72(4), 1611–1644).  Returns (scaled_returns, exposure_multiplier).
105	
106	    λ_t = clip(σ* / σ̂_{t-1}, lam_min, lam_max)
107	    scaled_returns_t = portfolio_returns_t × λ_t
108	
109	    Note: uses ddof=1 (pandas default std) per 09b source, consistent
110	    with the published Moreira & Muir formula.  This differs from
111	    rolling_annualized_vol which uses ddof=0 per the 09 toy example.
112	    """
113	    realized_vol = portfolio_returns.rolling(window).std() * np.sqrt(ANNUALIZATION)
114	    raw_lambda = target_vol / realized_vol.clip(lower=1e-8)
115	    lam = raw_lambda.clip(lower=lam_min, upper=lam_max)
116	    lam_lagged = lam.shift(1)
117	    scaled = (portfolio_returns * lam_lagged).dropna()
118	    return scaled, lam_lagged.reindex(scaled.index)
119	```

**What this is.** Moreira-Muir-style volatility-managed portfolio
overlay: the exposure multiplier `λ_t = clip(σ_target / σ̂_{t-1},
lam_min, lam_max)` scales gross exposure inversely with the prior
period's realised volatility, then clips to a band. The defaults
(`target_vol=0.12`, `lam_min=0.25`, `lam_max=1.5`) are not the
unconstrained Moreira-Muir form (which would scale freely by `1/σ_t`);
the [0.25, 1.5] clip box is a practitioner adjustment from 09b to bound
turnover and leverage. The "lagged" suffix refers to the explicit
one-day shift on line 116, which makes the overlay computable in
production at the close of `t-1`.

So: yes, it is volatility-managed in the Moreira-Muir sense (scaling by
1/σ_t), but with two practitioner adjustments — (a) a `[lam_min,
lam_max]` clip band, and (b) an explicit one-period lag on the
multiplier. It is **not** a variance-targeting scheme (which would
target a fixed annualised variance and tend to scale by 1/σ_t² for
normal returns); the scaling exponent here is 1, not 2.

The non-lagged variant `apply_overlay(base_returns, exposure_signal)`
at `overlay.py:80-91` lags the signal internally too; `lagged_exposure`
at `overlay.py:71-77` is the underlying primitive.

---

## PART 4: Honest gaps

### Expected from the handoff doc but not found
- **EODHD client.** CLAUDE.md plans `src/aiam/data/` with "Panel,
  Universe, EODHD client + per-data-type modules". Nothing of that
  lands today — `src/aiam/` is the single line `__version__ = "0.0.1"`.
- **Strategy ABC.** Planned. Not present.
- **`harness.py` / `run_horse_race`.** Planned. Not present.
- **`tests/` directory.** Planned. Does not exist in NEW.
- **`notebooks/01_data_and_universe.ipynb` and `notebooks/05_horse_race.ipynb`.**
  Planned. The whole `notebooks/` directory is absent.
- **`nbstripout`-configured.** CLAUDE.md notes it must be configured
  before notebooks land. `.gitattributes` already references the
  `nbstripout` filter — but the corresponding git-config side
  (`filter.nbstripout.clean` / `smudge`) is not verifiable from the
  filesystem, and the lack of any `.ipynb` files in NEW means it has
  not yet been exercised.
- **`docs/PENDING.md`.** Referenced by OLD CLAUDE.md as the canonical
  pending-work tracker for the OLD repo. The NEW repo has no
  `docs/PENDING.md` equivalent — only an empty `docs/execution_packets/`.

### Found in OLD that the NEW handoff doc didn't mention
- **`paam_lab.ml`** (`ml/common.py`, `ml/linear.py`) — full ML
  workflow scaffolding (chronological splits, standardiser, design
  matrix builder, target panel builder, prediction-frame builder, two
  metrics-frame builders, ridge closed-form). The NEW handoff lists
  "ML, DL, RL" as comparison targets but doesn't call out that paam_lab
  already has the ML common harness pieces ready to port.
- **`paam_lab.evaluation.portfolio`** — `build_portfolio_proxy` and
  `build_rank_proxy_bundle` produce wealth/turnover/transaction-cost
  bundles from ML prediction frames with a 10-bp default cost. This is
  the bridge between ML predictions and a portfolio backtest, and
  belongs in any future `evaluation/` namespace.
- **`paam_lab.io.handoff`** — Chapter 10 handoff loader is a fully
  worked example of a Panel-loader contract (manifest.json + three
  parquet files: clean prices, returns ready, features tidy long). The
  NEW handoff doc's `Panel` planning section should consider whether to
  inherit this manifest schema as the EODHD cache contract.
- **Six covariance estimators / 10+ portfolio constructors / VMP /
  drawdown overlay / Black-Litterman with full view-matrix support / six
  hierarchical methods (HRP, HARP, HAP, HCP, plus _bisect / _harp_bisect
  helpers).** The OLD `construction.py` is 420 lines and covers more
  ground than "classical methods" suggests at first glance — in
  particular Black-Litterman view construction is exposed via
  `bl_posterior_returns(pi, Sigma, P, Q, Omega, tau)`, ready for a
  Strategy subclass that takes view inputs at construction time.
- **`data_quality.py` (325 lines).** Validation, missingness summary,
  duplicate-date detection, stale-segment detection, extreme-move /
  split-like flagging, clean-and-audit, before/after report,
  matplotlib overlay. This is a complete data-QA module the NEW handoff
  doesn't mention. EODHD data will need exactly this kind of QA layer
  before any strategy fit.
- **`features.py`.** The six-feature cross-sectional panel
  (`log_ret_1d, mom_5d, mom_20d, vol_20d, z_20d, cs_rank_mom_20d`) is
  one fixed recipe today. NEW handoff says nothing about whether to
  port that recipe forward or replace it with a richer feature set.
- **Regime engine `data/macro_cache/` payload (17 parquets, ~1.2 MB).**
  Five FRED indicators + two yield-curve series + four extended ETF
  histories + six portfolio ETF histories — the regime engine's
  reproducibility hinges on this being available. The NEW handoff
  plans "EODHD as the common data panel" but does not specify how the
  macro indicators (which EODHD does not cover natively for FRED) get
  re-hosted.
- **`bootstrap.py`.** `discover_repo_root` / `ensure_src_on_path`
  notebook helpers. Tiny but every notebook calls them; NEW will need
  its own variant unless notebooks always run from a clean `.venv` with
  `pip install -e .`.

### Worth porting that the handoff omitted
- **The Sharpe-rf bug fix first.** Calling it out explicitly so the new
  `aiam.evaluation.performance_stats` lands correct, with a golden test
  that asserts the rf-aware value. The OLD test suite never asserted a
  Sharpe-from-performance_stats golden, so the bug is invisible to its
  test layer.
- **Two-tier ABC (PointInTime + Sequential).** Section A above. Worth
  promoting from "good idea" to "documented in CLAUDE.md before v0
  starts."
- **Pre-computed monthly regime label file as the regime-strategy
  contract** (Section D, option A). The OLD code's layered design (19b
  produces `regime_signals.parquet`, 19c/19d consume) is genuinely
  good — preserving it as a separate offline step keeps
  `RegimeConditionalStrategy` small at runtime.
- **`paam_lab.evaluation.portfolio.build_portfolio_proxy`'s
  transaction-cost-from-turnover hook** (10 bp default). The NEW
  handoff doesn't mention transaction costs; if the horse race is
  meant to be honest, the cost layer needs to be wired in from day
  one.
- **The 19c/19d notebook-local re-implementations of EW/GMV/MSR/MDP/RP/HRP
  are duplicates of paam_lab.construction.** Those notebook-local
  copies should be replaced by paam_lab imports as part of the port;
  carrying them forward into NEW unchanged would re-create the same
  duplication.

Report written to: ~/Projects/next-gen-aiam/docs/execution_packets/repo_review_for_claude_chat.md
