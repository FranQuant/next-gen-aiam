"""Walk-forward BL-views backtest for five Black-Litterman view generators.

Usage (offline validation):
    python scripts/run_bl_views_backtest.py --generator llm-anthropic --mock --max-dates 3
    python scripts/run_bl_views_backtest.py --generator momentum --max-dates 3
    python scripts/run_bl_views_backtest.py --generator equilibrium --max-dates 3

Live run (after API keys are set in .env):
    python scripts/run_bl_views_backtest.py --generator llm-anthropic
    python scripts/run_bl_views_backtest.py --generator llm-openai
    python scripts/run_bl_views_backtest.py --generator momentum
    python scripts/run_bl_views_backtest.py --generator mean_rev
    python scripts/run_bl_views_backtest.py --generator equilibrium
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiam.data.panel import Panel
from aiam.data.split import TEST_START
from aiam.dl.walkforward import generate_refit_dates
from aiam.estimators.covariance import ledoit_wolf_cov
from aiam.estimators.views import equilibrium_only, mean_reversion_views, momentum_views
from aiam.evaluation.performance import performance_stats
from aiam.llm.cache import PromptCache
from aiam.llm.client import AnthropicClient, MockClient, OpenAIClient
from aiam.llm.views import LLMViewGenerator
from aiam.strategy.black_litterman import BlackLitterman

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRICES_PATH = Path("data/cache/prices_29.parquet")
RESULTS_DIR = Path("results/llm")

# Fixed canned ViewSet for --mock validation (assets from the 29-asset universe)
_CANNED_RESPONSE = json.dumps({
    "views": [
        {"asset": "SPY.US",  "expected_excess_return":  0.08, "confidence": 0.70},
        {"asset": "IEF.US",  "expected_excess_return": -0.02, "confidence": 0.60},
        {"asset": "GLD.US",  "expected_excess_return":  0.05, "confidence": 0.50},
        {"asset": "NVDA.US", "expected_excess_return":  0.12, "confidence": 0.65},
        {"asset": "TLT.US",  "expected_excess_return": -0.03, "confidence": 0.55},
    ],
    "rationale": "mock: bullish US equities and gold, bearish duration",
})


class _ViewTracker:
    """Wraps a view generator callable to record per-refit view counts."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.view_counts: list[int] = []

    def __call__(
        self,
        returns: pd.DataFrame,
        asof: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P, Q, Omega = self._inner(returns, asof)
        self.view_counts.append(int(len(Q)))
        return P, Q, Omega


def _make_tracked_view_gen(generator_name: str, mock: bool) -> _ViewTracker:
    """Return a _ViewTracker wrapping the appropriate generator."""
    if generator_name == "equilibrium":
        inner = equilibrium_only
    elif generator_name == "momentum":
        inner = momentum_views
    elif generator_name == "mean_rev":
        inner = mean_reversion_views
    elif generator_name in ("llm-anthropic", "llm-openai"):
        if mock:
            client: object = MockClient([_CANNED_RESPONSE])
            cache = PromptCache(disabled=True)
        else:
            cache = PromptCache()
            if generator_name == "llm-anthropic":
                client = AnthropicClient(model="claude-opus-4-7")
            else:
                client = OpenAIClient(model="gpt-5.5")
        inner = LLMViewGenerator(client, cache=cache)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown generator: {generator_name!r}")
    return _ViewTracker(inner)


def run_backtest(
    generator_name: str,
    *,
    mock: bool = False,
    max_dates: int | None = None,
) -> dict:
    """Walk the monthly refit grid, compute BL weights, accrue daily OOS returns.

    Returns:
        dict with keys:
            returns_series  – daily pd.Series of portfolio returns
            weights_frame   – pd.DataFrame, index=refit_dates, columns=assets
            diagnostics     – dict of summary statistics
    """
    logger.info("Loading prices from %s", PRICES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    prices = prices[prices.index.dayofweek < 5]

    panel = Panel({"prices": prices})
    returns = prices.pct_change().dropna(how="all")

    test_end = returns.index[-1]
    calendar = returns.index

    refit_dates = generate_refit_dates(
        TEST_START, test_end, cadence="monthly", calendar=calendar
    )
    logger.info("Total monthly refit dates: %d (OOS %s → %s)", len(refit_dates), TEST_START.date(), test_end.date())

    if max_dates is not None:
        refit_dates = refit_dates[:max_dates]
        logger.info("Capped to first %d refit dates", len(refit_dates))

    tracker = _make_tracked_view_gen(generator_name, mock)
    strategy = BlackLitterman(
        view_generator=tracker,
        cov_estimator=ledoit_wolf_cov,
        lookback=252,
        tau=0.05,
        delta=2.5,
        long_only=True,
    )

    # Compute BL weights at each refit date
    refit_weights: dict[pd.Timestamp, pd.Series] = {}
    for i, rd in enumerate(refit_dates):
        logger.info("Refit %d/%d  asof=%s", i + 1, len(refit_dates), rd.date())
        w = strategy.predict_weights(panel, asof=rd)
        refit_weights[rd] = w

    sorted_refits = sorted(refit_weights.keys())

    # Accrue daily portfolio returns over the full OOS window.
    # At each trading day t, apply weights from the most-recent refit <= t
    # to the NEXT day's returns (standard t → t+1 lag, no look-ahead).
    oos_days = returns.index[returns.index >= TEST_START]
    portfolio_returns: dict[pd.Timestamp, float] = {}

    for t in oos_days:
        idx = bisect.bisect_right(sorted_refits, t) - 1
        if idx < 0:
            continue
        w = refit_weights[sorted_refits[idx]]
        next_days = returns.index[returns.index > t]
        if next_days.empty:
            break
        t1 = next_days[0]
        ret_t1 = returns.loc[t1]
        aligned = w.reindex(ret_t1.index, fill_value=0.0)
        portfolio_returns[t1] = float(aligned.dot(ret_t1.fillna(0.0)))

    returns_series = pd.Series(portfolio_returns, name="portfolio_return")
    weights_frame = pd.DataFrame(refit_weights).T
    weights_frame.index.name = "refit_date"

    # Diagnostics
    stats = performance_stats(returns_series.dropna())

    turnover_vals: list[float] = []
    sorted_w = [refit_weights[rd] for rd in sorted_refits]
    for prev_w, cur_w in zip(sorted_w[:-1], sorted_w[1:]):
        prev_aligned = prev_w.reindex(cur_w.index, fill_value=0.0)
        turnover_vals.append(float((cur_w - prev_aligned).abs().sum() / 2.0))

    def _safe(v: float) -> float | None:
        return None if not np.isfinite(v) else float(v)

    diagnostics = {
        "generator": generator_name,
        "mock": mock,
        "n_refits": len(sorted_refits),
        "n_oos_days": int(returns_series.notna().sum()),
        "mean_views_per_refit": _safe(float(np.mean(tracker.view_counts))) if tracker.view_counts else None,
        "mean_turnover_per_refit": _safe(float(np.mean(turnover_vals))) if turnover_vals else None,
        "sharpe_ratio": _safe(stats["sharpe_ratio"]),
        "annualized_return": _safe(stats["annualized_return"]),
        "annualized_volatility": _safe(stats["annualized_volatility"]),
        "max_drawdown": _safe(stats["max_drawdown"]),
    }

    return {
        "returns_series": returns_series,
        "weights_frame": weights_frame,
        "diagnostics": diagnostics,
    }


def save_results(generator_name: str, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = RESULTS_DIR / f"{generator_name}_returns.parquet"
    result["returns_series"].to_frame().to_parquet(returns_path)
    logger.info("Returns   → %s", returns_path)

    weights_path = RESULTS_DIR / f"{generator_name}_weights.parquet"
    result["weights_frame"].to_parquet(weights_path)
    logger.info("Weights   → %s", weights_path)

    diag_path = RESULTS_DIR / f"{generator_name}_diagnostics.json"
    diag_path.write_text(json.dumps(result["diagnostics"], indent=2))
    logger.info("Diagnostics → %s", diag_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward BL-views backtest")
    parser.add_argument(
        "--generator",
        choices=["equilibrium", "momentum", "mean_rev", "llm-anthropic", "llm-openai"],
        required=True,
        help="View generator to use",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N refit dates (smoke test)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockClient for LLM generators (no network calls)",
    )
    args = parser.parse_args()

    # Guard: live LLM runs need API keys
    if args.generator in ("llm-anthropic", "llm-openai") and not args.mock:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        key_var = "ANTHROPIC_API_KEY" if args.generator == "llm-anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(key_var):
            logger.error(
                "Live run requires %s to be set. Add it to .env or use --mock for offline validation.",
                key_var,
            )
            sys.exit(1)

    result = run_backtest(args.generator, mock=args.mock, max_dates=args.max_dates)
    save_results(args.generator, result)

    diag = result["diagnostics"]
    weights_check = result["weights_frame"].sum(axis=1)
    print(f"\n=== {args.generator} {'[MOCK]' if args.mock else ''} ===")
    print(f"  Refits:            {diag['n_refits']}")
    print(f"  OOS days:          {diag['n_oos_days']}")
    print(f"  Mean views/refit:  {diag['mean_views_per_refit']}")
    print(f"  Sharpe:            {diag['sharpe_ratio']:.4f}" if diag["sharpe_ratio"] else "  Sharpe:            n/a")
    print(f"  Ann Return:        {diag['annualized_return']:.4f}" if diag["annualized_return"] else "  Ann Return:        n/a")
    print(f"  Ann Vol:           {diag['annualized_volatility']:.4f}" if diag["annualized_volatility"] else "  Ann Vol:           n/a")
    print(f"  Max DD:            {diag['max_drawdown']:.4f}" if diag["max_drawdown"] else "  Max DD:            n/a")
    print(f"  Turnover/refit:    {diag['mean_turnover_per_refit']:.4f}" if diag["mean_turnover_per_refit"] else "  Turnover/refit:    n/a")
    print(f"  Weights sum (min/max): {weights_check.min():.4f} / {weights_check.max():.4f}")


if __name__ == "__main__":
    main()
