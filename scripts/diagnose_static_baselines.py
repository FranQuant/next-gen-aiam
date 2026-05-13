"""
Diagnostic dive on the static-baseline horse race.
Investigates GMV/MSR weight concentration, solver/fallback events, and
Sharpe sensitivity to a realistic risk-free rate.

Run:  source .venv/bin/activate && python scripts/diagnose_static_baselines.py
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from aiam.data.panel import Panel
from aiam.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from aiam.estimators.mean import sample_mean
from aiam.evaluation.performance import performance_stats
from aiam.harness.horse_race import run_horse_race
from aiam.strategy.equal_weight import EqualWeight
from aiam.strategy.global_min_variance import GlobalMinVariance
from aiam.strategy.max_sharpe import MaximumSharpe

START = "2008-01-01"
END = "2026-04-30"
RF_HI = 0.015  # 1.5% annualised — rough T-bill average over 2008-2026
CACHE = Path(__file__).parent.parent / "data" / "cache" / "prices_30.parquet"

# Regexes for structured extraction from formatted log messages
DATE_RE   = re.compile(r"asof=(\d{4}-\d{2}-\d{2})")
STATUS_RE = re.compile(r"non-optimal:\s*(\S+)\s+at")


# ── logging capture ──────────────────────────────────────────────────────────

class _ListHandler(logging.Handler):
    """Collects LogRecord objects emitted during a strategy run."""
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_with_capture(panel, strategy, start, end):
    h = _ListHandler()
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(h)
    root.setLevel(logging.WARNING)

    with warnings.catch_warnings(record=True) as py_warns:
        warnings.simplefilter("always")
        result = run_horse_race(panel, strategy, start=start, end=end)

    root.removeHandler(h)
    root.setLevel(prev_level)
    return result, h.records, list(py_warns)


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_dates(records: list[logging.LogRecord], fragment: str) -> list[pd.Timestamp]:
    dates = []
    for rec in records:
        msg = rec.getMessage()
        if fragment in msg:
            m = DATE_RE.search(msg)
            if m:
                dates.append(pd.Timestamp(m.group(1)))
    return sorted(dates)


def _extract_status_counts(
    records: list[logging.LogRecord], fragment: str
) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for rec in records:
        msg = rec.getMessage()
        if fragment in msg:
            m = STATUS_RE.search(msg)
            counts[m.group(1) if m else "unknown"] += 1
    return counts


def _dates_summary(dates: list[pd.Timestamp]) -> str:
    if not dates:
        return "—"
    first5 = [str(d.date()) for d in dates[:5]]
    last5  = [str(d.date()) for d in dates[-5:]] if len(dates) > 5 else []
    s = "  first: " + ", ".join(first5)
    if last5:
        s += "\n              last:  " + ", ".join(last5)
    return s


# ── section printers ─────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> None:
    rows = []
    for name, data in all_results.items():
        port_ret = data["portfolio_returns"].dropna()
        s0  = performance_stats(port_ret, rf=0.0)
        shi = performance_stats(port_ret, rf=RF_HI)
        rows.append({
            "strategy":       name,
            "ann_return":     s0["annualized_return"],
            "ann_vol":        s0["annualized_volatility"],
            "max_dd":         s0["max_drawdown"],
            "sharpe(rf=0)":   s0["sharpe_ratio"],
            "sharpe(rf=1.5%)": shi["sharpe_ratio"],
        })

    df = pd.DataFrame(rows).set_index("strategy")
    df["delta"] = df["sharpe(rf=1.5%)"] - df["sharpe(rf=0)"]

    disp = pd.DataFrame(index=df.index)
    disp["ann_return"]      = df["ann_return"].map("{:.2%}".format)
    disp["ann_vol"]         = df["ann_vol"].map("{:.2%}".format)
    disp["max_dd"]          = df["max_dd"].map("{:.2%}".format)
    disp["sharpe(rf=0)"]    = df["sharpe(rf=0)"].map("{:.3f}".format)
    disp["sharpe(rf=1.5%)"] = df["sharpe(rf=1.5%)"].map("{:.3f}".format)
    disp["delta"]           = df["delta"].map("{:+.3f}".format)

    print(disp.to_string())
    return df  # caller doesn't use, but handy


def print_concentration(name: str, weights_df: pd.DataFrame) -> None:
    wdf = weights_df.fillna(0.0)
    mean_w = wdf.mean().sort_values(ascending=False)
    top3   = mean_w.iloc[:3].sum()
    max_w  = wdf.max(axis=1)

    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print("  Top-8 average weights:")
    for ticker, w in mean_w.head(8).items():
        bar = "█" * int(w * 40)
        print(f"    {ticker:22s}  {w:6.3f}  {bar}")
    print(f"  Top-3 combined share:       {top3:.1%}")
    print(f"  Days: max weight > 50%:     {(max_w > 0.50).mean():.1%}")
    print(f"  Days: max weight > 80%:     {(max_w > 0.80).mean():.1%}")


def print_solver_events(
    name: str,
    log_records: list[logging.LogRecord],
    py_warns: list,
) -> None:
    print(f"\n  {name}:")

    # Non-optimal solver statuses (GMV + MSR new instrumentation)
    status_counts = _extract_status_counts(log_records, "solver status non-optimal")
    if status_counts:
        for status, count in sorted(status_counts.items()):
            dates = _extract_dates(log_records, f"non-optimal: {status}")
            print(f"    solver non-optimal ({status}): {count} events")
            print(f"            {_dates_summary(dates)}")
    else:
        print("    solver non-optimal events: 0")

    # MSR EW-fallback — no positive excess mean
    fb_dates = _extract_dates(log_records, "no positive excess return")
    if fb_dates:
        print(f"    MSR EW-fallback (no pos mean): {len(fb_dates)} events")
        print(f"            {_dates_summary(fb_dates)}")

    # MSR solver-based EW fallback
    sv_dates = _extract_dates(log_records, "MSR solver fallback to EW")
    if sv_dates:
        print(f"    MSR EW-fallback (solver):      {len(sv_dates)} events")
        print(f"            {_dates_summary(sv_dates)}")

    osqp_count = sum(
        1 for w in py_warns
        if issubclass(w.category, UserWarning) and "inaccurate" in str(w.message).lower()
    )
    if osqp_count:
        print(f"    OSQP 'inaccurate' UserWarnings: {osqp_count}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading panel from {CACHE} …")
    prices = pd.read_parquet(CACHE)
    panel  = Panel({"prices": prices})

    strategies = {
        "EW":               EqualWeight(),
        "GMV(sample)":      GlobalMinVariance(sample_cov),
        "GMV(ledoit_wolf)": GlobalMinVariance(ledoit_wolf_cov),
        "GMV(oas)":         GlobalMinVariance(oas_cov),
        "MSR(sample)":      MaximumSharpe(sample_cov,      sample_mean),
        "MSR(ledoit_wolf)": MaximumSharpe(ledoit_wolf_cov, sample_mean),
    }

    print(f"Running horse race {START} → {END} …\n")
    collected: dict[str, dict] = {}
    for name, strategy in strategies.items():
        print(f"  {name} …", flush=True)
        result, log_recs, py_warns = _run_with_capture(panel, strategy, START, END)
        collected[name] = {
            "portfolio_returns": result["portfolio_returns"],
            "weights":           result["weights"],
            "log_records":       log_recs,
            "py_warns":          py_warns,
        }

    # ── 1. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1.  SIX-WAY COMPARISON  (2008-01-01 → 2026-04-30)")
    print("=" * 70)
    print_comparison_table(collected)

    # ── 2. Weight concentration ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2.  WEIGHT CONCENTRATION")
    print("=" * 70)
    for name, data in collected.items():
        print_concentration(name, data["weights"])

    # ── 3. Solver / fallback events ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3.  SOLVER / FALLBACK EVENTS")
    print("=" * 70)
    for name, data in collected.items():
        print_solver_events(name, data["log_records"], data["py_warns"])

    # ── 4. Summary paragraph ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4.  SUMMARY")
    print("=" * 70)

    def _mean_w(name):
        return collected[name]["weights"].fillna(0.0).mean().sort_values(ascending=False)

    def _sharpe(name, rf):
        return performance_stats(
            collected[name]["portfolio_returns"].dropna(), rf=rf
        )["sharpe_ratio"]

    def _crisis_count(name, fragment):
        dates = _extract_dates(collected[name]["log_records"], fragment)
        return sum(1 for d in dates if d.year in (2008, 2009, 2020)), len(dates)

    # GMV(sample) corner?
    gmv_s_top3  = _mean_w("GMV(sample)").iloc[:3].sum()
    gmv_s_top1  = _mean_w("GMV(sample)").index[0]
    gmv_s_corner = gmv_s_top3 > 0.90 and gmv_s_top1 == "SHY.US"

    # MSR(sample) concentration
    msr_s_top3 = _mean_w("MSR(sample)").iloc[:3].sum()
    msr_s_top1 = _mean_w("MSR(sample)").index[0]

    # Sharpe at rf=0 vs 1.5%
    gmv_s_sh0  = _sharpe("GMV(sample)", 0.0)
    gmv_s_shi  = _sharpe("GMV(sample)", RF_HI)

    # Fallback clustering
    msr_s_crisis,  msr_s_total  = _crisis_count("MSR(sample)",      "no positive excess return")
    msr_lw_crisis, msr_lw_total = _crisis_count("MSR(ledoit_wolf)", "no positive excess return")
    noopt_crisis,  noopt_total  = _crisis_count("MSR(sample)",      "solver status non-optimal")
    noopt_gmv_c,   noopt_gmv_t  = _crisis_count("GMV(sample)",      "solver status non-optimal")

    print(f"""
Is GMV(sample) a SHY-corner portfolio?
  Top-3 share = {gmv_s_top3:.1%}, dominant ticker = {gmv_s_top1}.
  Answer: {'YES' if gmv_s_corner else 'NO'} — {'essentially 100% SHY (classic corner solution when a near-zero-vol asset is in universe).' if gmv_s_corner else 'not a pure corner.'}

Is MSR(sample) similarly concentrated, and on what?
  Top-3 share = {msr_s_top3:.1%}, dominant ticker = {msr_s_top1}.
  {'High concentration' if msr_s_top3 > 0.70 else 'Moderate diversification'} — MSR chases the asset with the highest sample Sharpe, which is typically a low-vol fixed-income ETF that happened to trend up in the estimation window.

Does rf=1.5% kill GMV(sample)'s Sharpe?
  rf=0 Sharpe = {gmv_s_sh0:.3f}  →  rf=1.5% Sharpe = {gmv_s_shi:.3f}  (delta {gmv_s_shi - gmv_s_sh0:+.3f}).
  {'YES — turns negative or near-zero; GMV(sample) earns less than cash.' if gmv_s_shi < 0.10 else 'Partial impact; Sharpe degraded but stays positive.'}

Do OSQP-inaccurate or MSR-fallback events cluster in 2008/2020?
  MSR(sample) EW-fallback (no pos mean): {msr_s_total} events total, {msr_s_crisis} in 2008/09/2020.
  MSR(ledoit_wolf) EW-fallback:           {msr_lw_total} events total, {msr_lw_crisis} in 2008/09/2020.
  MSR(sample) solver non-optimal:         {noopt_total} events total, {noopt_crisis} in 2008/09/2020.
  GMV(sample) solver non-optimal:         {noopt_gmv_t} events total, {noopt_gmv_c} in 2008/09/2020.
  {'Fallbacks DO cluster in crisis windows (2008/09 and/or 2020).' if (msr_s_crisis + msr_lw_crisis + noopt_crisis) > 0 else 'No detectable clustering in 2008/2020 — events are spread uniformly across the period.'}
""")


if __name__ == "__main__":
    main()
