#!/usr/bin/env python
"""CLI for the deterministic Notebook 03 MSR ensemble workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aiam.strategy.ml_ensemble_msr import (  # noqa: E402
    MLEnsembleMSRConfig,
    run_ml_ensemble_msr_research,
    summary_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic MSR(Ensemble_mu_hat).")
    parser.add_argument(
        "--ohlcv-path",
        default="data/cache/prices_29_ohlcv_2003_2026.parquet",
        help="Local cached OHLCV parquet path.",
    )
    parser.add_argument(
        "--prices-path",
        default="data/cache/prices_29.parquet",
        help="Local cached wide price parquet path.",
    )
    parser.add_argument(
        "--returns-path",
        default="data/cache/returns_29_2003_2026.parquet",
        help="Local cached wide return parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ml_ensemble_msr",
        help="Artifact directory used only with --write-artifacts.",
    )
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--cov-lookback", type=int, default=504)
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--print-summary-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MLEnsembleMSRConfig(
        train_end=args.train_end,
        test_start=args.test_start,
        cov_lookback=args.cov_lookback,
    )
    result = run_ml_ensemble_msr_research(
        ohlcv_path=args.ohlcv_path,
        prices_path=args.prices_path,
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        write_artifacts=args.write_artifacts,
        config=config,
    )
    payload = summary_payload(result)
    if args.print_summary_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
