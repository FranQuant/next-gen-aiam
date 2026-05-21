"""Stage-2 smoke script: calls one real LLM provider and prints parsed views.

Run manually (never executed during the build):
    python scripts/llm_smoke_views.py
    python scripts/llm_smoke_views.py --provider openai
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM view generator smoke test")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    from aiam.llm import AnthropicClient, LLMViewGenerator, OpenAIClient
    from aiam.llm.cache import PromptCache

    if args.provider == "anthropic":
        client = AnthropicClient(model="claude-opus-4-7")
        print("Provider: Anthropic (claude-opus-4-7)")
    else:
        client = OpenAIClient(model="gpt-5.5")
        print("Provider: OpenAI (gpt-5.5)")

    # Synthetic returns — stand-in until real data is wired
    rng = np.random.default_rng(42)
    tickers = ["SPY", "IEF", "GLD", "QQQ"]
    n_obs = 300
    dates = pd.bdate_range("2023-01-01", periods=n_obs)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (n_obs, len(tickers))),
        index=dates,
        columns=tickers,
    )
    asof = returns.index[-1]

    print(f"\nasof: {asof.date()}")
    print(f"Universe: {tickers}")
    print(f"Returns shape: {returns.shape}")

    gen = LLMViewGenerator(client, cache=PromptCache())
    P, Q, Omega = gen(returns, asof)

    print(f"\n--- Results ---")
    print(f"P shape: {P.shape}")
    print(f"Q: {Q}")
    print(f"Omega diagonal: {Omega.diagonal()}")
    if len(Q) > 0:
        print("\nViews:")
        for i, q in enumerate(Q):
            asset = tickers[P[i].argmax()]
            print(f"  {asset}: expected_return={q:+.4f}, omega={Omega[i,i]:.6f}")


if __name__ == "__main__":
    main()
