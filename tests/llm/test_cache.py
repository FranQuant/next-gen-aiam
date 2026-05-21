from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aiam.llm.cache import PromptCache
from aiam.llm.client import MockClient
from aiam.llm.views import LLMViewGenerator


# ── Basic cache round-trip ───────────────────────────────────────────────────

def test_set_get_roundtrip(tmp_path):
    cache = PromptCache(cache_dir=tmp_path)
    cache.set("model-x", "sys", "prompt", "response text")
    result = cache.get("model-x", "sys", "prompt")
    assert result == "response text"


def test_get_returns_none_on_miss(tmp_path):
    cache = PromptCache(cache_dir=tmp_path)
    result = cache.get("model-x", None, "unseen prompt")
    assert result is None


def test_different_models_different_keys(tmp_path):
    cache = PromptCache(cache_dir=tmp_path)
    cache.set("model-a", None, "prompt", "resp-a")
    cache.set("model-b", None, "prompt", "resp-b")
    assert cache.get("model-a", None, "prompt") == "resp-a"
    assert cache.get("model-b", None, "prompt") == "resp-b"


def test_different_system_different_keys(tmp_path):
    cache = PromptCache(cache_dir=tmp_path)
    cache.set("m", "sys-a", "prompt", "resp-a")
    cache.set("m", "sys-b", "prompt", "resp-b")
    assert cache.get("m", "sys-a", "prompt") == "resp-a"
    assert cache.get("m", "sys-b", "prompt") == "resp-b"


def test_disabled_cache_never_stores(tmp_path):
    cache = PromptCache(cache_dir=tmp_path, disabled=True)
    cache.set("m", None, "p", "r")
    assert cache.get("m", None, "p") is None


# ── Cache hit reduces client calls ───────────────────────────────────────────

def _make_returns(n_assets: int = 2, n_obs: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2022-01-01", periods=n_obs)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, (n_obs, n_assets)),
        index=dates,
        columns=["SPY", "IEF"],
    )


def _canned_response() -> str:
    return json.dumps({
        "views": [
            {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.6},
        ]
    })


def test_cache_hit_no_extra_client_call(tmp_path):
    """Calling the generator twice with the same (returns, asof) → only 1 client call."""
    mock = MockClient([_canned_response()])
    cache = PromptCache(cache_dir=tmp_path)
    gen = LLMViewGenerator(mock, cache=cache)

    returns = _make_returns()
    asof = returns.index[-1]

    gen(returns, asof)
    assert mock.call_count == 1

    gen(returns, asof)
    assert mock.call_count == 1  # cache hit — client not called again


def test_different_asof_causes_new_call(tmp_path):
    """Different asof → different prompt → cache miss → new client call."""
    mock = MockClient([_canned_response(), _canned_response()])
    cache = PromptCache(cache_dir=tmp_path)
    gen = LLMViewGenerator(mock, cache=cache)

    returns = _make_returns()
    asof1 = returns.index[-1]
    asof2 = returns.index[-2]

    gen(returns, asof1)
    gen(returns, asof2)
    assert mock.call_count == 2
