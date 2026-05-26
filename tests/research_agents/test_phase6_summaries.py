from __future__ import annotations

import inspect
from pathlib import Path

from aiam.research_agents.artifacts import load_artifact_registry
from aiam.research_agents.summaries import (
    BACKTEST_ARTIFACT_IDS,
    summarize_backtest_metrics,
    summarize_historical_llm_experiment_context,
    summarize_pca_dashboard,
    summarize_regime_signal,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARIES_SOURCE = REPO_ROOT / "src/aiam/research_agents/summaries.py"


def _registry():
    return load_artifact_registry(repo_root=REPO_ROOT)


def test_backtest_summary_returns_expected_structure():
    summary = summarize_backtest_metrics(registry=_registry())

    assert isinstance(summary, dict)
    assert summary["section"] == "backtest_metrics"
    assert summary["artifacts_used"]
    assert summary["caveats"]
    assert summary["table_shapes"]
    assert summary["available_columns"]
    assert summary["headline_metrics"]


def test_backtest_summary_uses_only_approved_artifact_ids():
    registry = _registry()
    summary = summarize_backtest_metrics(registry=registry)

    assert set(summary["artifacts_used"]) == set(BACKTEST_ARTIFACT_IDS)
    assert set(summary["artifacts_used"]).issubset(registry)


def test_regime_summary_treats_regimes_as_research_features():
    summary = summarize_regime_signal(registry=_registry())
    caveat_text = " ".join(summary["caveats"]).lower()

    assert isinstance(summary, dict)
    assert summary["section"] == "regime_signal"
    assert summary["artifacts_used"]
    assert summary["caveats"]
    assert "research features" in caveat_text
    assert "instructions for market action" in caveat_text
    assert summary["regime_columns"]["data_published_regime_signals"]
    assert summary["date_range"]


def test_pca_summary_includes_methodology_gap_context():
    summary = summarize_pca_dashboard(registry=_registry())

    assert isinstance(summary, dict)
    assert summary["section"] == "pca_dashboard"
    assert summary["artifacts_used"]
    assert summary["caveats"]
    assert summary["methodology_gaps"]
    assert {
        "methodology_element",
        "availability_in_repo",
        "07b_v1_treatment",
    }.issubset(summary["methodology_gaps"][0])


def test_historical_llm_context_labels_weights_as_historical_not_recommendations():
    summary = summarize_historical_llm_experiment_context(registry=_registry())
    caveat_text = " ".join(summary["caveats"]).lower()

    assert isinstance(summary, dict)
    assert summary["section"] == "historical_llm_experiment_context"
    assert summary["artifacts_used"]
    assert summary["caveats"]
    assert summary["diagnostics_keys"]
    assert summary["return_table_shapes"]
    assert summary["weight_table_shapes"]
    assert "historical weight artifacts" in caveat_text
    assert "historical experiment outputs" in caveat_text
    assert "not target allocations" in caveat_text
    assert "not recommendations" in caveat_text


def test_summarizers_do_not_expose_arbitrary_path_parameters():
    for summarizer in [
        summarize_backtest_metrics,
        summarize_regime_signal,
        summarize_pca_dashboard,
        summarize_historical_llm_experiment_context,
    ]:
        signature = inspect.signature(summarizer)
        assert "path" not in signature.parameters
        assert "manifest_path" not in signature.parameters
        assert "repo_root" not in signature.parameters
        assert list(signature.parameters) in ([], ["registry"])


def test_summaries_module_security_forbidden_terms_absent():
    source = SUMMARIES_SOURCE.read_text(encoding="utf-8").lower()
    forbidden_terms = [
        "dotenv",
        "os.getenv",
        "os.environ",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "openai",
        "anthropic",
        "tavily",
        "broker",
        "order",
        "trade",
    ]

    for term in forbidden_terms:
        assert term not in source
