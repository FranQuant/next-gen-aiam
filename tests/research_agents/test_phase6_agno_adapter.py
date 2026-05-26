from __future__ import annotations

import inspect
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aiam.research_agents import agno_adapter


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCE = REPO_ROOT / "src/aiam/research_agents/agno_adapter.py"
SCRIPT_SOURCE = REPO_ROOT / "scripts/run_phase6_agno_demo.py"


def test_agno_adapter_imports_without_optional_agent_dependencies():
    assert agno_adapter.get_validated_phase6_packet_markdown
    assert agno_adapter.build_phase6_agno_agent


def test_get_validated_phase6_packet_markdown_returns_required_headings():
    markdown = agno_adapter.get_validated_phase6_packet_markdown()

    for heading in [
        "# Phase 6 Research Committee Packet",
        "## Run Context",
        "## Evidence Inventory",
        "## Backtest Metrics",
        "## Regime Signal",
        "## PCA Dashboard",
        "## Historical LLM Experiment Context",
        "## Caveats",
        "## Human Review Checklist",
    ]:
        assert heading in markdown


def test_public_functions_do_not_expose_arbitrary_path_parameters():
    for public_function in [
        agno_adapter.get_validated_phase6_packet_markdown,
        agno_adapter.build_phase6_agno_agent,
    ]:
        signature = inspect.signature(public_function)
        assert "path" not in signature.parameters
        assert "manifest_path" not in signature.parameters
        assert "repo_root" not in signature.parameters
        assert "db_path" not in signature.parameters

    assert list(inspect.signature(agno_adapter.get_validated_phase6_packet_markdown).parameters) == []
    assert list(inspect.signature(agno_adapter.build_phase6_agno_agent).parameters) == ["model_id"]


def test_build_phase6_agno_agent_requires_explicit_model_id():
    with pytest.raises(ValueError, match="model_id is required"):
        agno_adapter.build_phase6_agno_agent(model_id="")


def test_build_phase6_agno_agent_raises_clear_error_when_optional_dependencies_missing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name.startswith("agno"):
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(agno_adapter, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="Optional agent dependencies are not installed"):
        agno_adapter.build_phase6_agno_agent(model_id="demo-model")


def test_cli_default_mode_works_without_optional_agent_dependencies():
    result = subprocess.run(
        [sys.executable, "scripts/run_phase6_agno_demo.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["section_count"] == 8


def test_cli_print_packet_works_without_optional_agent_dependencies():
    result = subprocess.run(
        [sys.executable, "scripts/run_phase6_agno_demo.py", "--print-packet"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith("# Phase 6 Research Committee Packet")
    assert "## Human Review Checklist" in result.stdout


def test_agno_live_run_is_not_executed_in_tests():
    source = Path(__file__).read_text(encoding="utf-8")
    live_flag = "--run" + "-agent"

    assert live_flag not in source


def test_security_forbidden_terms_absent_from_agno_demo_sources():
    forbidden_terms = [
        "dotenv",
        "os.getenv",
        "os.environ",
        "tavily",
        "anthropic",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "broker",
        "order",
        "trade",
        "api_key",
        "secret",
        "token",
        "password",
        "credential",
    ]
    for source_path in [ADAPTER_SOURCE, SCRIPT_SOURCE]:
        source = source_path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in source
