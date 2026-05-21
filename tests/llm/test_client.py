from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiam.llm.client import AnthropicClient, MockClient


# ── MockClient ───────────────────────────────────────────────────────────────

def test_mock_client_returns_canned_response():
    mock = MockClient(["hello"])
    assert mock.complete("prompt") == "hello"


def test_mock_client_call_count():
    mock = MockClient(["a", "b"])
    mock.complete("p1")
    mock.complete("p2")
    assert mock.call_count == 2


def test_mock_client_callable_responses():
    mock = MockClient(lambda p, system=None: f"echo:{p[:5]}")
    result = mock.complete("hello world")
    assert result == "echo:hello"


def test_mock_client_clamps_to_last_response():
    mock = MockClient(["only"])
    mock.complete("p1")
    mock.complete("p2")  # beyond list length — clamps to last
    assert mock.call_count == 2


# ── AnthropicClient kwargs construction ─────────────────────────────────────

def _make_fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def test_anthropic_no_temperature_by_default():
    """Default AnthropicClient (temperature=None) must not include 'temperature' in kwargs."""
    client = AnthropicClient()
    fake_resp = _make_fake_response("result")

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_resp
        client.complete("test prompt")

    _, call_kwargs = MockAnthropic.return_value.messages.create.call_args
    assert "temperature" not in call_kwargs, (
        f"temperature should not be sent when None, got kwargs keys: {list(call_kwargs)}"
    )


def test_anthropic_temperature_included_when_set():
    """Explicit temperature value must be forwarded."""
    client = AnthropicClient(temperature=0.0)
    fake_resp = _make_fake_response("result")

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_resp
        client.complete("test prompt")

    _, call_kwargs = MockAnthropic.return_value.messages.create.call_args
    assert "temperature" in call_kwargs
    assert call_kwargs["temperature"] == 0.0


def test_anthropic_system_included_when_provided():
    client = AnthropicClient()
    fake_resp = _make_fake_response("result")

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_resp
        client.complete("prompt", system="sys")

    _, call_kwargs = MockAnthropic.return_value.messages.create.call_args
    assert call_kwargs.get("system") == "sys"


def test_anthropic_system_absent_when_not_provided():
    client = AnthropicClient()
    fake_resp = _make_fake_response("result")

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_resp
        client.complete("prompt")

    _, call_kwargs = MockAnthropic.return_value.messages.create.call_args
    assert "system" not in call_kwargs
