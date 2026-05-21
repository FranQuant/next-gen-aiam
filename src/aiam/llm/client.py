from __future__ import annotations

import logging
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-opus-4-7",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text: str = response.content[0].text
        logger.debug("AnthropicClient model=%s prompt_chars=%d", self.model, len(prompt))
        return text


class OpenAIClient:
    """Uses the OpenAI Responses API. Reasoning models ignore temperature/max_tokens."""

    def __init__(self, model: str = "gpt-5.5", max_tokens: int = 2000) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        import openai

        client = openai.OpenAI()
        full_input = f"{system}\n\n{prompt}" if system else prompt
        response = client.responses.create(model=self.model, input=full_input)
        text: str = response.output_text
        logger.debug("OpenAIClient model=%s prompt_chars=%d", self.model, len(prompt))
        return text


class MockClient:
    """Offline test client; records call count for cache-hit assertions."""

    def __init__(self, responses: list[str] | Callable) -> None:
        self._responses = responses
        self._call_count = 0

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        if callable(self._responses):
            result = self._responses(prompt, system=system)
        else:
            idx = min(self._call_count, len(self._responses) - 1)
            result = self._responses[idx]
        self._call_count += 1
        logger.debug("MockClient call=%d", self._call_count)
        return result

    @property
    def call_count(self) -> int:
        return self._call_count
