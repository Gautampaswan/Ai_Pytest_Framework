"""In-memory fakes for LLM completions and async weather clients."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock


class FakeChatCompletion:
    """Minimal OpenAI-compatible chat.completions.create response."""

    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def fake_completion_factory(
    content: str | Callable[..., str],
    captured: dict[str, Any] | None = None,
):
    """Return a create() stand-in that records kwargs and returns content."""

    def _create(**kwargs: Any) -> FakeChatCompletion:
        if captured is not None:
            captured.clear()
            captured.update(kwargs)
        text = content(**kwargs) if callable(content) else content
        return FakeChatCompletion(text)

    return _create


def make_async_weather_client(forecast) -> AsyncMock:
    """Async context manager that mimics python_weather.Client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=forecast)
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    return client
