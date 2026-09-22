"""Reusable assertions that encode product contracts, not one-off checks."""

from __future__ import annotations

from typing import Any, Iterable

from tests.config.settings import settings
from tests.support.factories import WEATHER_SCHEMA_KEYS


def assert_weather_schema(payload: dict[str, Any]) -> None:
    missing = WEATHER_SCHEMA_KEYS - payload.keys()
    assert not missing, f"Weather payload missing keys: {sorted(missing)}"
    assert isinstance(payload.get("temperature"), int)
    assert isinstance(payload.get("feels_like"), int)
    assert isinstance(payload.get("coordinates"), dict)
    assert "X" in payload["coordinates"] and "Y" in payload["coordinates"]


def assert_farewell_printed(stdout: str) -> None:
    assert settings.FAREWELL_MESSAGE in stdout


def assert_no_think_tags(text: str) -> None:
    assert "<think>" not in text
    assert "</think>" not in text


def assert_prompts_equal(actual: Iterable[str], expected: Iterable[str]) -> None:
    assert list(actual) == list(expected)


def assert_deepeval(test_case, metrics) -> None:
    """Run DeepEval assert_test without spinning an extra event loop."""
    from deepeval import assert_test

    assert_test(test_case, metrics, run_async=False)
