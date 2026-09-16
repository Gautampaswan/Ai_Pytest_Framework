"""Root pytest fixtures and session bootstrap for the Weather AI Agent suite."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator

import pytest

from tests.support.factories import make_forecast, make_weather_payload
from tests.support.harness import WeatherAgentHarness

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def weather_payload() -> dict:
    return make_weather_payload()


@pytest.fixture
def forecast():
    return make_forecast()


@pytest.fixture
def agent_harness(monkeypatch, capsys, weather_payload) -> WeatherAgentHarness:
    return WeatherAgentHarness(monkeypatch, capsys, weather_payload)


@pytest.fixture
def isolated_env(monkeypatch) -> Iterator[None]:
    """Prevent leftover process env from leaking into config tests."""
    monkeypatch.delenv("MODEL", raising=False)
    yield


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("MODEL", "llama3.2")
