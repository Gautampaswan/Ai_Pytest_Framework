"""Root pytest fixtures, markers, and session bootstrap for the Weather AI Agent suite."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

from tests.config.settings import settings
from tests.support.factories import make_forecast, make_weather_payload
from tests.support.harness import WeatherAgentHarness
from tests.support.testdata import load_json, load_yaml

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def weather_payload() -> dict:
    return make_weather_payload()


@pytest.fixture
def london_payload() -> dict:
    return load_json("weather_payloads.json")["london_rain"]


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


@pytest.fixture
def conversation_scenarios() -> list[dict]:
    return load_yaml("conversation_scenarios.yaml")["scenarios"]


@pytest.fixture
def farewell_cases() -> dict:
    return load_json("farewell_cases.json")


@pytest.fixture
def deepeval_cases() -> list[dict]:
    return load_yaml("deepeval_cases.yaml")["cases"]


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("MODEL", settings.DEFAULT_MODEL)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-tag tests from their folder when a marker is not already set."""
    marker_by_dir = {
        "unit": pytest.mark.unit,
        "integration": pytest.mark.integration,
        "e2e": pytest.mark.e2e,
    }
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        for folder, marker in marker_by_dir.items():
            if folder in parts and not item.get_closest_marker(folder):
                item.add_marker(marker)
