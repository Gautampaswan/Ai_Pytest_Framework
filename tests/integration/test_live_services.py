"""Opt-in live checks against Ollama and the weather provider.

Run with: pytest -m live
Skipped automatically when the dependency is unreachable.
"""

from __future__ import annotations

import httpx
import pytest

import lib.ai as ai_mod
from lib.weather import getweather
from tests.config.settings import settings
from tests.support.assertions import assert_weather_schema
from tests.support.factories import WEATHER_SCHEMA_KEYS


def _ollama_up() -> bool:
    try:
        response = httpx.get(
            settings.OLLAMA_TAGS_URL, timeout=settings.OLLAMA_HEALTH_TIMEOUT_S
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.live


@pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running on localhost:11434")
class TestLiveOllama:
    def test_model_is_installed(self):
        models = httpx.get(
            settings.OLLAMA_TAGS_URL, timeout=settings.LIVE_REQUEST_TIMEOUT_S
        ).json().get("models", [])
        names = {item.get("name", "") for item in models}

        assert any(ai_mod.MODEL in name for name in names), f"{ai_mod.MODEL} is not pulled in Ollama"

    def test_analyze_returns_non_empty_plain_text(self, weather_payload):
        reply = ai_mod.analyze(weather_payload, "In one short sentence, what is the temperature?")

        assert isinstance(reply, str)
        assert reply.strip()
        assert "<think>" not in reply

    def test_is_bye_accepts_farewell(self):
        assert ai_mod.isBye("bye") is True

    def test_is_bye_rejects_weather_question(self):
        assert ai_mod.isBye("what is the temperature?") is False


@pytest.mark.asyncio
async def test_live_weather_fetch_for_dhaka():
    try:
        result = await getweather("Dhaka")
    except SystemExit:
        pytest.skip("Weather provider is unavailable")

    assert WEATHER_SCHEMA_KEYS <= result.keys()
    assert_weather_schema(result)
    assert result["location"]
