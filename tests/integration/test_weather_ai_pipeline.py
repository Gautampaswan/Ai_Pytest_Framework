"""Cross-module pipeline tests (mocked weather + LLM + formatter)."""

from __future__ import annotations

import pytest

from lib.text import format as format_reply
from lib.weather import getweather
from tests.support.assertions import assert_no_think_tags, assert_weather_schema
from tests.support.fakes import make_async_weather_client
from tests.support.harness import WeatherAgentHarness


pytestmark = pytest.mark.integration


class TestWeatherToCliPipeline:
    @pytest.mark.asyncio
    async def test_weather_mapper_emits_cli_ready_payload(self, monkeypatch, forecast):
        client = make_async_weather_client(forecast)
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        payload = await getweather("Dhaka")

        assert_weather_schema(payload)
        assert payload["location"] == "Dhaka"
        assert payload["temperature"] == 25

    def test_mapped_weather_is_the_payload_passed_to_analyze(
        self, monkeypatch, capsys, weather_payload
    ):
        seen: list[dict] = []

        def analyze_reply(api: dict, prompt: str) -> str:
            seen.append(api)
            return f"{api['location']} {api['temperature']}C"

        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=analyze_reply,
        )
        result = harness.run(["Dhaka", "How hot is it?", "bye"])

        assert seen[0] == weather_payload
        assert "Dhaka 25C" in result.stdout

    def test_formatter_is_applied_before_cli_print(
        self, monkeypatch, capsys, weather_payload
    ):
        raw = "<think>chain</think>\nLight rain this afternoon."
        assert format_reply(raw) == "Light rain this afternoon."

        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=raw,
        )
        result = harness.run(["Dhaka", "Do I need a coat?", "bye"])

        assert "Light rain this afternoon." in result.stdout
        assert_no_think_tags(result.stdout)

    def test_alternate_city_payload_is_not_rewritten_by_cli(
        self, monkeypatch, capsys, london_payload
    ):
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            london_payload,
            analyze_reply=lambda api, prompt: f"{api['location']} {api['description']}",
        )

        result = harness.run(["london", "summary", "bye"])

        assert result.locations == ["London"]
        assert "London Light rain" in result.stdout
