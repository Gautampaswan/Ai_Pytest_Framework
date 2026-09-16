"""CLI lifecycle: location prompt, multi-turn chat, and deterministic exit."""

from __future__ import annotations

import pytest

from tests.config.settings import settings
from tests.support.assertions import assert_farewell_printed
from tests.support.harness import WeatherAgentHarness


pytestmark = pytest.mark.e2e


class TestCliLifecycle:
    def test_prints_location_prompt_and_user_prefix(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(["", "bye"])

        assert "Location(Default-Dhaka): " in result.stdout
        assert "You: " in result.stdout
        assert "AI: " in result.stdout

    def test_default_location_matches_product_setting(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(["", "bye"])

        assert result.locations == [settings.DEFAULT_LOCATION]

    def test_weather_is_fetched_once_for_the_session(
        self, agent_harness: WeatherAgentHarness
    ):
        result = agent_harness.run(
            ["Paris", "temp?", "rain?", "wind?", "bye"]
        )

        assert result.locations == ["Paris"]
        assert len(result.analyze_calls) == 3
        assert_farewell_printed(result.stdout)

    def test_exit_message_is_stable(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(["Dhaka", "bye"])

        assert result.stdout.count(settings.FAREWELL_MESSAGE) == 1
