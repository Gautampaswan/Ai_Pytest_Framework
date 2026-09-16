"""End-to-end CLI conversation tests using the agent harness (no live services)."""

from __future__ import annotations

import pytest

from tests.support.harness import WeatherAgentHarness


pytestmark = pytest.mark.e2e


SCENARIOS = [
    pytest.param(
        ["", "What's the temperature?", "bye"],
        "Dhaka",
        ["What's the temperature?"],
        id="default-city-then-temperature",
    ),
    pytest.param(
        ["mumbai", "Will it rain today?", "bye"],
        "Mumbai",
        ["Will it rain today?"],
        id="named-city-rain-question",
    ),
    pytest.param(
        ["Delhi", "humidity?", "wind speed?", "bye"],
        "Delhi",
        ["humidity?", "wind speed?"],
        id="multi-turn-follow-up",
    ),
]


class TestAgentConversation:
    @pytest.mark.parametrize("inputs, expected_city, expected_prompts", SCENARIOS)
    def test_conversation_uses_one_weather_snapshot(
        self,
        agent_harness: WeatherAgentHarness,
        weather_payload,
        inputs,
        expected_city,
        expected_prompts,
    ):
        result = agent_harness.run(inputs)

        assert result.locations == [expected_city]
        assert result.prompts == expected_prompts
        assert all(call[0] is weather_payload or call[0] == weather_payload for call in result.analyze_calls)
        assert "Bye. Have a nice day." in result.stdout

    def test_reply_is_grounded_in_injected_weather(
        self, monkeypatch, capsys, weather_payload
    ):
        def reply(api: dict, prompt: str) -> str:
            return f"{api['location']} is {api['temperature']}C. Asked: {prompt}"

        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=reply,
        )

        result = harness.run(["Dhaka", "How hot is it?", "bye"])

        assert "Dhaka is 25C. Asked: How hot is it?" in result.stdout
