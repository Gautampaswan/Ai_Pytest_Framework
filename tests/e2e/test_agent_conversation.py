"""End-to-end CLI conversation tests using the agent harness (no live services)."""

from __future__ import annotations

import pytest

from tests.support.assertions import assert_farewell_printed
from tests.support.harness import WeatherAgentHarness
from tests.support.testdata import load_yaml


pytestmark = pytest.mark.e2e


def _conversation_params():
    scenarios = load_yaml("conversation_scenarios.yaml")["scenarios"]
    return [
        pytest.param(
            case["inputs"],
            case["expected_city"],
            case["expected_prompts"],
            id=case["id"],
        )
        for case in scenarios
    ]


class TestAgentConversation:
    @pytest.mark.parametrize("inputs, expected_city, expected_prompts", _conversation_params())
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
        assert all(
            call[0] is weather_payload or call[0] == weather_payload
            for call in result.analyze_calls
        )
        assert_farewell_printed(result.stdout)

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
