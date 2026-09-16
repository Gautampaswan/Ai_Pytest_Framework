"""Unit tests for CLI orchestration in main() with injected collaborators."""

from __future__ import annotations

import pytest

from tests.support.assertions import assert_farewell_printed, assert_prompts_equal
from tests.support.harness import WeatherAgentHarness
from tests.support.testdata import load_json


pytestmark = pytest.mark.unit

FAREWELL = load_json("farewell_cases.json")


class TestMainLocation:
    def test_blank_input_defaults_to_dhaka(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(["", "bye"])

        assert result.locations == ["Dhaka"]

    def test_city_is_capitalized_before_weather_fetch(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(["delhi", "bye"])

        assert result.locations == ["Delhi"]

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("mumbai", "Mumbai"),
            ("NEW YORK", "New york"),
            ("lOnDoN", "London"),
        ],
    )
    def test_str_capitalize_contract(self, agent_harness: WeatherAgentHarness, raw, expected):
        result = agent_harness.run([raw, "bye"])

        assert result.locations == [expected]


class TestMainConversation:
    def test_analyze_receives_weather_payload_and_user_prompt(
        self, agent_harness: WeatherAgentHarness, weather_payload
    ):
        result = agent_harness.run(["Dhaka", "What is the temperature?", "bye"])

        assert_prompts_equal(result.prompts, ["What is the temperature?"])
        assert result.analyze_calls[0][0] == weather_payload
        assert "It is 25C and cloudy." in result.stdout

    @pytest.mark.parametrize("farewell", FAREWELL["keyword_exits"])
    def test_keyword_bye_exits_without_calling_analyze(
        self, agent_harness: WeatherAgentHarness, farewell: str
    ):
        result = agent_harness.run(["Dhaka", farewell])

        assert result.prompts == []
        assert_farewell_printed(result.stdout)

    def test_model_detected_farewell_exits_without_bye_keyword(
        self, monkeypatch, capsys, weather_payload
    ):
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            is_bye=lambda text: text.lower() == "see you later",
        )

        result = harness.run(["Dhaka", "see you later"])

        assert result.prompts == []
        assert_farewell_printed(result.stdout)

    def test_weather_question_is_not_treated_as_exit(
        self, agent_harness: WeatherAgentHarness
    ):
        result = agent_harness.run(["Dhaka", "what is the humidity?", "bye"])

        assert result.prompts == ["what is the humidity?"]
        assert_farewell_printed(result.stdout)

    def test_multi_turn_then_exit(self, agent_harness: WeatherAgentHarness):
        result = agent_harness.run(
            ["London", "Will it rain?", "What about humidity?", "bye"]
        )

        assert result.locations == ["London"]
        assert result.prompts == ["Will it rain?", "What about humidity?"]
        assert result.stdout.count("It is 25C and cloudy.") == 2

    def test_think_tags_are_stripped_from_model_output(
        self, monkeypatch, capsys, weather_payload
    ):
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply="<think>hidden</think>\nCarry an umbrella.",
        )

        result = harness.run(["Dhaka", "Do I need an umbrella?", "bye"])

        assert "Carry an umbrella." in result.stdout
        assert "<think>" not in result.stdout

    @pytest.mark.negative
    def test_raises_when_conversation_inputs_are_exhausted(
        self, agent_harness: WeatherAgentHarness
    ):
        with pytest.raises(AssertionError, match="more input than the test provided"):
            agent_harness.run(["Dhaka"])
