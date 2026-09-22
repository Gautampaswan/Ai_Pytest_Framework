"""DeepEval scores the weather → analyze → format pipeline (mocked LLM)."""

from __future__ import annotations

import pytest

from lib.text import format as format_reply
from tests.support.assertions import assert_deepeval, assert_no_think_tags
from tests.support.deepeval_metrics import (
    PlainTextStyleMetric,
    WeatherGroundednessMetric,
    make_weather_llm_test_case,
    offline_weather_metrics,
)
from tests.support.harness import WeatherAgentHarness


pytestmark = [pytest.mark.integration, pytest.mark.deepeval]


class TestDeepEvalOnCliPipeline:
    def test_printed_reply_passes_offline_deepeval_metrics(
        self, monkeypatch, capsys, weather_payload
    ):
        reply = "It is 25C and partly cloudy in Dhaka."
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=reply,
        )

        result = harness.run(["Dhaka", "What is the temperature?", "bye"])

        assert reply in result.stdout
        test_case = make_weather_llm_test_case(
            prompt="What is the temperature?",
            actual_output=reply,
            payload=weather_payload,
            extra_mentions=["partly cloudy"],
        )
        assert_deepeval(test_case, offline_weather_metrics())

    def test_formatter_output_is_what_deepeval_scores(
        self, monkeypatch, capsys, weather_payload
    ):
        raw = "<think>internal</think>\nDhaka is 25C. Carry a light jacket."
        cleaned = format_reply(raw)
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=raw,
        )

        result = harness.run(["Dhaka", "What should I wear?", "bye"])

        assert_no_think_tags(result.stdout)
        assert cleaned in result.stdout
        test_case = make_weather_llm_test_case(
            prompt="What should I wear?",
            actual_output=cleaned,
            payload=weather_payload,
        )
        assert_deepeval(test_case, [PlainTextStyleMetric(), WeatherGroundednessMetric()])

    @pytest.mark.negative
    def test_hallucinated_reply_fails_groundedness(
        self, monkeypatch, capsys, weather_payload
    ):
        reply = "Paris is 40C and sunny."
        harness = WeatherAgentHarness(
            monkeypatch,
            capsys,
            weather_payload,
            analyze_reply=reply,
        )
        harness.run(["Dhaka", "How hot is it?", "bye"])

        test_case = make_weather_llm_test_case(
            prompt="How hot is it?",
            actual_output=reply,
            payload=weather_payload,
        )
        metric = WeatherGroundednessMetric()
        metric.measure(test_case)

        assert metric.is_successful() is False
        assert "Dhaka" in metric.reason or "25" in metric.reason
