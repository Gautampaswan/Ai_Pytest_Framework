"""Unit tests for DeepEval custom metrics (deterministic, no judge LLM)."""

from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from tests.support.assertions import assert_deepeval
from tests.support.deepeval_metrics import (
    PlainTextStyleMetric,
    WeatherGroundednessMetric,
    make_weather_llm_test_case,
    offline_weather_metrics,
)
from tests.support.testdata import load_json, load_yaml


pytestmark = [pytest.mark.unit, pytest.mark.deepeval]


def _cases() -> list[dict]:
    return load_yaml("deepeval_cases.yaml")["cases"]


def _payloads() -> dict:
    return load_json("weather_payloads.json")


def _case_to_test_case(case: dict) -> LLMTestCase:
    payload = _payloads()[case["payload"]]
    return make_weather_llm_test_case(
        prompt=case["prompt"],
        actual_output=case["actual_output"],
        payload=payload,
        expected_output=case.get("expected_output"),
        extra_mentions=case.get("extra_mentions"),
        name=case["id"],
    )


class TestWeatherGroundednessMetric:
    @pytest.mark.parametrize(
        "case",
        [pytest.param(case, id=case["id"]) for case in _cases()],
    )
    def test_scores_required_weather_facts(self, case: dict):
        test_case = _case_to_test_case(case)
        metric = WeatherGroundednessMetric()

        metric.measure(test_case)

        assert metric.is_successful() is case["should_pass"]
        if case["should_pass"]:
            assert metric.score == 1.0
        else:
            assert metric.score < 1.0
            assert metric.reason
            assert "Missing" in metric.reason

    def test_raises_without_facts(self):
        metric = WeatherGroundednessMetric()
        test_case = LLMTestCase(input="summary", actual_output="Sunny.")

        with pytest.raises(ValueError, match="required_mentions"):
            metric.measure(test_case)
        assert metric.error


class TestPlainTextStyleMetric:
    def test_accepts_brief_plain_reply(self, weather_payload):
        test_case = make_weather_llm_test_case(
            prompt="How hot is it?",
            actual_output="Dhaka is 25C and partly cloudy.",
            payload=weather_payload,
        )
        metric = PlainTextStyleMetric()

        metric.measure(test_case)

        assert metric.is_successful() is True
        assert metric.score == 1.0

    def test_rejects_markdown_and_think_tags(self):
        cases = _cases()
        markdown_case = next(case for case in cases if case["id"] == "markdown-style-violation")
        test_case = _case_to_test_case(markdown_case)
        metric = PlainTextStyleMetric()

        metric.measure(test_case)

        assert metric.is_successful() is False
        assert markdown_case["style_should_pass"] is False

    def test_rejects_empty_reply(self, weather_payload):
        test_case = make_weather_llm_test_case(
            prompt="summary",
            actual_output="   ",
            payload=weather_payload,
        )
        metric = PlainTextStyleMetric()

        metric.measure(test_case)

        assert metric.is_successful() is False
        assert "empty" in metric.reason


class TestDeepEvalAssertTest:
    def test_assert_test_passes_grounded_plain_reply(self, weather_payload):
        test_case = make_weather_llm_test_case(
            prompt="What is the temperature?",
            actual_output="It is 25C in Dhaka.",
            payload=weather_payload,
            expected_output="Dhaka is 25C.",
        )

        assert_deepeval(test_case, offline_weather_metrics())
