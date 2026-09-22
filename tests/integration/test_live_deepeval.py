"""Opt-in DeepEval LLM-as-judge checks against a live Ollama model.

Run with: pytest -m "live and deepeval"
Skipped automatically when Ollama is unreachable.
"""

from __future__ import annotations

import httpx
import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models import OllamaModel
from deepeval.test_case import SingleTurnParams

import lib.ai as ai_mod
from lib.text import format as format_reply
from tests.config.settings import settings
from tests.support.assertions import assert_deepeval
from tests.support.deepeval_metrics import make_weather_llm_test_case


def _ollama_up() -> bool:
    try:
        response = httpx.get(
            settings.OLLAMA_TAGS_URL, timeout=settings.OLLAMA_HEALTH_TIMEOUT_S
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _judge() -> OllamaModel:
    return OllamaModel(
        model=settings.DEFAULT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )


pytestmark = [
    pytest.mark.live,
    pytest.mark.deepeval,
    pytest.mark.timeout(120),
    pytest.mark.skipif(not _ollama_up(), reason="Ollama is not running on localhost:11434"),
]


class TestLiveDeepEvalJudge:
    def test_analyze_reply_is_faithful_to_weather_snapshot(self, weather_payload):
        raw = ai_mod.analyze(
            weather_payload,
            "In one short sentence, what is the temperature and sky in this city?",
        )
        reply = format_reply(raw)
        test_case = make_weather_llm_test_case(
            prompt="In one short sentence, what is the temperature and sky in this city?",
            actual_output=reply,
            payload=weather_payload,
            expected_output="Dhaka is about 25C and partly cloudy.",
        )
        metric = FaithfulnessMetric(
            threshold=settings.DEEPEVAL_THRESHOLD,
            model=_judge(),
            async_mode=False,
        )

        assert_deepeval(test_case, [metric])

    def test_analyze_reply_is_relevant_to_the_user_prompt(self, weather_payload):
        raw = ai_mod.analyze(weather_payload, "Should I take an umbrella today?")
        reply = format_reply(raw)
        test_case = make_weather_llm_test_case(
            prompt="Should I take an umbrella today?",
            actual_output=reply,
            payload=weather_payload,
        )
        metric = AnswerRelevancyMetric(
            threshold=settings.DEEPEVAL_THRESHOLD,
            model=_judge(),
            async_mode=False,
        )

        assert_deepeval(test_case, [metric])

    def test_geval_brief_plain_weather_style(self, weather_payload):
        raw = ai_mod.analyze(weather_payload, "Give a one-line weather summary.")
        reply = format_reply(raw)
        test_case = make_weather_llm_test_case(
            prompt="Give a one-line weather summary.",
            actual_output=reply,
            payload=weather_payload,
            expected_output="Partly cloudy in Dhaka, about 25C.",
        )
        metric = GEval(
            name="Weather Briefness",
            criteria=(
                "The actual output is a short, friendly, plain-text weather answer. "
                "It must not use markdown. It may mention temperature or sky from the input context."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            threshold=settings.DEEPEVAL_THRESHOLD,
            model=_judge(),
            async_mode=False,
        )

        assert_deepeval(test_case, [metric])
