"""DeepEval metrics and LLMTestCase builders for the Weather AI Agent.

Offline metrics are deterministic (no judge LLM). Live tests pass an Ollama
model into DeepEval's Faithfulness / AnswerRelevancy / GEval metrics.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from tests.config.settings import settings

_MARKDOWN_RE = re.compile(
    r"(```)|(\*\*)|(^|\n)\s{0,3}#{1,6}\s|(\[[^\]]+\]\([^)]+\))",
    re.MULTILINE,
)


def weather_retrieval_context(payload: dict[str, Any]) -> list[str]:
    """Turn a weather snapshot into DeepEval retrieval_context facts."""
    return [
        f"location: {payload['location']}",
        f"temperature: {payload['temperature']}",
        f"description: {payload['description']}",
        f"humidity: {payload['humidity']}",
        f"kind: {payload['kind']}",
    ]


def default_required_mentions(
    payload: dict[str, Any], extra: Iterable[str] | None = None
) -> list[str]:
    mentions = [str(payload["location"]), str(payload["temperature"])]
    if extra:
        mentions.extend(str(item) for item in extra)
    return mentions


def make_weather_llm_test_case(
    *,
    prompt: str,
    actual_output: str,
    payload: dict[str, Any],
    expected_output: str | None = None,
    extra_mentions: Iterable[str] | None = None,
    name: str | None = None,
) -> LLMTestCase:
    context = weather_retrieval_context(payload)
    return LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
        retrieval_context=context,
        metadata={
            "required_mentions": default_required_mentions(payload, extra_mentions),
            "location": payload.get("location"),
        },
        name=name or f"{payload.get('location')}:{prompt[:48]}",
    )


def _required_mentions(test_case: LLMTestCase) -> list[str]:
    metadata = test_case.metadata or {}
    configured = metadata.get("required_mentions")
    if configured:
        return [str(item) for item in configured if str(item).strip()]

    mentions: list[str] = []
    for fact in test_case.retrieval_context or []:
        if ":" in fact:
            mentions.append(fact.split(":", 1)[1].strip())
        elif fact.strip():
            mentions.append(fact.strip())
    return mentions


class WeatherGroundednessMetric(BaseMetric):
    """Score 1.0 only when every required weather fact appears in the reply."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            settings.DEEPEVAL_STRICT_THRESHOLD if threshold is None else threshold
        )
        self.async_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            output = (test_case.actual_output or "").lower()
            mentions = _required_mentions(test_case)
            if not mentions:
                raise ValueError("WeatherGroundednessMetric needs required_mentions or retrieval_context")
            missing = [item for item in mentions if item.lower() not in output]
            hit_count = len(mentions) - len(missing)
            self.score = hit_count / len(mentions)
            self.reason = (
                "All required weather facts are present."
                if not missing
                else f"Missing weather facts in actual_output: {missing}"
            )
            self.success = self.score >= self.threshold
            return self.score
        except Exception as exc:
            self.error = str(exc)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "Weather Groundedness"


class PlainTextStyleMetric(BaseMetric):
    """Agent contract: brief plain text, no markdown, no leaked think tags."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            settings.DEEPEVAL_STRICT_THRESHOLD if threshold is None else threshold
        )
        self.async_mode = False
        self.include_reason = True

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            text = test_case.actual_output or ""
            problems: list[str] = []
            if not text.strip():
                problems.append("empty reply")
            if "<think>" in text or "</think>" in text:
                problems.append("leaked <think> tags")
            if _MARKDOWN_RE.search(text):
                problems.append("markdown formatting")
            self.score = 0.0 if problems else 1.0
            self.reason = (
                "Reply is plain text without think tags."
                if not problems
                else "Style violations: " + "; ".join(problems)
            )
            self.success = self.score >= self.threshold
            return self.score
        except Exception as exc:
            self.error = str(exc)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "Plain Text Style"


def offline_weather_metrics() -> list[BaseMetric]:
    return [WeatherGroundednessMetric(), PlainTextStyleMetric()]
