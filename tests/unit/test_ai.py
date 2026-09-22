"""Unit tests for the LLM adapter: intent detection and weather analysis."""

from __future__ import annotations

import pytest

import lib.ai as ai_mod
from tests.support.fakes import fake_completion_factory


pytestmark = pytest.mark.unit


class TestIsBye:
    def test_true_when_model_returns_one(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "1")

        assert ai_mod.isBye("see you later") is True

    def test_false_when_model_returns_zero(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "0")

        assert ai_mod.isBye("what is the humidity?") is False

    @pytest.mark.parametrize("raw", ["1\n", " 1 ", "1"])
    def test_true_when_model_returns_one_with_whitespace(self, monkeypatch, raw):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: raw)

        assert ai_mod.isBye("goodbye") is True

    def test_true_for_any_nonzero_integer(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "2")

        assert ai_mod.isBye("later") is True

    @pytest.mark.negative
    def test_raises_when_model_returns_non_numeric(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "yes")

        with pytest.raises(ValueError):
            ai_mod.isBye("bye")

    @pytest.mark.negative
    @pytest.mark.regression
    def test_raises_when_think_tags_wrap_the_digit(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "<think>reason</think>\n1")

        with pytest.raises(ValueError):
            ai_mod.isBye("see you later")


class TestAnalyzePrompting:
    def test_sends_weather_payload_and_user_prompt_to_model(self, monkeypatch, weather_payload):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("It is 25C and cloudy.", captured),
        )

        result = ai_mod.analyze(weather_payload, "How hot is it?")

        assert result == "It is 25C and cloudy."
        user_message = captured["messages"][-1]["content"]
        assert "How hot is it?" in user_message
        assert "25" in user_message
        assert captured["model"] == ai_mod.MODEL

    def test_includes_system_prompt_for_brief_plain_replies(self, monkeypatch, weather_payload):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("Sunny.", captured),
        )

        ai_mod.analyze(weather_payload, "summary")

        roles = [message["role"] for message in captured["messages"]]
        assert "system" in roles
        system_text = next(m["content"] for m in captured["messages"] if m["role"] == "system")
        assert "plain text" in system_text.lower()

    def test_forwards_empty_user_prompt_with_payload(self, monkeypatch, weather_payload):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("Need a question.", captured),
        )

        result = ai_mod.analyze(weather_payload, "")

        assert result == "Need a question."
        assert str(weather_payload["location"]) in captured["messages"][-1]["content"]

    def test_returns_think_tags_verbatim_from_model(self, monkeypatch, weather_payload):
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("<think>hidden</think>\nPack a jacket."),
        )

        result = ai_mod.analyze(weather_payload, "What should I wear?")

        assert result == "<think>hidden</think>\nPack a jacket."


class TestByePrompting:
    def test_asks_model_for_binary_exit_signal(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("1", captured),
        )

        result = ai_mod.bye("goodbye")

        assert result == "1"
        assert "goodbye" in captured["messages"][-1]["content"]
        assert "JUST RETURN THE NUMBER" in captured["messages"][-1]["content"]

    def test_uses_configured_model(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("0", captured),
        )

        ai_mod.bye("humidity?")

        assert captured["model"] == ai_mod.MODEL


class TestExtractLocationPrompting:
    def test_asks_model_to_return_city_or_none(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            ai_mod.client.chat.completions,
            "create",
            fake_completion_factory("Mumbai", captured),
        )

        result = ai_mod.extract_location("What is the humidity in Mumbai?")

        assert result == "Mumbai"
        assert "Mumbai" in captured["messages"][-1]["content"]
        assert "NONE" in captured["messages"][-1]["content"]
