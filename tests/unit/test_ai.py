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

    def test_raises_when_model_returns_non_numeric(self, monkeypatch):
        monkeypatch.setattr(ai_mod, "bye", lambda _text: "yes")

        with pytest.raises(ValueError):
            ai_mod.isBye("bye")


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
