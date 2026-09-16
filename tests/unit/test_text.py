"""Unit tests for presentation helpers (CLI output and think-tag stripping)."""

from __future__ import annotations

import pytest

from lib.text import display
from lib.text import format as format_reply


pytestmark = pytest.mark.unit


class TestFormatReply:
    def test_returns_plain_text_unchanged(self):
        raw = "It is 25C and sunny in Dhaka."

        result = format_reply(raw)

        assert result == raw

    def test_strips_single_think_block(self):
        raw = "<think>internal reasoning</think>\nIt will rain later."

        result = format_reply(raw)

        assert result == "It will rain later."
        assert "<think>" not in result

    def test_strips_multiline_think_block(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nTake an umbrella."

        result = format_reply(raw)

        assert result == "Take an umbrella."

    @pytest.mark.parametrize(
        "raw",
        [
            "Hello <think>hidden</think> world",
            "<think>only</think>",
        ],
    )
    def test_never_leaks_think_tags(self, raw: str):
        result = format_reply(raw)

        assert "<think>" not in result
        assert "</think>" not in result


class TestDisplay:
    def test_prints_role_and_message(self, capsys):
        display("AI: ", "Sunny with light wind.")

        captured = capsys.readouterr().out

        assert "AI: " in captured
        assert "Sunny with light wind." in captured

    def test_end_argument_controls_trailing_newlines(self, capsys):
        display("You: ", end="")

        captured = capsys.readouterr().out

        assert captured.endswith("\n\n") is False
        assert "You: " in captured
