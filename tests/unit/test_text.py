"""Unit tests for presentation helpers (CLI output and think-tag stripping)."""

from __future__ import annotations

import pytest

from lib.text import display
from lib.text import format as format_reply
from tests.support.assertions import assert_no_think_tags


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
        assert_no_think_tags(result)

    def test_strips_multiline_think_block(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nTake an umbrella."

        result = format_reply(raw)

        assert result == "Take an umbrella."

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Hello <think>hidden</think> world", "Hello world"),
            ("<think>only</think>", ""),
            ("<think>a</think>\n<think>b</think>\nDone.", "Done."),
        ],
    )
    def test_never_leaks_think_tags(self, raw: str, expected: str):
        result = format_reply(raw)

        assert_no_think_tags(result)
        assert result == expected

    def test_leaves_unclosed_think_block_in_place(self):
        raw = "<think>still reasoning\nCarry a coat."

        result = format_reply(raw)

        assert result == raw
        assert "<think>" in result

    def test_empty_string_is_unchanged(self):
        assert format_reply("") == ""


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

    def test_default_end_adds_blank_line(self, capsys):
        display("AI: ", "OK")

        captured = capsys.readouterr().out

        assert captured.endswith("\n\n")
