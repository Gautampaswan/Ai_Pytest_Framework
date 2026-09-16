"""Unit tests for terminal helpers."""

from __future__ import annotations

import pytest

from lib import utils


pytestmark = pytest.mark.unit


class TestClear:
    def test_uses_cls_on_windows(self, monkeypatch):
        commands: list[str] = []
        monkeypatch.setattr(utils, "name", "nt")
        monkeypatch.setattr(utils, "system", lambda cmd: commands.append(cmd) or 0)

        utils.clear()

        assert commands == ["cls"]

    def test_uses_clear_on_posix(self, monkeypatch):
        commands: list[str] = []
        monkeypatch.setattr(utils, "name", "posix")
        monkeypatch.setattr(utils, "system", lambda cmd: commands.append(cmd) or 0)

        utils.clear()

        assert commands == ["clear"]
