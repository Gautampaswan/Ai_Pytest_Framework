"""Drives the CLI agent with injected I/O — the test equivalent of a UI driver."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import main


@dataclass
class AgentRunResult:
    stdout: str
    inputs: list[str]
    locations: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    analyze_calls: list[tuple[dict[str, Any], str]] = field(default_factory=list)


class WeatherAgentHarness:
    """Runs main.main() without live weather or LLM calls."""

    def __init__(
        self,
        monkeypatch,
        capsys,
        weather_payload: dict[str, Any],
        *,
        analyze_reply: str | Callable[[dict[str, Any], str], str] = "It is 25C and cloudy.",
        is_bye: Callable[[str], bool] | None = None,
    ) -> None:
        self._monkeypatch = monkeypatch
        self._capsys = capsys
        self._weather_payload = weather_payload
        self._analyze_reply = analyze_reply
        self._is_bye = is_bye or (lambda text: "bye" in text.lower())
        self.locations: list[str] = []
        self.prompts: list[str] = []
        self.analyze_calls: list[tuple[dict[str, Any], str]] = []

    def _install(self, inputs: Sequence[str]) -> None:
        queue = list(inputs)

        def fake_input(_prompt: str = "") -> str:
            if not queue:
                raise AssertionError("Agent asked for more input than the test provided.")
            return queue.pop(0)

        async def fake_getweather(location: str) -> dict[str, Any]:
            self.locations.append(location)
            return self._weather_payload

        def fake_analyze(api: dict[str, Any], prompt: str) -> str:
            self.prompts.append(prompt)
            self.analyze_calls.append((api, prompt))
            if callable(self._analyze_reply):
                return self._analyze_reply(api, prompt)
            return self._analyze_reply

        self._monkeypatch.setattr("builtins.input", fake_input)
        self._monkeypatch.setattr(main, "getweather", fake_getweather)
        self._monkeypatch.setattr(main, "analyze", fake_analyze)
        self._monkeypatch.setattr(main, "isBye", self._is_bye)
        self._monkeypatch.setattr(main, "clear", lambda: None)

    def run(self, inputs: Iterable[str]) -> AgentRunResult:
        supplied = list(inputs)
        self._install(supplied)
        main.main()
        return AgentRunResult(
            stdout=self._capsys.readouterr().out,
            inputs=supplied,
            locations=list(self.locations),
            prompts=list(self.prompts),
            analyze_calls=list(self.analyze_calls),
        )
