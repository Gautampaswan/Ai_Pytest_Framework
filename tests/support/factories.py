"""Test-data builders for weather payloads used across the suite."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

WEATHER_SCHEMA_KEYS = frozenset(
    {
        "coordinates",
        "country",
        "datetime",
        "description",
        "feels_like",
        "humidity",
        "kind",
        "local_population",
        "location",
        "precipitation",
        "pressure",
        "region",
        "temperature",
        "ultraviolet",
        "visibility",
        "wind_direction",
        "wind_speed",
    }
)


def make_named(value: str | None) -> SimpleNamespace | None:
    if value is None:
        return None
    return SimpleNamespace(name=value)


def make_forecast(**overrides: Any) -> SimpleNamespace:
    """Build a python_weather-like forecast object for mapper tests."""
    defaults: dict[str, Any] = {
        "coordinates": (23.8103, 90.4125),
        "country": "Bangladesh",
        "datetime": datetime(2026, 9, 16, 12, 30),
        "description": "Partly cloudy",
        "feels_like": 86,
        "humidity": 70,
        "kind": make_named("CLOUDY"),
        "local_population": 8906039,
        "location": "Dhaka",
        "precipitation": 0.0,
        "pressure": 1012,
        "region": "Dhaka Division",
        "temperature": 77,
        "ultraviolet": make_named("MODERATE"),
        "visibility": 10,
        "wind_direction": make_named("N"),
        "wind_speed": 8,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_weather_payload(**overrides: Any) -> dict[str, Any]:
    """Build the dict shape produced by lib.weather.getweather."""
    payload: dict[str, Any] = {
        "coordinates": {"X": 23.8103, "Y": 90.4125},
        "country": "Bangladesh",
        "datetime": "2026-09-16T12:30",
        "description": "Partly cloudy",
        "feels_like": 30,
        "humidity": 70,
        "kind": "CLOUDY",
        "local_population": 8906039,
        "location": "Dhaka",
        "precipitation": 0.0,
        "pressure": 1012,
        "region": "Dhaka Division",
        "temperature": 25,
        "ultraviolet": "MODERATE",
        "visibility": 10,
        "wind_direction": "N",
        "wind_speed": 8,
    }
    payload.update(overrides)
    return payload
