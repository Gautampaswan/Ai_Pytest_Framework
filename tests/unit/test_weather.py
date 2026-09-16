"""Unit tests for weather mapping, conversions, and failure handling."""

from __future__ import annotations

import pytest
import python_weather

from lib.weather import getweather
from tests.support.factories import WEATHER_SCHEMA_KEYS, make_forecast
from tests.support.fakes import make_async_weather_client


pytestmark = pytest.mark.unit


class TestGetWeatherSuccess:
    @pytest.mark.asyncio
    async def test_maps_forecast_to_contract_schema(self, monkeypatch, forecast):
        client = make_async_weather_client(forecast)
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        result = await getweather("Dhaka")

        assert WEATHER_SCHEMA_KEYS <= result.keys()
        assert result["location"] == "Dhaka"
        assert result["country"] == "Bangladesh"
        assert result["coordinates"] == {"X": 23.8103, "Y": 90.4125}
        assert result["datetime"] == "2026-09-16T12:30"
        client.get.assert_awaited_once_with("Dhaka")

    @pytest.mark.asyncio
    async def test_converts_fahrenheit_temperature_to_celsius(self, monkeypatch):
        forecast = make_forecast(temperature=77, feels_like=86)
        client = make_async_weather_client(forecast)
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        result = await getweather("Dhaka")

        assert result["temperature"] == 25
        assert result["feels_like"] == 30

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "override", "expected"),
        [
            ("kind", None, "UNKNOWN"),
            ("ultraviolet", None, "UNKNOWN"),
            ("wind_direction", None, "UNKNOWN"),
            ("region", None, "Not Eligible"),
        ],
    )
    async def test_applies_fallback_labels(self, monkeypatch, field, override, expected):
        forecast = make_forecast(**{field: override})
        client = make_async_weather_client(forecast)
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        result = await getweather("Dhaka")

        assert result[field] == expected

    @pytest.mark.asyncio
    async def test_requests_imperial_units(self, monkeypatch, forecast):
        seen: dict[str, object] = {}
        client = make_async_weather_client(forecast)

        def factory(*_args, **kwargs):
            seen.update(kwargs)
            return client

        monkeypatch.setattr("lib.weather.python_weather.Client", factory)

        await getweather("London")

        assert seen.get("unit") == python_weather.IMPERIAL


class TestGetWeatherFailures:
    @pytest.mark.asyncio
    async def test_weather_api_error_exits_process(self, monkeypatch):
        client = make_async_weather_client(make_forecast())

        async def _raise(_location: str):
            raise python_weather.Error("upstream down")

        client.get.side_effect = _raise
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        with pytest.raises(SystemExit):
            await getweather("Nowhere")

    @pytest.mark.asyncio
    async def test_unexpected_error_exits_process(self, monkeypatch):
        client = make_async_weather_client(make_forecast())

        async def _raise(_location: str):
            raise RuntimeError("timeout")

        client.get.side_effect = _raise
        monkeypatch.setattr("lib.weather.python_weather.Client", lambda *a, **k: client)

        with pytest.raises(SystemExit):
            await getweather("Dhaka")
