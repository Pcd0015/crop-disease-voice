"""
Tests the weather risk logic without needing a live OpenWeatherMap key or
network access — HTTP is mocked.

Run with: pytest tests/test_weather.py -v
"""
import os
from unittest.mock import patch, MagicMock

import pytest

from app.services.weather import get_weather, _assess_risk


def _mock_response(temp_c, humidity, description="clear sky"):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "main": {"temp": temp_c, "humidity": humidity},
        "weather": [{"description": description}],
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_high_humidity_mild_temp_flags_risk():
    assert _assess_risk(temp_c=20, humidity=85) is not None


def test_low_humidity_no_risk():
    assert _assess_risk(temp_c=20, humidity=40) is None


def test_very_high_humidity_flags_risk_regardless_of_temp():
    assert _assess_risk(temp_c=35, humidity=90) is not None


@patch.dict(os.environ, {"OPENWEATHER_API_KEY": "fake_key_for_test"})
@patch("app.services.weather.requests.get")
def test_get_weather_returns_risk_note_in_risky_conditions(mock_get):
    mock_get.return_value = _mock_response(temp_c=22, humidity=88, description="light rain")

    result = get_weather("Pune")

    assert result.city == "Pune"
    assert result.temp_c == 22
    assert result.humidity == 88
    assert result.risk_note is not None


@patch.dict(os.environ, {"OPENWEATHER_API_KEY": "fake_key_for_test"})
@patch("app.services.weather.requests.get")
def test_get_weather_no_risk_note_in_dry_conditions(mock_get):
    mock_get.return_value = _mock_response(temp_c=30, humidity=35, description="clear sky")

    result = get_weather("Pune")

    assert result.risk_note is None


@patch.dict(os.environ, {}, clear=True)
def test_get_weather_raises_without_api_key():
    with pytest.raises(ValueError):
        get_weather("Pune")


@patch.dict(os.environ, {"OPENWEATHER_API_KEY": "fake_key_for_test"})
@patch("app.services.weather.requests.get")
def test_get_weather_raises_on_city_not_found(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    with pytest.raises(ValueError):
        get_weather("Nonexistentville")
