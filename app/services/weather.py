"""
Real-time weather context for disease risk — free tier (OpenWeatherMap,
1000 calls/day, no card required). Fungal diseases like late blight spread
fastest in specific humidity/temperature bands, so we surface that as a
plain-language risk note alongside the diagnosis, per the original plan's
"Real-time Weather Context" advanced feature.

Get a free key at https://home.openweathermap.org/users/sign_up
"""
import os
from dataclasses import dataclass
from typing import Optional

import requests

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


@dataclass
class WeatherInfo:
    city: str
    temp_c: float
    humidity: int
    description: str
    risk_note: Optional[str]  # None if conditions are low-risk


def _assess_risk(temp_c: float, humidity: int) -> Optional[str]:
    """
    Simple, transparent heuristic (not a substitute for local agronomic
    advice): most fungal crop diseases (late blight, powdery mildew, leaf
    spot) spread fastest in humid, mild conditions. This flags that window
    rather than claiming precision it doesn't have.
    """
    if humidity >= 80 and 15 <= temp_c <= 25:
        return (
            "High humidity and mild temperatures right now — good conditions "
            "for fungal disease to spread. Consider checking your plants more "
            "often over the next few days."
        )
    if humidity >= 85:
        return (
            "Humidity is very high right now, which raises fungal disease risk "
            "even outside the ideal temperature range — keep an eye on your crop."
        )
    return None


def get_weather(city: str) -> WeatherInfo:
    """
    Raises requests.RequestException on network failure, and ValueError if
    OPENWEATHER_API_KEY isn't set or the city isn't found — callers should
    catch these and degrade gracefully (weather is a bonus, not required
    for a diagnosis to work).
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY not set")

    response = requests.get(
        BASE_URL,
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=8,
    )
    if response.status_code == 404:
        raise ValueError(f"City not found: {city}")
    response.raise_for_status()
    data = response.json()

    temp_c = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return WeatherInfo(
        city=city,
        temp_c=temp_c,
        humidity=humidity,
        description=description,
        risk_note=_assess_risk(temp_c, humidity),
    )
