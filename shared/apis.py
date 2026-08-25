from __future__ import annotations

import time
from typing import Any
from .trace import traced

import httpx

TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_ATTEMPTS = 3
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
HEADERS = {
    "User-Agent": "trip-planner-compare/0.1 (learning project; https://github.com/alekhya-p)"
}

class ApiError(RuntimeError):
    """Any failure reaching or understanding an upstream API."""


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET returning parsed JSON, retrying only transient failures."""
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT,
                follow_redirects=True,
            )
        except httpx.RequestError as exc:
            last_error = f"network error: {type(exc).__name__}"
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUS:
                # A real answer, not a blip — fail now instead of burning retries.
                raise ApiError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")
            last_error = f"HTTP {response.status_code}"

        if attempt < MAX_ATTEMPTS:
            time.sleep(0.4 * 2 ** (attempt - 1))   # 0.4s, then 0.8s

    raise ApiError(f"{url} unreachable after {MAX_ATTEMPTS} attempts ({last_error})")

@traced
def geocode_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates and country.

    Args:
        city: City name, e.g. "Lisbon". May include a country to disambiguate.

    Returns:
        name, country, country_code, latitude, longitude, timezone, population.
    """
    if not city or not city.strip():
        raise ApiError("geocode_city needs a non-empty city name")

    payload = _get(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city.strip(), "count": 1, "language": "en", "format": "json"},
    )
    results = payload.get("results") or []
    if not results:
        raise ApiError(f"no city found matching {city!r}")

    top = results[0]
    return {
        "name": top.get("name"),
        "country": top.get("country"),
        "country_code": top.get("country_code"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone"),
        "population": top.get("population"),
    }

@traced
def get_weather(latitude: float, longitude: float, days: int = 7) -> dict[str, Any]:
    """Fetch a daily weather forecast for a coordinate pair.

    Args:
        latitude: Decimal degrees, from geocode_city.
        longitude: Decimal degrees, from geocode_city.
        days: Forecast length, 1-16.

    Returns:
        avg_high_c, total_precipitation_mm, rainy_days, and a per-day forecast list.
    """
    if not 1 <= days <= 16:
        raise ApiError(f"days must be between 1 and 16, got {days}")

    payload = _get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": days,
            "timezone": "auto",
        },
    )

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        raise ApiError(f"forecast returned no daily data for {latitude},{longitude}")

    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_sum") or []

    forecast = [
        {
            "date": dates[i],
            "temp_max_c": highs[i] if i < len(highs) else None,
            "temp_min_c": lows[i] if i < len(lows) else None,
            "precipitation_mm": rain[i] if i < len(rain) else None,
        }
        for i in range(len(dates))
    ]

    # Do the arithmetic here, not in the LLM — models are unreliable at it.
    usable_highs = [h for h in highs if h is not None]
    usable_rain = [r for r in rain if r is not None]
    return {
        "timezone": payload.get("timezone"),
        "days": len(forecast),
        "avg_high_c": round(sum(usable_highs) / len(usable_highs), 1) if usable_highs else None,
        "total_precipitation_mm": round(sum(usable_rain), 1) if usable_rain else None,
        "rainy_days": sum(1 for r in usable_rain if r >= 1.0),
        "forecast": forecast,
    }

@traced
def get_wikipedia_summary(title: str) -> dict[str, Any]:
    """Fetch the lead summary of a Wikipedia article.

    Args:
        title: Article title or place name, e.g. "Lisbon".

    Returns:
        title, description, extract (the lead paragraph), url.
    """
    if not title or not title.strip():
        raise ApiError("get_wikipedia_summary needs a non-empty title")

    slug = title.strip().replace(" ", "_")
    payload = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")

    # Wikipedia answers "not found" with a 200-ish JSON body, not an error status.
    if payload.get("type", "").endswith("not_found"):
        raise ApiError(f"no Wikipedia article for {title!r}")

    return {
        "title": payload.get("title"),
        "description": payload.get("description"),
        "extract": payload.get("extract"),
        "url": (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
    }

@traced
def get_exchange_rate(base: str, target: str) -> dict[str, Any]:
    """Fetch the latest FX rate between two currencies.

    Args:
        base: ISO-4217 code you hold, e.g. "USD".
        target: ISO-4217 code you will spend, e.g. "EUR".

    Returns:
        base, target, rate, date.
    """
    base, target = base.strip().upper(), target.strip().upper()
    if len(base) != 3 or len(target) != 3:
        raise ApiError(f"currency codes must be 3 letters, got {base!r} and {target!r}")
    if base == target:
        # Frankfurter errors on identical currencies — answer it ourselves.
        return {"base": base, "target": target, "rate": 1.0, "date": "n/a"}

    payload = _get("https://api.frankfurter.dev/v1/latest", {"base": base, "symbols": target})
    rate = (payload.get("rates") or {}).get(target)
    if rate is None:
        raise ApiError(f"no rate available for {base}->{target}")

    return {"base": base, "target": target, "rate": rate, "date": payload.get("date")}
