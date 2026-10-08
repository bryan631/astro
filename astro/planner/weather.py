"""Cloud cover from Open-Meteo (no API key). Returns None when offline."""

import http.client
import json
import urllib.request
from datetime import UTC, datetime


def cloud_cover_pct(lat: float, lon: float, when: datetime, timeout_s: float = 3) -> float | None:
    # ~1 km is plenty for a forecast, and says less about where the user lives.
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}"
           "&hourly=cloud_cover&timezone=UTC&forecast_days=2")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            hourly = json.load(r)["hourly"]
        key = when.astimezone(UTC).strftime("%Y-%m-%dT%H:00")
        return float(hourly["cloud_cover"][hourly["time"].index(key)])
    except (OSError, ValueError, KeyError, http.client.HTTPException):  # best effort
        return None


DEW_SPREAD_F = 3  # optics dew up once the air is this close to its dew point
HOURLY = ("cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,precipitation_probability,"
          "temperature_2m,dew_point_2m,wind_speed_10m")


def fetch_hourly(lat: float, lon: float) -> dict:
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}"
           f"&hourly={HOURLY}&current=cloud_cover&timezone=auto&forecast_days=2"
           "&temperature_unit=fahrenheit&wind_speed_unit=mph")
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def forecast_text(lat: float, lon: float, hours: int = 24, get=fetch_hourly) -> str | None:
    """The next `hours` hours, one line each in the site's local time, for Claude to answer
    weather questions from. None when offline."""
    try:
        data = get(lat, lon)
        h, now = data["hourly"], data["current"]
        start = next((i for i, t in enumerate(h["time"]) if t[:13] >= now["time"][:13]), 0)
        lines = [f"Now {now['time'][11:16]} local; cloud cover {now['cloud_cover']}%."]
        for i in range(start, min(start + hours, len(h["time"]))):
            lines.append(
                f"{h['time'][i][5:16].replace('T', ' ')}: clouds {h['cloud_cover'][i]}% "
                f"(low {h['cloud_cover_low'][i]}, mid {h['cloud_cover_mid'][i]}, "
                f"high {h['cloud_cover_high'][i]}), rain {h['precipitation_probability'][i]}%, "
                f"{h['temperature_2m'][i]:.0f}F, dew point {h['dew_point_2m'][i]:.0f}F, "
                f"wind {h['wind_speed_10m'][i]:.0f} mph")
        hours_ahead = range(start, min(start + hours, len(h["time"])))
        dewy = [i for i in hours_ahead if h["temperature_2m"][i] - h["dew_point_2m"][i] <= DEW_SPREAD_F]
        lines.append(f"Dew likely from {h['time'][dewy[0]][5:16].replace('T', ' ')} (air within "
                     f"{DEW_SPREAD_F}F of the dew point): the dew heaters help." if dewy
                     else "Dew unlikely in this period.")
        return "\n".join(lines)
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):  # best effort
        return None
