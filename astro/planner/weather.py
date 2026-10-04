"""Cloud cover from Open-Meteo (no API key). Returns None when offline."""

import json
import urllib.request
from datetime import UTC, datetime


def cloud_cover_pct(lat: float, lon: float, when: datetime, timeout_s: float = 3) -> float | None:
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=cloud_cover&timezone=UTC&forecast_days=2")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            hourly = json.load(r)["hourly"]
        key = when.astimezone(UTC).strftime("%Y-%m-%dT%H:00")
        return float(hourly["cloud_cover"][hourly["time"].index(key)])
    except (OSError, ValueError, KeyError):
        return None
