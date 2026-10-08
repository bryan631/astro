"""Sunrise, sunset and twilight times for the site, as text for Claude."""

from dataclasses import replace
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body
from astropy.time import Time

from astro.pointing.coords import Site

STEP_MIN = 2
# Geometric Sun altitudes (no refraction); -0.833 is the standard sunrise/sunset (refraction
# plus the Sun's radius). Dusk names going down; the dawn names are the same crossings going up.
LEVELS = [(-0.833, "sunset", "sunrise"),
          (-6.0, "civil dusk", "civil dawn"),
          (-12.0, "nautical dusk: dark enough to observe", "nautical dawn: too light to observe"),
          (-18.0, "astronomical dusk: fully dark", "astronomical dawn")]


def sun_events(site: Site, start: datetime, hours: float = 24) -> list[tuple[datetime, str]]:
    """Sunrise, sunset and twilight crossings in the next `hours`, in time order (UTC)."""
    t = Time(start) + np.arange(int(hours * 60 / STEP_MIN) + 1) * STEP_MIN * u.min
    frame = replace(site, pressure_hpa=0).frame(start).replicate_without_data(obstime=t)
    alt = get_body("sun", t, frame.location).transform_to(frame).alt.deg
    events = []
    for level, down, up in LEVELS:
        d = alt - level
        for i in np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
            f = d[i] / (d[i] - d[i + 1])  # linear between the two samples: well under a minute
            when = (t[i] + f * STEP_MIN * u.min).to_datetime(timezone=start.tzinfo)
            events.append((when, down if d[i] > 0 else up))
    return sorted(events)


def sun_text(site: Site, start: datetime) -> str:
    tz = site.timezone
    now = start.astimezone(tz)
    lines = [f"Now {now:%a %I:%M %p} local ({tz.key}); the next 24 hours:"]
    lines += [f"{t.astimezone(tz):%a %I:%M %p}: {name}" for t, name in sun_events(site, start)]
    return "\n".join(lines)
