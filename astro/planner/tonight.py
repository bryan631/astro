"""'What's good tonight?': rank visible targets for a site and night."""

from dataclasses import dataclass
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord, get_body
from astropy.time import Time

from astro.planner.catalog import Target, load_targets
from astro.planner.horizon import HorizonMask
from astro.planner.moon_features import best_features
from astro.pointing.coords import Site

# Weight = how rewarding it is in an 8" scope.
PLANETS = {"mercury": 2.0, "venus": 3.0, "mars": 3.0, "jupiter": 3.5, "saturn": 3.5,
           "uranus": 1.0, "neptune": 0.8}
PLANET_NOTES = {
    "mercury": "Hard to catch; low in twilight.",
    "venus": "Brilliant; shows phases like a small Moon.",
    "mars": "A small orange disk; best when close to Earth.",
    "jupiter": "Cloud bands and four bright moons.",
    "saturn": "The rings! Always a favorite.",
    "uranus": "A tiny blue-green disk.",
    "neptune": "A faint blue dot; a challenge.",
}
DARK_SUN_ALT = -12.0  # nautical twilight is dark enough from the suburbs
STEP_MIN = 15


@dataclass(frozen=True)
class Choice:
    name: str
    category: str
    note: str
    best_time: datetime
    best_alt_deg: float
    minutes_visible: int
    score: float


def night_times(site: Site, start: datetime, hours: float = 14) -> Time:
    """Sample times from `start` for `hours`, keeping only those when the Sun is low enough."""
    steps = int(hours * 60 / STEP_MIN)
    times = Time(start) + np.arange(steps) * STEP_MIN * u.min
    frame = site.frame(start).replicate_without_data(obstime=times)
    sun_alt = get_body("sun", times, frame.location).transform_to(frame).alt.deg
    return times[sun_alt < DARK_SUN_ALT]


def plan(site: Site, start: datetime, mask: HorizonMask | None = None,
         targets: list[Target] | None = None, cloud_cover: float | None = None,
         per_category: int = 3, hours: float = 4) -> dict[str, list[Choice]]:
    """Ranked choices by category for the next `hours`. `cloud_cover` (0-100) lowers scores."""
    mask = mask or HorizonMask()
    targets = load_targets() if targets is None else targets
    times = night_times(site, start, hours)
    if len(times) == 0:
        return {}
    frame = site.frame(start).replicate_without_data(obstime=times)
    moon = get_body("moon", times, frame.location)
    moon_aa = moon.transform_to(frame)
    moon_up = moon_aa.alt.deg > 0
    # Lit fraction from the Sun-Moon elongation: a full Moon brightens the whole sky.
    elongation = get_body("sun", times, frame.location).separation(moon).rad
    moon_lit = (1 - np.cos(elongation)) / 2

    candidates: list[tuple[str, str, str, SkyCoord, float]] = []
    for p, weight in PLANETS.items():
        candidates.append((p.capitalize(), "planet", PLANET_NOTES[p],
                           get_body(p, times, frame.location), weight))
    candidates.append(("Moon", "moon", "Craters and mountains along the shadow line.", moon, 3.0))
    for t in targets:
        coord = SkyCoord(ra=t.ra * u.deg, dec=t.dec * u.deg)
        candidates.append((t.name, t.category, t.note, coord, 1.0))

    out: dict[str, list[Choice]] = {}
    for name, cat, note, coord, weight in candidates:
        aa = coord.transform_to(frame)
        alt, az = aa.alt.deg, aa.az.deg
        visible = alt > mask.min_alt(az)
        if not visible.any():
            continue
        i = int(np.argmax(np.where(visible, alt, -90)))
        score = weight * (alt[i] / 90 + visible.sum() * STEP_MIN / 600)
        if cat not in ("planet", "moon") and moon_up[i]:
            score *= moonlight_factor(cat, aa[i].separation(moon_aa[i]).deg, float(moon_lit[i]))
        if cloud_cover is not None:
            score *= 1 - cloud_cover / 200
        best = Choice(name, cat, note, times[i].to_datetime(timezone=start.tzinfo),
                      round(float(alt[i]), 1), int(visible.sum()) * STEP_MIN, round(score, 3))
        out.setdefault(cat, []).append(best)
    if moon_choice := out.get("moon"):
        out["moon feature"] = _moon_features(moon_choice[0], Time(moon_choice[0].best_time))
    return {c: sorted(v, key=lambda x: -x.score)[:per_category] for c, v in out.items()}


def moon_minus_sun_deg(when: Time) -> float:
    """Moon's ecliptic longitude minus the Sun's: 0 new, 90 first quarter, 180 full."""
    ecl = GeocentricTrueEcliptic(equinox=when)
    moon = get_body("moon", when).transform_to(ecl).lon.deg
    sun = get_body("sun", when).transform_to(ecl).lon.deg
    return float((moon - sun) % 360)


def _moon_features(moon: Choice, when: Time) -> list[Choice]:
    """Features near the shadow line, seen when (and where) the Moon is best."""
    return [Choice(f.name, "moon feature", f.note, moon.best_time, moon.best_alt_deg,
                   moon.minutes_visible, round(moon.score * quality, 3))
            for f, quality in best_features(moon_minus_sun_deg(when))]


def moonlight_factor(category: str, moon_sep_deg: float, moon_lit: float) -> float:
    """Score multiplier for a target while the Moon is up. Faint fuzzies suffer most, doubles
    and clusters less. Near the Moon is worst, but a bright Moon washes out the whole
    (Bortle 8) sky, and a thin crescent hardly matters."""
    worst = 0.5 if category in ("nebula", "galaxy") else 0.15
    near = 1.0 if moon_sep_deg < 30 else 0.5
    return 1 - worst * near * moon_lit


def next_dark(site: Site, now: datetime, within_hours: float = 24) -> datetime | None:
    """Now if it's dark enough, else the next time the Sun is below DARK_SUN_ALT (so "tonight"
    asked in the afternoon plans the coming night). None if it won't get dark (polar summer)."""
    times = Time(now) + np.arange(int(within_hours * 60 / STEP_MIN)) * STEP_MIN * u.min
    frame = site.frame(now).replicate_without_data(obstime=times)
    sun_alt = get_body("sun", times, frame.location).transform_to(frame).alt.deg
    dark = np.flatnonzero(sun_alt < DARK_SUN_ALT)
    if len(dark) == 0:
        return None
    return now if dark[0] == 0 else times[dark[0]].to_datetime(timezone=now.tzinfo)
