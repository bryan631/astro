"""Almanac answers for Claude: Sun and twilight, the Moon, when a target is up, sky events.
All return plain text in the site's local time."""

from dataclasses import replace
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, get_body
from astropy.time import Time

from astro.planner.horizon import HorizonMask
from astro.planner.tonight import PLANETS
from astro.pointing.coords import Site

# Geometric Sun altitudes (no refraction); -0.833 is the standard sunrise/sunset (refraction
# plus the Sun's radius). Dusk names going down; the dawn names are the same crossings going up.
SUN_LEVELS = [(-0.833, "sunset", "sunrise"),
              (-6.0, "civil dusk", "civil dawn"),
              (-12.0, "nautical dusk: dark enough to observe", "nautical dawn: too light to observe"),
              (-18.0, "astronomical dusk: fully dark", "astronomical dawn")]
# Peak nights (month, day), the same every year to within a day; rate = meteors/hour, dark sky.
METEOR_SHOWERS = [((1, 3), "Quadrantids", 110), ((4, 22), "Lyrids", 18),
                  ((5, 6), "Eta Aquariids", 50), ((8, 12), "Perseids", 100),
                  ((10, 21), "Orionids", 20), ((11, 17), "Leonids", 15),
                  ((12, 14), "Geminids", 150), ((12, 22), "Ursids", 10)]


def compass(az: float) -> str:
    """Azimuth in degrees -> 'northeast' etc. (8 points)."""
    points = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    return points[round(az % 360 / 45) % 8]


def _times(start: datetime, hours: float, step_min: float) -> Time:
    return Time(start) + np.arange(int(hours * 60 / step_min) + 1) * step_min * u.min


def _frame(site: Site, start: datetime, t: Time, refraction: bool = True) -> AltAz:
    s = site if refraction else replace(site, pressure_hpa=0)
    return s.frame(start).replicate_without_data(obstime=t)


def _crossings(t: Time, d: np.ndarray, tz) -> list[tuple[datetime, bool]]:
    """Where `d` changes sign: (time, rising) pairs, interpolated between samples."""
    out = []
    for i in np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        f = d[i] / (d[i] - d[i + 1])
        out.append(((t[i] + f * (t[i + 1] - t[i])).to_datetime(timezone=tz), d[i] < 0))
    return out


def _local(site: Site, t: datetime, date: bool = False) -> str:
    """'Fri 06:18 AM', or with the date ('Mon Oct 26 06:00 PM') for events days away."""
    return t.astimezone(site.timezone).strftime("%a %b %d %I:%M %p" if date else "%a %I:%M %p")


def _header(site: Site, start: datetime) -> str:
    return f"Now {_local(site, start)} local ({site.timezone.key})."


# --- Sun -----------------------------------------------------------------------------------
def sun_events(site: Site, start: datetime, hours: float = 24) -> list[tuple[datetime, str]]:
    """Sunrise, sunset and twilight crossings in the next `hours`, in time order."""
    t = _times(start, hours, 2)
    frame = _frame(site, start, t, refraction=False)
    alt = get_body("sun", t, frame.location).transform_to(frame).alt.deg
    events = [(when, up if rising else down)
              for level, down, up in SUN_LEVELS
              for when, rising in _crossings(t, alt - level, start.tzinfo)]
    return sorted(events)


def sun_text(site: Site, start: datetime) -> str:
    lines = [f"{_header(site, start)} The next 24 hours:"]
    lines += [f"{_local(site, when)}: {name}" for when, name in sun_events(site, start)]
    return "\n".join(lines)


# --- Moon ----------------------------------------------------------------------------------
def _moon_lit(t: Time) -> tuple[np.ndarray, np.ndarray]:
    """Fraction lit (0-1) and waxing (bool) at times `t`."""
    sun, moon = get_body("sun", t), get_body("moon", t)
    lit = (1 - np.cos(sun.separation(moon).rad)) / 2
    ahead = (moon.geocentrictrueecliptic.lon - sun.geocentrictrueecliptic.lon).wrap_at(360 * u.deg)
    return np.atleast_1d(lit), np.atleast_1d(ahead.deg < 180)


def phase_name(lit: float, waxing: bool) -> str:
    if lit < 0.03:
        return "new"
    if lit > 0.97:
        return "full"
    side = "waxing" if waxing else "waning"
    if 0.45 <= lit <= 0.55:
        return "first quarter" if waxing else "last quarter"
    return f"{side} {'crescent' if lit < 0.5 else 'gibbous'}"


def moon_text(site: Site, start: datetime) -> str:
    lit, waxing = _moon_lit(Time(start))
    lines = [_header(site, start),
             f"The Moon is {phase_name(lit[0], waxing[0])}, {lit[0]:.0%} lit."]
    t = _times(start, 24, 5)
    frame = _frame(site, start, t)
    aa = get_body("moon", t, frame.location).transform_to(frame)
    alt = aa.alt.deg
    lines.append(f"Right now it's {alt[0]:.0f} degrees up toward the {compass(aa.az.deg[0])}."
                 if alt[0] > 0 else "It's below the horizon right now.")
    lines += [f"{_local(site, when)}: moon{'rise' if rising else 'set'}"
              for when, rising in _crossings(t, alt - 0.125, start.tzinfo)]
    # Next full and new Moon: the lit fraction's turning points over the next 31 days.
    days = _times(start, 31 * 24, 60)
    f, _ = _moon_lit(days)
    seen = set()  # 31 days is longer than a lunar month: keep only the first of each
    for i in range(1, len(f) - 1):
        for phase, turning in (("full", f[i] > 0.9 and f[i - 1] < f[i] >= f[i + 1]),
                               ("new", f[i] < 0.1 and f[i - 1] > f[i] <= f[i + 1])):
            if turning and phase not in seen:
                seen.add(phase)
                when = days[i].to_datetime(timezone=start.tzinfo)
                lines.append(f"Next {phase} Moon: {_local(site, when, True)}")
    return "\n".join(lines)


# --- When a target is up -------------------------------------------------------------------
def target_text(site: Site, start: datetime, name: str, coord_at, mask: HorizonMask) -> str:
    """Rise, clear of the trees, highest, behind the trees and set in the next 24 hours.
    `coord_at(times, location)` gives the target's SkyCoord at those times."""
    t = _times(start, 24, 5)
    frame = _frame(site, start, t)
    aa = coord_at(t, frame.location).transform_to(frame)
    alt, az = aa.alt.deg, aa.az.deg
    trees = np.asarray(mask.min_alt(az), dtype=float)
    lines = [_header(site, start), f"{name}, the next 24 hours:"]
    lines.append(f"Right now: {alt[0]:.0f} degrees up toward the {compass(az[0])}"
                 + (" (behind the trees)." if 0 < alt[0] < trees[0] else "." if alt[0] > 0
                    else " (below the horizon)."))
    events = [(w, "rises" if r else "sets") for w, r in _crossings(t, alt, start.tzinfo)]
    if trees.any():
        events += [(w, "clears the trees" if r else "goes behind the trees")
                   for w, r in _crossings(t, alt - np.maximum(trees, 0), start.tzinfo)]
    i = int(np.argmax(alt))
    if alt[i] > 0:
        events.append((t[i].to_datetime(timezone=start.tzinfo),
                       f"highest, {alt[i]:.0f} degrees up toward the {compass(az[i])}"))
    if alt.min() > 0:
        lines.append("It never sets from here.")
    elif alt.max() <= 0:
        lines.append("It doesn't rise in the next 24 hours.")
    lines += [f"{_local(site, w)}: {what}" for w, what in sorted(events)]
    return "\n".join(lines)


# --- Sky events ----------------------------------------------------------------------------
def events_text(site: Site, start: datetime, days: int = 30) -> str:
    """Meteor shower peaks and close Moon-planet / planet-planet pairings in the next `days`."""
    tz = site.timezone
    today = start.astimezone(tz).date()
    events = []
    for (month, day), name, rate in METEOR_SHOWERS:
        for year in (today.year, today.year + 1):
            peak = datetime(year, month, day, tzinfo=tz)
            if 0 <= (peak.date() - today).days <= days:
                events.append((peak, (f"{peak:%a %b %d}: {name} meteor shower peaks (up to about "
                                      f"{rate} an hour from a dark site; best after midnight)")))
    t = _times(start, days * 24, 60)
    bodies = {b: get_body(b, t) for b in ("moon", *PLANETS)}
    names = list(bodies)
    for a, first in enumerate(names):
        for second in names[a + 1:]:
            limit = 4.0 if first == "moon" else 2.0
            sep = bodies[first].separation(bodies[second]).deg
            for i in range(1, len(sep) - 1):
                if sep[i] < limit and sep[i - 1] > sep[i] <= sep[i + 1]:
                    when = t[i].to_datetime(timezone=start.tzinfo)
                    events.append((when, (f"{_local(site, when, True)}: {first.capitalize()} and "
                                          f"{second.capitalize()} are {sep[i]:.1f} degrees apart")))
    lines = [f"{_header(site, start)} Sky events in the next {days} days:"]
    lines += [text for _, text in sorted(events)] or ["No meteor shower peaks or close pairings."]
    return "\n".join(lines)
