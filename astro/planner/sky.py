"""What's up and where, for the session's answers: target names and positions, what a target
is, when it rises and sets, tonight's plan, and the cloud forecast behind it."""

import threading
import time
from datetime import datetime, timedelta

import astropy.units as u
from astropy.coordinates import SkyCoord, get_body

from astro.intents import match_name
from astro.planner.almanac import compass, target_text
from astro.planner.catalog import Target, load_targets
from astro.planner.horizon import HorizonMask
from astro.planner.moon_features import FEATURES
from astro.planner.tonight import PLANET_NOTES, PLANETS, next_dark, plan
from astro.pointing.coords import Site, body_altaz, radec_to_altaz
from astro.pointing.geometry import separation_deg
from astro.pointing.labels import altaz_now

MOON_FEATURES = {f.name: f for f in FEATURES}  # pointing at one means pointing at the Moon
CLOUD_CACHE_S = 15 * 60  # Open-Meteo is hourly; don't ask on every request
LATER_MIN = 30  # "tonight" more than this far ahead: say when it gets dark
CLOUDY_PCT = 50  # at or above this cloud cover, tonight's suggestions mention the clouds
NOTHING_UP = "Nothing good is up right now."
BODIES = (*PLANETS, "moon")


class Sky:
    def __init__(self, weather=None):
        """`weather(lat, lon, when)`: cloud % or None offline; None: no forecast at all."""
        self.catalog = {t.name: t for t in load_targets()}
        for t in list(self.catalog.values()):
            self.catalog.setdefault(t.id, t)
        self.weather = weather
        self.cached_clouds: float | None = None  # the last forecast, for readers that mustn't fetch
        self._clouds_at, self._clouds_site = -1e9, None
        self._clouds_lock = threading.Lock()

    def targets(self) -> list[Target]:
        """Each Go to target once (the catalog has it by name and by id)."""
        return list({t.id: t for t in self.catalog.values()}.values())

    def names(self) -> list[str]:
        return [p.capitalize() for p in PLANETS] + ["Moon", *MOON_FEATURES, *self.catalog]

    def altaz_of(self, name: str, site: Site, when: datetime) -> tuple[float, float]:
        if name in MOON_FEATURES:  # the Moon fills the main camera: aim at its center
            name = "Moon"
        if name.lower() in PLANETS or name == "Moon":
            return body_altaz(name.lower(), site, when)
        t = self.catalog[name]
        return radec_to_altaz(t.ra, t.dec, site, when)

    def in_field(self, alt: float, az: float, radius_deg: float, site: Site, when: datetime) -> list[str]:
        """Planets, the Moon, then Go to targets within `radius_deg` of (alt, az)."""
        found = [b.capitalize() for b in BODIES
                 if separation_deg(alt, az, *body_altaz(b, site, when)) < radius_deg]
        targets = self.targets()
        t_alt, t_az = altaz_now([t.ra for t in targets], [t.dec for t in targets], site, when)
        return found + [t.name for t, a, z in zip(targets, t_alt, t_az, strict=True)
                        if separation_deg(alt, az, a, z) < radius_deg]

    def nearest_body(self, alt: float, az: float, site: Site, when: datetime) -> tuple[float, str]:
        """(degrees away, name) of the planet or Moon nearest (alt, az)."""
        sep, body = min((separation_deg(alt, az, *body_altaz(b, site, when)), b) for b in BODIES)
        return sep, body.capitalize()

    def describe(self, spoken: str, site: Site, when: datetime, horizon: HorizonMask) -> str:
        """What a target is and where it is right now."""
        name = match_name(spoken, self.names())
        if name is None:
            return f"I don't know {spoken}."
        if name.lower() in PLANETS:
            kind, note = "a planet", PLANET_NOTES[name.lower()]
        elif name == "Moon":
            kind, note = "our Moon", "Craters and mountains show best along the shadow line."
        elif name in MOON_FEATURES:
            kind, note = "a feature on the Moon", MOON_FEATURES[name].note
        else:
            target = self.catalog[name]
            kind, note = f"a {target.category} ({target.id})", target.note
        alt, az = self.altaz_of(name, site, when)
        where = (f"Right now it's about {alt:.0f} degrees up, toward the {compass(az)}."
                 if alt > 0 else "It's below the horizon right now.")
        if 0 < alt < float(horizon.min_alt(az)):
            where += " That's behind the trees from here."
        return f"{name} is {kind}. {note} {where}"

    def timing(self, spoken: str, site: Site, when: datetime, horizon: HorizonMask) -> str:
        """When a target rises, clears the trees, is highest and sets."""
        name = match_name(spoken, self.names())
        if name is None:
            return f"I don't know {spoken}."
        body = "moon" if name in MOON_FEATURES or name == "Moon" else name.lower()
        if body in PLANETS or body == "moon":
            def coord_at(t, loc):
                return get_body(body, t, loc)
        else:
            target = self.catalog[name]

            def coord_at(t, loc):
                return SkyCoord(ra=target.ra * u.deg, dec=target.dec * u.deg)
        return target_text(site, when, name, coord_at, horizon)

    def clouds(self, site: Site, when: datetime) -> float | None:
        """Cloud cover now (%) at `site`; None offline. Cached per place, under its own lock so
        a slow fetch never stalls guidance."""
        here = (site.lat_deg, site.lon_deg)
        with self._clouds_lock:
            stale = time.monotonic() - self._clouds_at > CLOUD_CACHE_S or here != self._clouds_site
            if self.weather is not None and stale:  # a GPS move refetches
                self.cached_clouds = self.weather(*here, when)
                self._clouds_at, self._clouds_site = time.monotonic(), here
            return self.cached_clouds

    def plan(self, site: Site, horizon: HorizonMask, now: datetime
             ) -> tuple[dict, datetime | None, float | None]:
        """(choices by category, when it gets dark, cloud %) for the coming night, in local time
        (asked at 4 PM, this plans the night ahead). ~0.5 s of astropy: call it unlocked."""
        start = next_dark(site, now.astimezone(site.timezone))
        clouds = self.clouds(site, now)
        if start is None:
            return {}, None, clouds
        return plan(site, start, mask=horizon, cloud_cover=clouds), start, clouds

    def target_list(self, site: Site, horizon: HorizonMask, now: datetime, limit: int = 20) -> list[dict]:
        """The Go to list: tonight's best first, each with where it is now and whether it's up
        (above the treeline)."""
        start = next_dark(site, now.astimezone(site.timezone)) or now
        choices = plan(site, start, mask=horizon, cloud_cover=self.clouds(site, now), per_category=6)
        out = []
        for c in sorted((c for cs in choices.values() for c in cs), key=lambda c: -c.score)[:limit]:
            alt, az = self.altaz_of(c.name, site, now)
            out.append({"name": c.name, "category": c.category, "note": c.note,
                        "alt": round(alt, 1), "az": round(az, 1),
                        "up": bool(alt > float(horizon.min_alt(az)))})
        return sorted(out, key=lambda t: not t["up"])  # what can be seen now first, best first


def tonight_text(choices: dict, start: datetime | None, clouds: float | None, now: datetime
                 ) -> tuple[str, list[str] | None]:
    """What to say, and the next suggestions for "next" (None: nothing is up)."""
    flat = sorted((c for cs in choices.values() for c in cs), key=lambda c: -c.score)
    if not flat:
        return NOTHING_UP, None
    best = flat[0]
    others = ", ".join(c.name for c in flat[1:3])
    when = ""
    if start is not None and start - now > timedelta(minutes=LATER_MIN):
        when = f"It's still light out. Once it's dark, around {spoken_clock(start)}: "
    sky = ""
    if clouds is not None and clouds >= CLOUDY_PCT:
        sky = f"It looks about {clouds:.0f} percent cloudy, so it may come and go. "
    return (f"{when}{sky}{best.name} is the best. {best.note} Other good ones: {others}.",
            [c.name for c in flat[1:6]])


def by_category_text(choices: dict, clouds: float | None) -> str:
    """Compact text for the agent: best target per category, local times."""
    lines = [f"{cat}: {cs[0].name} (best around {spoken_clock(cs[0].best_time)}). {cs[0].note}"
             for cat, cs in choices.items()]
    if clouds is not None:
        lines.append(f"cloud cover: about {clouds:.0f}%")
    return "\n".join(lines) or NOTHING_UP


def spoken_clock(t: datetime) -> str:
    """Local wall-clock time as spoken: '9:15 PM'."""
    return t.strftime("%I:%M %p").lstrip("0")
