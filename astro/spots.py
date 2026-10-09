"""Named observing spots, each with its own treeline (data/spots.toml, git-ignored).

The same TOML, stored as the SPOTS_TOML repo secret, feeds the online "Tonight" report.
"""

import json
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from pygeomag import GeoMag

from astro.planner.horizon import HorizonMask
from astro.pointing.coords import Site

SAVED = Path("data/spots.toml")


@dataclass(frozen=True)
class Spot:
    name: str
    site: Site
    mask: HorizonMask


def parse(text: str) -> tuple[list[Spot], str | None]:
    """Spots and the current spot's name from spots.toml text."""
    cfg = tomllib.loads(text)
    spots = [Spot(s["name"], Site(s["lat_deg"], s["lon_deg"], s.get("elevation_m", 0.0)),
                  HorizonMask(tuple(sorted((float(az), float(alt)) for az, alt in s["horizon"]))))
             for s in cfg.get("spot", [])]
    return spots, cfg.get("current")


def dumps(spots: list[Spot], current: str | None) -> str:
    out = ["# Observing spots: name, place and treeline ([azimuth, minimum altitude] degrees)."]
    if current:
        out.append(f"current = {json.dumps(current)}")  # JSON escapes are valid TOML
    for s in spots:
        rows = ", ".join(f"[{az:.0f}, {alt:.0f}]" for az, alt in s.mask.points)
        out += ["", "[[spot]]", f"name = {json.dumps(s.name)}", f"lat_deg = {s.site.lat_deg:.5f}",
                f"lon_deg = {s.site.lon_deg:.5f}", f"elevation_m = {s.site.elevation_m:.0f}",
                f"horizon = [{rows}]"]
    return "\n".join(out) + "\n"


def load(root: Path) -> tuple[list[Spot], str | None]:
    path = root / SAVED
    return parse(path.read_text()) if path.exists() else ([], None)


def save(root: Path, spots: list[Spot], current: str | None) -> None:
    path = root / SAVED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(spots, current))


def upsert(spots: list[Spot], spot: Spot) -> list[Spot]:
    """Add the spot, or replace the one with the same name (case-insensitive)."""
    rest = [s for s in spots if s.name.lower() != spot.name.lower()]
    return [*rest, spot]


def with_mask(spots: list[Spot], name: str, mask: HorizonMask) -> list[Spot]:
    return [replace(s, mask=mask) if s.name.lower() == name.lower() else s for s in spots]


def declination_deg(lat: float, lon: float, when: datetime) -> float:
    """Magnetic declination (east positive, World Magnetic Model): true az = magnetic az + this."""
    year = when.year + (when.timetuple().tm_yday - 0.5) / 365.25
    return float(GeoMag().calculate(glat=lat, glon=lon, alt=0, time=year).d)
