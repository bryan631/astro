"""Observing-site storage: the tablet's GPS fix overrides the checked-in default.

`data/site.toml` is per installation and git-ignored, so real coordinates never reach the repo.
"""

import tomllib
from pathlib import Path

from astro.pointing.coords import Site

DEFAULT = Path("config/site.toml")
SAVED = Path("data/site.toml")


def has_saved(root: Path) -> bool:
    return (root / SAVED).exists()


def load(root: Path) -> Site:
    path = root / SAVED if has_saved(root) else root / DEFAULT
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    return Site(cfg["lat_deg"], cfg["lon_deg"], cfg.get("elevation_m", 0.0))


def save(root: Path, site: Site) -> None:
    path = root / SAVED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# From the tablet's GPS. Delete to fall back to {DEFAULT}.\n"
                    f"lat_deg = {site.lat_deg:.5f}\nlon_deg = {site.lon_deg:.5f}\n"
                    f"elevation_m = {site.elevation_m:.0f}\n")
