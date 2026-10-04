"""Saved horizon mask for this installation (data/horizon.toml, git-ignored)."""

import tomllib
from pathlib import Path

from astro.planner.horizon import HorizonMask

SAVED = Path("data/horizon.toml")
STELLARIUM = Path("data/horizon_stellarium.txt")


def load(root: Path) -> HorizonMask:
    """The walked horizon, or the default flat 20-degree mask if none was recorded."""
    path = root / SAVED
    if not path.exists():
        return HorizonMask()
    with path.open("rb") as f:
        points = tomllib.load(f)["points"]
    return HorizonMask(tuple((float(az), float(alt)) for az, alt in points))


def save(root: Path, mask: HorizonMask) -> None:
    path = root / SAVED
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join(f"  [{az:.1f}, {alt:.1f}]" for az, alt in mask.points)
    path.write_text(f"# Horizon walk: [azimuth, minimum altitude] in degrees.\npoints = [\n{rows},\n]\n")
    (root / STELLARIUM).write_text(mask.to_stellarium())
