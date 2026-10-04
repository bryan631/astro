"""Build the real devices named in config/devices.toml."""

import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from astro.devices.base import Camera
from astro.pointing.coords import Site
from astro.pointing.encoders import EncoderAxis
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.pointing.solve_tracker import SolveTracker


def load(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def open_camera(cfg: dict) -> Camera:
    from astro.devices.svbony import SvbonyCamera  # needs the vendor SDK; import only when used

    if cfg["driver"] != "svbony":
        raise ValueError(f"unknown camera driver {cfg['driver']!r}")
    cam = SvbonyCamera(cfg["model"])
    cam.connect()
    if "exposure_s" in cfg:
        cam.set_exposure(cfg["exposure_s"])
    if "gain" in cfg:
        cam.set_gain(cfg["gain"])
    return cam


def build_pointing(cfg: dict, solver: FinderSolver, site: Site,
                   clock: Callable[[], datetime]) -> tuple[FinderSync | SolveTracker, Camera]:
    """Finder camera plus the mount: encoders + model (mcu), or plate solving alone (solve)."""
    mount = cfg["mount"]
    if mount["driver"] not in ("solve", "mcu"):  # check config before touching hardware
        raise ValueError(f"unknown mount driver {mount['driver']!r}")
    finder_cam = open_camera(cfg["finder"])
    if mount["driver"] == "solve":
        return SolveTracker(finder_cam, solver, site, clock).start(), finder_cam
    from astro.devices.mcu import Mcu  # mcu

    mcu = Mcu(mount.get("port") or None).start()
    az = EncoderAxis(mount.get("counts_per_rev", 9216), mount.get("az_sign", 1))
    alt = EncoderAxis(mount.get("counts_per_rev", 9216), mount.get("alt_sign", 1))

    def encoders() -> tuple[float, float]:
        az_counts, alt_counts = mcu.counts()
        return alt.to_degrees(alt_counts), az.to_degrees(az_counts)

    return FinderSync(finder_cam, solver, MountModel(), encoders, site, clock), finder_cam
