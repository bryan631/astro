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


MOUNT_DRIVERS = ("solve", "mcu")
CAMERA_DRIVERS = ("svbony",)


def validate(cfg: dict) -> None:
    """Reject unknown drivers before any hardware is touched."""
    if cfg["mount"]["driver"] not in MOUNT_DRIVERS:
        raise ValueError(f"unknown mount driver {cfg['mount']['driver']!r}")
    if cfg["finder"]["driver"] not in CAMERA_DRIVERS:
        raise ValueError(f"unknown finder driver {cfg['finder']['driver']!r}")
    if cfg["main"]["driver"] not in (*CAMERA_DRIVERS, "none"):
        raise ValueError(f"unknown main camera driver {cfg['main']['driver']!r}")


def build_pointing(cfg: dict, solver: FinderSolver, site: Site, clock: Callable[[], datetime]
                   ) -> tuple[FinderSync | SolveTracker, Callable[[], None]]:
    """Finder camera plus the mount: encoders + model (mcu), or plate solving alone (solve).

    Returns (pointing, close). If anything fails part-way, what was opened is closed again."""
    validate(cfg)
    mount = cfg["mount"]
    finder_cam = open_camera(cfg["finder"])
    if mount["driver"] == "solve":
        # Not started here: the session installs its exposure gate first, then starts it.
        tracker = SolveTracker(finder_cam, solver, site, clock)

        def close_tracker() -> None:
            tracker.stop()
            finder_cam.close()

        return tracker, close_tracker
    from astro.devices.mcu import Mcu  # mcu

    mcu = None
    try:
        mcu = Mcu(mount.get("port") or None)
        mcu.start()
    except Exception:  # close whatever opened: the serial port and its workers, the camera
        if mcu is not None:
            mcu.close()
        finder_cam.close()
        raise
    az = EncoderAxis(mount.get("counts_per_rev", 9216), mount.get("az_sign", 1))
    alt = EncoderAxis(mount.get("counts_per_rev", 9216), mount.get("alt_sign", 1))

    def encoders() -> tuple[float, float]:
        az_counts, alt_counts = mcu.counts()
        return alt.to_degrees(alt_counts), az.to_degrees(az_counts)

    def close_mcu() -> None:
        mcu.close()
        finder_cam.close()

    return FinderSync(finder_cam, solver, MountModel(), encoders, site, clock), close_mcu
