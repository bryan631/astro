"""Build the real devices named in config/devices.toml."""

import logging
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from astro.devices.base import Camera
from astro.pointing.coords import Site
from astro.pointing.encoders import COUNTS_PER_REV, EncoderAxis
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.pointing.solve_tracker import SolveTracker

log = logging.getLogger(__name__)


def load(path: Path) -> dict:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("main", {}).setdefault("driver", "none")  # no main camera is fine
    return cfg


def open_camera(cfg: dict) -> Camera:
    from astro.devices.svbony import SvbonyCamera  # needs the vendor SDK; import only when used

    if cfg["driver"] != "svbony":
        raise ValueError(f"unknown camera driver {cfg['driver']!r}")
    cam = SvbonyCamera(cfg["model"])
    try:
        cam.connect()
    except Exception:  # unplugged: start anyway, and keep trying to reconnect on each capture
        log.exception("%s not connected; will keep trying", cfg["model"])
        cam._lost = True
    if "exposure_s" in cfg:
        cam.set_exposure(cfg["exposure_s"])
    if "gain" in cfg:
        cam.set_gain(cfg["gain"])
    return cam


MOUNT_DRIVERS = ("solve", "mcu", "handset")  # handset: IntelliScope RS-232 fallback (P2-3)
CAMERA_DRIVERS = ("svbony",)


def validate(cfg: dict) -> None:
    """Reject unknown drivers before any hardware is touched."""
    for section in ("mount", "finder"):
        if "driver" not in cfg.get(section, {}):
            raise ValueError(f"devices config: [{section}] needs a driver")
    if cfg["mount"]["driver"] not in MOUNT_DRIVERS:
        raise ValueError(f"unknown mount driver {cfg['mount']['driver']!r}")
    if cfg["finder"]["driver"] not in CAMERA_DRIVERS:
        raise ValueError(f"unknown finder driver {cfg['finder']['driver']!r}")
    main = cfg.get("main", {}).get("driver", "none")
    if main not in (*CAMERA_DRIVERS, "none"):
        raise ValueError(f"unknown main camera driver {main!r}")


def build_pointing(cfg: dict, solver: FinderSolver, site: Site, clock: Callable[[], datetime]
                   ) -> tuple[FinderSync | SolveTracker, Callable[[], None]]:
    """Finder camera plus the mount: encoders + model (mcu), or plate solving alone (solve).

    Returns (pointing, close). If anything fails part-way, what was opened is closed again."""
    validate(cfg)  # all drivers known before any hardware opens
    mount = cfg["mount"]
    finder_cam = open_camera(cfg["finder"])
    if mount["driver"] == "solve":
        # Not started here: the session installs its exposure gate first, then starts it.
        tracker = SolveTracker(finder_cam, solver, site, clock)

        def close_tracker() -> None:
            tracker.stop()
            finder_cam.close()

        return tracker, close_tracker
    source = None  # encoder counts: the Nano Every (mcu) or the IntelliScope handset
    try:
        source = _open_encoders(mount)
    except Exception:  # close whatever opened: the serial port and its workers, the camera
        if source is not None:
            source.close()
        finder_cam.close()
        raise
    counts = mount.get("counts_per_rev", COUNTS_PER_REV)
    az = EncoderAxis(counts, mount.get("az_sign", 1))
    alt = EncoderAxis(counts, mount.get("alt_sign", 1))

    def encoders() -> tuple[float, float]:
        az_counts, alt_counts = source.counts()
        return alt.to_degrees(alt_counts), az.to_degrees(az_counts)

    finder = FinderSync(finder_cam, solver, MountModel(), encoders, site, clock)
    finder.raw_counts = source.counts  # shown in the debug view
    finder.encoder_age = source.position_age  # session stops guiding on frozen counts
    if hasattr(source, "on_reboot"):
        source.on_reboot = lambda: finder.reset(finder.site)  # counts reset: model is wrong
        finder.encoder_boots = lambda: source.boots  # restore no saved model after a boot

    def close_encoders() -> None:
        source.close()
        finder_cam.close()

    return finder, close_encoders


def _open_encoders(mount: dict):
    """The encoder source for the mount driver; caller closes it if anything later fails."""
    if mount["driver"] == "handset":
        from astro.devices.handset import Handset

        return Handset(mount.get("port") or "/dev/ttyUSB0")
    from astro.devices.mcu import Mcu

    mcu = Mcu(mount.get("port") or None)
    try:
        return mcu.start()
    except Exception:
        mcu.close()
        raise
