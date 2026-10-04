"""Calibration that survives a restart (requirement CV7): data/calibration.json, git-ignored.

Holds what took the user effort to learn: the left/right convention, the main camera's axes,
the finder-to-main offset, and the mount model (only meaningful while the encoder counts
survive, i.e. until the encoder board restarts, and only at the site it was built for).
"""

import json
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from astro.pointing.mount_model import MountModel, Sync

SAVED = Path("data/calibration.json")
VERSION = 1


_save_lock = threading.Lock()  # saves come from the session worker and the MCU reader thread


def load(root: Path) -> dict:
    """The saved calibration, or {} if there is none or it is unreadable in any way."""
    path = root / SAVED
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        _validate(data)
    except (ValueError, KeyError, TypeError, AttributeError):  # start fresh, don't refuse to start
        return {}
    return data


def _validate(data: dict) -> None:
    """Raise if any field has the wrong shape (so load() falls back to a fresh start)."""
    if data["version"] != VERSION:
        raise ValueError("other version")
    if data.get("right_is_plus_az") is not None and not isinstance(data["right_is_plus_az"], bool):
        raise TypeError("right_is_plus_az")
    if data.get("camera_axes") is not None and np.asarray(data["camera_axes"], float).shape != (2, 2):
        raise ValueError("camera_axes")
    if data.get("main_offset") is not None:
        o = data["main_offset"]
        float(o["d_az_sky_deg"]), float(o["d_alt_deg"]), int(o["observations"])
    if data.get("mount") is not None:
        _lat, _lon = (float(v) for v in data["mount"]["site"])
        model_from_dict(data["mount"])


def save(root: Path, data: dict) -> None:
    path = root / SAVED
    path.parent.mkdir(parents=True, exist_ok=True)
    with _save_lock:  # and a unique temp file, so two writers can't clobber each other
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".calibration-", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({**data, "version": VERSION}, f, indent=1)
        os.replace(tmp, path)  # never leave a half-written file


def model_to_dict(model: MountModel) -> dict:
    return {"az_offset_deg": model.az_offset_deg, "alt_offset_deg": model.alt_offset_deg,
            "tilt_n_deg": model.tilt_n_deg, "tilt_e_deg": model.tilt_e_deg,
            "syncs": [asdict(s) for s in model.syncs]}


def model_from_dict(d: dict) -> MountModel:
    return MountModel(float(d["az_offset_deg"]), float(d["alt_offset_deg"]),
                      float(d["tilt_n_deg"]), float(d["tilt_e_deg"]),
                      [Sync(**{k: float(v) for k, v in s.items()}) for s in d["syncs"]])
