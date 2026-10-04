"""Calibration that survives a restart (requirement CV7): data/calibration.json, git-ignored.

Holds what took the user effort to learn: the left/right convention, the main camera's axes,
the finder-to-main offset, and the mount model (only meaningful while the encoder counts
survive, i.e. until the encoder board restarts, and only at the site it was built for).
"""

import json
from dataclasses import asdict
from pathlib import Path

from astro.pointing.mount_model import MountModel, Sync

SAVED = Path("data/calibration.json")
VERSION = 1


def load(root: Path) -> dict:
    path = root / SAVED
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError:  # corrupt file: start fresh rather than refuse to start
        return {}
    return data if data.get("version") == VERSION else {}


def save(root: Path, data: dict) -> None:
    path = root / SAVED
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**data, "version": VERSION}, indent=1))
    tmp.replace(path)  # never leave a half-written file


def model_to_dict(model: MountModel) -> dict:
    return {"az_offset_deg": model.az_offset_deg, "alt_offset_deg": model.alt_offset_deg,
            "tilt_n_deg": model.tilt_n_deg, "tilt_e_deg": model.tilt_e_deg,
            "syncs": [asdict(s) for s in model.syncs]}


def model_from_dict(d: dict) -> MountModel:
    return MountModel(d["az_offset_deg"], d["alt_offset_deg"], d["tilt_n_deg"], d["tilt_e_deg"],
                      [Sync(**s) for s in d["syncs"]])
