"""Real finder frames recorded with scripts/hwcheck/capture_cases.py (see tests/data/finder)."""

import json
from pathlib import Path

import numpy as np

FINDER = Path(__file__).parent / "data" / "finder"


def manifest() -> list[dict]:
    return json.loads((FINDER / "manifest.json").read_text())


def load(name: str) -> np.ndarray:
    """Raw Bayer frame by file stem, e.g. load("lens_cap")."""
    return np.load(FINDER / f"{name}.npz")["raw"]
