"""Curated deep-sky targets loaded from config/targets.toml."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "targets.toml"


@dataclass(frozen=True)
class Target:
    name: str
    id: str
    category: str
    ra: float
    dec: float
    note: str
    size_arcmin: tuple[float, float] = (0.0, 0.0)  # (major, minor); 0: a point (double stars)
    pa_deg: float = 0.0  # major axis, east of north


def load_targets(path: Path = DEFAULT_PATH) -> list[Target]:
    with path.open("rb") as f:
        return [Target(**{**t, "size_arcmin": tuple(t.get("size_arcmin", (0.0, 0.0)))})
                for t in tomllib.load(f)["target"]]
