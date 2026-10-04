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


def load_targets(path: Path = DEFAULT_PATH) -> list[Target]:
    with path.open("rb") as f:
        return [Target(**t) for t in tomllib.load(f)["target"]]
