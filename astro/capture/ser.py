"""SER video writer (the planetary-imaging standard read by PSS, AutoStakkert, Siril).

Spec: http://www.grischa-hahn.homepage.t-online.de/astro/ser/ (v3). 8-bit frames only for now.
"""

import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np

COLOR_IDS = {"MONO": 0, "RGGB": 8, "GRBG": 9, "GBRG": 10, "BGGR": 11}
_HEADER = struct.Struct("<14s7i40s40s40sqq")  # 178 bytes
_EPOCH_OFFSET_TICKS = 621355968000000000  # 0001-01-01 -> 1970-01-01 in 100 ns ticks


def _ticks(t: datetime) -> int:
    return _EPOCH_OFFSET_TICKS + int(t.timestamp() * 1e7)



def _local_offset_ticks() -> int:
    """This machine's UTC offset in SER ticks (100 ns), for the header's local DateTime."""
    return int(datetime.now().astimezone().utcoffset().total_seconds() * 10_000_000)

class SerWriter:
    def __init__(self, path: Path, width: int, height: int, bayer: str = "GRBG",  # SV705C, per hardware checkout
                 instrument: str = "SV705C", telescope: str = "XT8i"):
        self.path, self.shape = Path(path), (height, width)
        self._color = COLOR_IDS[bayer]
        self._meta = (instrument, telescope)
        self._stamps: list[int] = []
        self._f = self.path.open("wb")
        self._f.write(b"\0" * _HEADER.size)  # real header written on close

    def write(self, frame: np.ndarray, when: datetime | None = None) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame of shape {self.shape}, got {frame.dtype} {frame.shape}")
        self._f.write(np.ascontiguousarray(frame).tobytes())
        self._stamps.append(_ticks(when or datetime.now(UTC)))

    def close(self) -> None:
        self._f.write(struct.pack(f"<{len(self._stamps)}q", *self._stamps))
        first = self._stamps[0] if self._stamps else _ticks(datetime.now(UTC))
        h, w = self.shape
        header = _HEADER.pack(b"LUCAM-RECORDER", 0, self._color, 0, w, h, 8, len(self._stamps),
                              b"astro", self._meta[0].encode(), self._meta[1].encode(),
                              first + _local_offset_ticks(), first)  # DateTime is local time
        self._f.seek(0)
        self._f.write(header)
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_ser(path: Path) -> tuple[dict, np.ndarray]:
    """8-bit SER reader. Frames are memory-mapped, so long recordings aren't loaded at once."""
    with Path(path).open("rb") as f:
        head = _HEADER.unpack(f.read(_HEADER.size))
    w, h, n = head[4], head[5], head[7]
    frames = np.memmap(path, np.uint8, "r", offset=_HEADER.size, shape=(n, h, w))
    meta = {"color_id": head[2], "width": w, "height": h, "frames": n,
            "instrument": head[9].rstrip(b"\0").decode()}
    return meta, frames
