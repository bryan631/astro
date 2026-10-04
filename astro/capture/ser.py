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


class SerWriter:
    def __init__(self, path: Path, width: int, height: int, bayer: str = "RGGB",
                 instrument: str = "SV705C", telescope: str = "XT8i"):
        self.path, self.size = Path(path), (height, width)
        self._color = COLOR_IDS[bayer]
        self._meta = (instrument, telescope)
        self._stamps: list[int] = []
        self._f = self.path.open("wb")
        self._f.write(b"\0" * _HEADER.size)  # real header written on close

    def write(self, frame: np.ndarray, when: datetime | None = None) -> None:
        if frame.shape != self.size or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame of shape {self.size}, got {frame.dtype} {frame.shape}")
        self._f.write(np.ascontiguousarray(frame).tobytes())
        self._stamps.append(_ticks(when or datetime.now(UTC)))

    def close(self) -> None:
        self._f.write(struct.pack(f"<{len(self._stamps)}q", *self._stamps))
        first = self._stamps[0] if self._stamps else _ticks(datetime.now(UTC))
        h, w = self.size
        header = _HEADER.pack(b"LUCAM-RECORDER", 0, self._color, 0, w, h, 8, len(self._stamps),
                              b"astro", self._meta[0].encode(), self._meta[1].encode(), first, first)
        self._f.seek(0)
        self._f.write(header)
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_ser(path: Path) -> tuple[dict, np.ndarray]:
    """Minimal reader (8-bit) used by tests and quick checks."""
    data = Path(path).read_bytes()
    f = _HEADER.unpack(data[: _HEADER.size])
    w, h, n = f[4], f[5], f[7]
    frames = np.frombuffer(data, np.uint8, n * w * h, _HEADER.size).reshape(n, h, w)
    return {"color_id": f[2], "width": w, "height": h, "frames": n, "instrument": f[9].rstrip(b"\0").decode()}, frames
