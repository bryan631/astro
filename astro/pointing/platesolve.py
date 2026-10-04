"""Finder plate solving with cedar-solve (tetra3): one frame -> RA/Dec/roll.

Roll follows tetra3's convention. cedar-solve pins Pillow<9, so it is installed with
--no-deps (see scripts/setup.sh and CI); Pillow comes from our own dependencies.

The finder (SV905C + 25 mm lens) sees ~11 x 8 deg, inside the bundled database's 10-30 deg
range. Raw Bayer frames are 2x2-binned to grayscale first: faster, and colour is not needed.
"""

import time
from dataclasses import dataclass

import numpy as np

FINDER_FOV_DEG = 11.0  # horizontal, from astro.optics for 1280 px x 31"/px


@dataclass(frozen=True)
class Solution:
    ra_deg: float
    dec_deg: float
    roll_deg: float
    fov_deg: float
    rmse_arcsec: float
    matches: int
    false_prob: float  # probability the match is a coincidence; lower is better
    ms: float


def bin2x2(raw: np.ndarray) -> np.ndarray:
    """Sum each 2x2 Bayer cell into one gray pixel (float32)."""
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    r = raw[:h, :w].astype(np.float32)
    return r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2]


class FinderSolver:
    def __init__(self, fov_deg: float = FINDER_FOV_DEG):
        import tetra3  # heavy import (loads a 14 MB database); keep it out of module import

        self.fov = fov_deg
        self._t3 = tetra3.Tetra3()

    def solve(self, image: np.ndarray, bayer: bool = True, max_false_prob: float = 1e-5,
              timeout_ms: int = 2000) -> Solution | None:
        gray = bin2x2(image) if bayer else image.astype(np.float32)
        t0 = time.perf_counter()
        from PIL import Image

        r = self._t3.solve_from_image(Image.fromarray(gray), fov_estimate=self.fov, fov_max_error=self.fov * 0.15,
                                      solve_timeout=timeout_ms)
        ms = (time.perf_counter() - t0) * 1000
        if r.get("RA") is None or r.get("Prob", 1) > max_false_prob:
            return None
        return Solution(float(r["RA"]), float(r["Dec"]), float(r["Roll"]), float(r["FOV"]),
                        float(r["RMSE"]), int(r["Matches"]), float(r["Prob"]), ms)
