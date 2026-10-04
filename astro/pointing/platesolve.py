"""Finder plate solving with cedar-solve (tetra3): one frame -> RA/Dec/roll.

Angles are degrees at the interface (astronomy convention, matching astropy and tetra3);
roll follows tetra3's convention. Install the solver with scripts/install-solver.sh.

The finder (SV905C + 25 mm lens) sees ~11 x 8 deg, inside the bundled database's 10-30 deg
range. Raw Bayer frames are 2x2-binned to grayscale first: faster, and colour is not needed.
"""

import time
from dataclasses import dataclass

import numpy as np
import tetra3
from PIL import Image

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
    h, w = raw.shape[0] // 2, raw.shape[1] // 2
    return raw[: 2 * h, : 2 * w].reshape(h, 2, w, 2).sum(axis=(1, 3), dtype=np.float32)


class FinderSolver:
    def __init__(self, fov_deg: float = FINDER_FOV_DEG):
        self.fov = fov_deg
        self._t3 = tetra3.Tetra3()  # loads the bundled 14 MB database

    def solve(self, image: np.ndarray, bayer: bool = True, max_false_prob: float = 1e-5,
              timeout_ms: int = 2000) -> Solution | None:
        gray = bin2x2(image) if bayer else image.astype(np.float32)
        t0 = time.perf_counter()
        r = self._t3.solve_from_image(Image.fromarray(gray), fov_estimate=self.fov,
                                      fov_max_error=self.fov * 0.15, solve_timeout=timeout_ms,
                                      match_threshold=max_false_prob)
        ms = (time.perf_counter() - t0) * 1000
        if r.get("RA") is None or r.get("Prob", 1) > max_false_prob:
            return None
        return Solution(float(r["RA"]), float(r["Dec"]), float(r["Roll"]), float(r["FOV"]),
                        float(r["RMSE"]), int(r["Matches"]), float(r["Prob"]), ms)
