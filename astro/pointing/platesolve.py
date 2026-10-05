"""Finder plate solving with cedar-solve (tetra3): one frame -> RA/Dec/roll.

Angles are degrees at the interface (astronomy convention, matching astropy and tetra3);
roll follows tetra3's convention. Install the solver with scripts/install-solver.sh.

The finder (SV905C + 25 mm lens) sees ~11 x 8 deg, inside the bundled database's 10-30 deg
range. Raw Bayer frames are 2x2-binned to grayscale first: faster, and color is not needed.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import tetra3
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy import ndimage

# Horizontal; measured by the first real-sky solve (astro.optics for 1280 px x 31"/px said ~11).
FINDER_FOV_DEG = 10.4
# A hot pixel stands this far (ADU) above all 8 neighbors while those neighbors stay near the
# local background; starlight always lifts the neighbors. 20 ADU flags 15 pixels on a capped
# SV905C frame at gain 1000 and spares stars.
HOT_PIXEL_ADU = 20
_RING = np.ones((3, 3), bool)
_RING[1, 1] = False  # the 8 neighbors


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
    scale_arcsec_px: float = 0.0  # sky angle per sensor pixel, from the solved field of view

    @property
    def confidence(self) -> float:
        """-log10(false_prob): 12 means a 1e-12 chance of a coincidental match. Higher is better."""
        return -math.log10(max(self.false_prob, 1e-300))


def bin2x2(raw: np.ndarray) -> np.ndarray:
    """Sum each 2x2 Bayer cell into one gray pixel (float32)."""
    h, w = raw.shape[0] // 2, raw.shape[1] // 2
    return raw[: 2 * h, : 2 * w].reshape(h, 2, w, 2).sum(axis=(1, 3), dtype=np.float32)


def remove_hot_pixels(raw: np.ndarray) -> np.ndarray:
    """Replace isolated hot pixels with the median of their 8 neighbors (float32 result)."""
    img = raw.astype(np.float32)
    neighbors = ndimage.maximum_filter(img, footprint=_RING, mode="nearest")
    ys, xs = np.nonzero(img - neighbors > HOT_PIXEL_ADU)  # few candidates: check only those
    if len(ys) == 0:
        return img
    # 5x5 patch around each candidate, all at once (edge-padded so borders work too).
    patches = sliding_window_view(np.pad(img, 2, mode="edge"), (5, 5))[ys, xs]
    background = np.median(patches.reshape(len(ys), 25), axis=1)
    hot = neighbors[ys, xs] - background < HOT_PIXEL_ADU  # neighbors dark: not a star
    ring = patches[:, 1:4, 1:4].reshape(len(ys), 9)[:, _RING.ravel()]
    img[ys[hot], xs[hot]] = np.median(ring[hot], axis=1)
    return img


def finder_gray(raw: np.ndarray) -> np.ndarray:
    """Raw finder frame -> hot pixels removed -> 2x2-binned gray. Used for focus and solving."""
    return bin2x2(remove_hot_pixels(raw))


class FinderSolver:
    def __init__(self, fov_deg: float = FINDER_FOV_DEG):
        self.fov = fov_deg
        self._t3 = tetra3.Tetra3()  # loads the bundled 14 MB database

    @property
    def star_table(self) -> np.ndarray:
        """The solver's star catalog (the simulated finder renders the sky from it)."""
        return self._t3.star_table

    def solve(self, image: np.ndarray, bayer: bool = True, max_false_prob: float = 1e-5,
              timeout_ms: int = 2000, binned: int = 1) -> Solution | None:
        """`binned`: how many sensor pixels one gray pixel spans (2 for finder_gray output)."""
        gray = finder_gray(image) if bayer else image.astype(np.float32)
        t0 = time.perf_counter()
        r = self._t3.solve_from_image(Image.fromarray(gray), fov_estimate=self.fov,
                                      fov_max_error=self.fov * 0.15, solve_timeout=timeout_ms,
                                      match_threshold=max_false_prob,
                                      # Sharp stars are only 1-2 px after binning; tetra3's default
                                      # morphological opening erases them (21/30 -> 28/30 solved
                                      # on simulated finder frames).
                                      binary_open=False)
        ms = (time.perf_counter() - t0) * 1000
        if r.get("RA") is None or r.get("Prob", 1) > max_false_prob:
            return None
        width_px = image.shape[1] * binned  # sensor pixels
        return Solution(float(r["RA"]), float(r["Dec"]), float(r["Roll"]), float(r["FOV"]),
                        float(r["RMSE"]), int(r["Matches"]), float(r["Prob"]), ms,
                        float(r["FOV"]) * 3600 / width_px)
