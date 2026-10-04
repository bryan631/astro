"""Quick-look helpers for live camera previews (hardware tools, focus checks)."""

from dataclasses import dataclass

import numpy as np

from astro.capture.focus import laplacian_variance
from astro.pointing.finder_sync import check_focus
from astro.pointing.platesolve import bin2x2

# Position of R, G1, G2, B within each 2x2 Bayer cell, as (row, col).
_LAYOUT = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}


def superpixel_rgb(raw: np.ndarray, bayer: str) -> np.ndarray:
    """Half-resolution RGB from a Bayer frame: one pixel per 2x2 cell, no interpolation."""
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = _LAYOUT[bayer]
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    r = raw[ry:h:2, rx:w:2].astype(np.float32)
    g = (raw[g1y:h:2, g1x:w:2].astype(np.float32) + raw[g2y:h:2, g2x:w:2]) / 2
    b = raw[by:h:2, bx:w:2].astype(np.float32)
    return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)


@dataclass(frozen=True)
class FocusNumbers:
    sharpness: float  # Laplacian variance: planets, Moon, daytime targets (higher = sharper)
    stars: int
    hfr_px: float  # star half-flux radius, binned pixels (lower = sharper)


def focus_numbers(raw: np.ndarray) -> FocusNumbers:
    gray = bin2x2(raw)
    report = check_focus(gray)
    return FocusNumbers(laplacian_variance(gray), report.stars, report.hfr_px)
