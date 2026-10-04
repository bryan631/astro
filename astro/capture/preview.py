"""Quick-look helpers for live camera previews (hardware tools, focus checks)."""

from dataclasses import dataclass

import numpy as np

from astro.capture.focus import laplacian_variance
from astro.pointing.finder_sync import check_focus
from astro.pointing.platesolve import bin2x2


@dataclass(frozen=True)
class FocusNumbers:
    sharpness: float  # Laplacian variance: planets, Moon, daytime targets (higher = sharper)
    stars: int
    hfr_px: float  # star half-flux radius, binned pixels (lower = sharper)


def focus_numbers(raw: np.ndarray) -> FocusNumbers:
    gray = bin2x2(raw)
    report = check_focus(gray)
    return FocusNumbers(laplacian_variance(gray), report.stars, report.hfr_px)
