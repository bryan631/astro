"""Live stacking for deep-sky targets on an untracked alt-az scope.

Each short exposure drifts and rotates (field rotation) relative to the first, so every
frame is registered to the reference by matching star triangles (astroalign), then added
to a running mean. Frames that can't be matched (clouds, bumps, trails) are skipped.
"""

from dataclasses import dataclass

import astroalign
import numpy as np

from astro.pointing.finder_sync import check_focus
from astro.process.planet import superpixel_rgb

# astroalign matches star triangles (3 minimum); 4 keeps a margin without rejecting double
# stars (PL2), whose short frames show the pair plus only a few field stars.
MIN_STARS = 4
MAX_CONTROL_POINTS = 40  # brightest stars used for matching


@dataclass
class StackStatus:
    frames_added: int
    frames_skipped: int
    last_error: str = ""


class LiveStack:
    def __init__(self, bayer: str):
        self.bayer = bayer
        self._ref_lum: np.ndarray | None = None
        self._sum: np.ndarray | None = None
        self._weight: np.ndarray | None = None  # pixels covered (edges shrink as frames rotate)
        self.status = StackStatus(0, 0)
        self._last_tf = None  # newest frame -> reference transform (None: the reference itself)

    def add(self, raw: np.ndarray) -> bool:
        """Register one raw Bayer frame to the reference and add it. False if it was skipped."""
        rgb = superpixel_rgb(raw, self.bayer)
        lum = rgb.mean(axis=-1)
        if self._ref_lum is None:  # first good frame defines the reference
            stars = check_focus(lum).stars  # only its star count: the focus limits are the finder's
            if stars < MIN_STARS:  # clouds or a capped lens: don't anchor the stack to it
                self.status.frames_skipped += 1
                self.status.last_error = f"only {stars} stars in the first frame"
                return False
            self._ref_lum, self._sum = lum, rgb.copy()
            self._weight = np.ones(lum.shape, np.float32)
            self.status.frames_added = 1
            self._last_tf = None
            return True
        try:
            tf, _ = astroalign.find_transform(lum, self._ref_lum,
                                              max_control_points=MAX_CONTROL_POINTS)
        except (astroalign.MaxIterError, ValueError) as e:  # too few / unmatched stars
            self.status.frames_skipped += 1
            self.status.last_error = str(e)
            return False
        self._last_tf = tf
        warped = [astroalign.apply_transform(tf, rgb[..., c], self._ref_lum) for c in range(3)]
        channels = [image for image, _ in warped]
        covered = ~warped[0][1]  # footprint (same for every channel): True where no data landed
        self._sum += np.stack(channels, axis=-1) * covered[..., None]
        self._weight += covered
        self.status.frames_added += 1
        return True

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the stacked frames (debayered: half the sensor)."""
        return self._ref_lum.shape[1], self._ref_lum.shape[0]

    def reference_center(self) -> tuple[float, float] | None:
        """Where the first frame's center (the field being stacked) is in the newest frame."""
        if self._ref_lum is None:
            return None
        h, w = self._ref_lum.shape
        if self._last_tf is None:
            return w / 2, h / 2
        x, y = self._last_tf.inverse(np.array([[w / 2, h / 2]]))[0]
        return float(x), float(y)

    @property
    def has_frames(self) -> bool:
        return self._sum is not None

    def image(self) -> np.ndarray:
        """Current mean (float RGB); edges only some frames covered are averaged correctly."""
        if self._sum is None:
            raise ValueError("no frames yet")
        return self._sum / np.maximum(self._weight, 1)[..., None]


def stretch(img: np.ndarray, black_pct: float = 25, white_pct: float = 99.8,
            gamma: float = 0.45) -> np.ndarray:
    """Display stretch for faint nebulae: clip the sky background, lift midtones, 8-bit."""
    lo, hi = np.percentile(img, black_pct), np.percentile(img, white_pct)
    scaled = ((img - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    return (scaled**gamma * 255).astype(np.uint8)
