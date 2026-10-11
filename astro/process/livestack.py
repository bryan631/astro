"""Live stacking for deep-sky targets on an untracked alt-az scope.

Each short exposure drifts and rotates (field rotation) relative to the first, so every
frame is registered to the reference by matching star triangles (astroalign), then added
to a running mean. When too few stars make triangles (0.2 s frames of a faint field), the
drift alone is measured by phase correlation: over a few minutes it is nearly a pure shift.
Frames that match neither way (clouds, bumps, trails) are skipped.
"""

import logging
from dataclasses import dataclass

import astroalign
import numpy as np
from scipy import ndimage
from skimage.transform import SimilarityTransform

from astro.pointing.finder_sync import check_focus
from astro.process.planet import superpixel_rgb
from astro.process.view import remove_hot_pixels

# astroalign matches star triangles (3 minimum); 4 keeps a margin without rejecting double
# stars (PL2), whose short frames show the pair plus only a few field stars.
MIN_STARS = 4
MAX_CONTROL_POINTS = 40  # brightest stars used for matching
# The shift fallback: triangle matching skipped 16-19 of ~30 frames of the Dumbbell, and the
# stack stopped with 7-14 frames (2026-10-10).
STAR_SMOOTH_PX = 2.0  # stars are wider than this; pixel noise isn't
HIGHPASS_PX = 10  # the sky's glow and gradients are smoother than this; stars much smaller
MIN_PEAK_SIGMA = 16.0  # a real pair peaked at 23; noise and an unrelated field at 8-10 (2026-10-10)
log = logging.getLogger(__name__)


@dataclass
class StackStatus:
    frames_added: int
    frames_skipped: int


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
                log.info("stack: first frame skipped", extra={"data": {"stars": stars}})
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
            shift, strength = measure_shift(lum, self._ref_lum)
            if strength < MIN_PEAK_SIGMA:
                self.status.frames_skipped += 1
                log.info("stack: frame skipped", extra={"data": {"why": str(e), "shift_peak": round(strength, 1)}})
                return False
            tf = SimilarityTransform(translation=shift)
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


def _highpass(lum: np.ndarray) -> np.ndarray:
    """Stars on black: hot pixels out and the pixel noise smoothed (the sensor's fixed pattern
    matched at zero shift over the faint stars, 2026-10-10), the smooth sky off, and nothing
    below it (noise and dark lanes)."""
    lum = ndimage.gaussian_filter(remove_hot_pixels(lum[..., None])[..., 0], STAR_SMOOTH_PX)
    img = lum - ndimage.gaussian_filter(lum, HIGHPASS_PX)
    return np.clip(img, 0, np.percentile(img, 99.9))  # one bright star can't decide alone


def measure_shift(lum: np.ndarray, ref: np.ndarray) -> tuple[tuple[float, float], float]:
    """(dx, dy) that moves `lum` onto `ref` (a feature at p in lum sits at p + shift in ref),
    by phase correlation, and the peak's height in noise sigmas (low: no real match)."""
    cross = np.fft.rfft2(_highpass(ref)) * np.conj(np.fft.rfft2(_highpass(lum)))
    surface = np.fft.irfft2(cross / np.maximum(np.abs(cross), 1e-9), s=ref.shape)
    iy, ix = np.unravel_index(np.argmax(surface), surface.shape)
    strength = float((surface[iy, ix] - surface.mean()) / max(float(surface.std()), 1e-12))
    h, w = surface.shape
    dy, dx = (_subpixel(surface[(iy + np.arange(-1, 2)) % h, ix]) + iy,
              _subpixel(surface[iy, (ix + np.arange(-1, 2)) % w]) + ix)
    return (float((dx + w / 2) % w - w / 2), float((dy + h / 2) % h - h / 2)), strength


def _subpixel(three: np.ndarray) -> float:
    """The peak's offset from the middle of three samples (a parabola through them)."""
    a, b, c = three
    den = a - 2 * b + c
    return float(0.5 * (a - c) / den) if den else 0.0
