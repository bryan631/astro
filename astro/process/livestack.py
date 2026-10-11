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
import cv2
import numpy as np
from scipy import fft
from skimage.transform import SimilarityTransform

from astro.pointing.finder_sync import check_focus
from astro.process.planet import superpixel_rgb
from astro.process.view import HOT_SIGMA

# astroalign matches star triangles (3 minimum); 4 keeps a margin without rejecting double
# stars (PL2), whose short frames show the pair plus only a few field stars.
MIN_STARS = 4
MAX_CONTROL_POINTS = 40  # brightest stars used for matching
# The shift fallback: triangle matching skipped 16-19 of ~30 frames of the Dumbbell, and the
# stack stopped with 7-14 frames (2026-10-10).
STAR_SMOOTH_PX = 2.0  # stars are wider than this; pixel noise isn't
HIGHPASS_PX = 10  # the sky's glow and gradients are smoother than this; stars much smaller
MIN_PEAK_SIGMA = 16.0  # a real pair peaked at 23; noise and an unrelated field at 8-10 (2026-10-10)
MIN_COVER = 0.5  # the finished picture keeps pixels at least this share of the frames reached
# OpenCV names a Bayer pattern by its second row: an RGGB sensor is its BayerBG.
DEMOSAIC = {"RGGB": cv2.COLOR_BayerBG2RGB, "GRBG": cv2.COLOR_BayerGB2RGB,
            "GBRG": cv2.COLOR_BayerGR2RGB, "BGGR": cv2.COLOR_BayerRG2RGB}
RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)  # a pixel's eight neighbors
log = logging.getLogger(__name__)


@dataclass
class StackStatus:
    frames_added: int
    frames_skipped: int


class LiveStack:
    def __init__(self, bayer: str, full_res: bool = False):
        """`full_res`: debayer to every pixel (the finder's small sensor) instead of one pixel
        per 2x2 cell (the main camera's, whose stars span several cells anyway)."""
        self.bayer, self.full_res = bayer, full_res
        self._ref_lum: np.ndarray | None = None
        self._ref_spectrum: np.ndarray | None = None  # for the shift fallback, made once
        self._ref_stars: np.ndarray | None = None  # the reference's stars, found once
        self._sum: np.ndarray | None = None
        self._weight: np.ndarray | None = None  # pixels covered (edges shrink as frames rotate)
        self.status = StackStatus(0, 0)
        self._last_tf = None  # newest frame -> reference transform (None: the reference itself)

    def add(self, raw: np.ndarray) -> bool:
        """Register one raw Bayer frame to the reference and add it. False if it was skipped."""
        rgb = debayer(raw, self.bayer) if self.full_res else superpixel_rgb(raw, self.bayer)
        lum = rgb.mean(axis=-1)
        if self._ref_lum is None:  # first good frame defines the reference
            stars = check_focus(lum).stars  # only its star count: the focus limits are the finder's
            if stars < MIN_STARS:  # clouds or a capped lens: don't anchor the stack to it
                self.status.frames_skipped += 1
                log.info("stack: first frame skipped", extra={"data": {"stars": stars}})
                return False
            self._ref_lum, self._sum = lum, rgb.copy()
            self._ref_spectrum = spectrum(lum)
            self._ref_stars = astroalign._find_sources(lum)[:MAX_CONTROL_POINTS]
            self._weight = np.ones(lum.shape, np.float32)
            self.status.frames_added = 1
            self._last_tf = None
            return True
        try:
            tf, _ = astroalign.find_transform(lum, self._ref_stars,
                                              max_control_points=MAX_CONTROL_POINTS)
        except (astroalign.MaxIterError, ValueError) as e:  # too few / unmatched stars
            shift, strength = measure_shift(lum, self._ref_spectrum)
            if strength < MIN_PEAK_SIGMA:
                self.status.frames_skipped += 1
                log.info("stack: frame skipped", extra={"data": {"why": str(e), "shift_peak": round(strength, 1)}})
                return False
            tf = SimilarityTransform(translation=shift)
        self._last_tf = tf
        # OpenCV: ~15x faster than astroalign's warps (250 ms a frame kept 1 in 10, 2026-10-11)
        h, w = lum.shape
        warped = cv2.warpAffine(rgb, tf.params[:2], (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan,) * 3)
        covered = ~np.isnan(warped[..., 0])  # where this frame landed
        self._sum += np.nan_to_num(warped)
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

    def picture(self) -> np.ndarray:
        """image() without the edges fewer than MIN_COVER of the frames reached: as the field
        drifts and turns, a frame's edge lands on few frames' worth of data, and those bands
        stayed noisy and off-color in the finished picture (2026-10-11)."""
        img = self.image()
        enough = self._weight >= MIN_COVER * self._weight.max()
        rows, cols = np.flatnonzero(enough.mean(axis=1) > 0.5), np.flatnonzero(enough.mean(axis=0) > 0.5)
        if len(rows) < 16 or len(cols) < 16:  # nothing sensible to keep: show it all
            return img
        return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def debayer(raw: np.ndarray, bayer: str) -> np.ndarray:
    """Full-resolution float RGB of a raw Bayer frame (OpenCV's edge-aware bilinear)."""
    return cv2.cvtColor(raw, DEMOSAIC[bayer]).astype(np.float32)


def stretch(img: np.ndarray, black_pct: float = 25, white_pct: float = 99.8,
            gamma: float = 0.45) -> np.ndarray:
    """Display stretch for faint nebulae: clip the sky background, lift midtones, 8-bit."""
    lo, hi = np.percentile(img, black_pct), np.percentile(img, white_pct)
    scaled = ((img - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    return (scaled**gamma * 255).astype(np.uint8)


def _highpass(lum: np.ndarray) -> np.ndarray:
    """Stars on black: hot pixels out (a pixel far above all eight neighbors gets the
    brightest of them) and the pixel noise smoothed (the sensor's fixed pattern matched at zero
    shift over the faint stars, 2026-10-10), the smooth sky off, and nothing below it (noise
    and dark lanes). OpenCV filters: scipy's took 0.4 s a frame."""
    lum = lum.astype(np.float32)
    neighbors = cv2.dilate(lum, RING)
    lum = np.where(lum - neighbors > HOT_SIGMA * (np.std(lum[::8, ::8]) + 1e-6), neighbors, lum)
    lum = cv2.GaussianBlur(lum, (0, 0), STAR_SMOOTH_PX)
    img = lum - cv2.GaussianBlur(lum, (0, 0), HIGHPASS_PX)
    return np.clip(img, 0, np.percentile(img, 99.9))  # one bright star can't decide alone


def spectrum(lum: np.ndarray) -> np.ndarray:
    """A frame's high-passed spectrum for measure_shift (the reference's is made once)."""
    return fft.rfft2(_highpass(lum), workers=-1)


def measure_shift(lum: np.ndarray, ref: np.ndarray) -> tuple[tuple[float, float], float]:
    """(dx, dy) that moves `lum` onto `ref` (a frame, or its spectrum()) (a feature at p in lum
    sits at p + shift in ref), by phase correlation, and the peak's height in noise sigmas
    (low: no real match)."""
    ref = spectrum(ref) if np.isrealobj(ref) else ref
    cross = ref * np.conj(spectrum(lum))
    surface = fft.irfft2(cross / np.maximum(np.abs(cross), 1e-9), s=lum.shape, workers=-1)
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
