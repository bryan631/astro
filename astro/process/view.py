"""A raw Bayer frame as a small JPEG for the tablet."""
import io

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.bahtinov import focus_number, measure_bahtinov
from astro.capture.focus import MAD_SIGMA, measure_focus, sky_map
from astro.process.planet import superpixel_rgb

MAX_WIDTH = 960
# Display rotation per camera, as mounted. The finder optics flip the view; the main camera was
# turned 180 degrees in its holder (2026-10-10) to undo the same flip.
ROTATE = {"finder": 180, "main": 0}
DAY_MEDIAN = 50  # 8-bit: without the Sun to go by, a typical pixel this bright is no night sky
HOT_SIGMA = 4  # this far above all its neighbors (in noise units) is a hot pixel, not a star
COLOR_SIGMA = 10  # a pixel this far above the sky shows its full color
MIN_STAR_PX = 20  # fewer clear star pixels than this: no color balance
SATURATED_RAW = 250
MIN_RANGE = 40  # 8-bit counts: a dark frame stays dark instead of stretching its noise to full scale
MIN_RANGE_NOISE = 12  # ... and in sky noise: the main camera's few stars stretched its noise gray
COLOR_BLUR_PX = 1.0  # each star colored by all its light: the finder's stars are 1-2 pixels
COLOR_BOOST = 2.0  # stars differ by ~R/B 0.6-1.5 (2026-10-10): doubled, as a finished picture does


def remove_hot_pixels(rgb: np.ndarray) -> np.ndarray:
    """Hot pixels look like stars that never move. A star spreads over several pixels; a hot
    pixel is one bright pixel with dark neighbors, so it gets its brightest neighbor's value."""
    ring = np.ones((3, 3, 1), bool)
    ring[1, 1, 0] = False
    neighbors = ndimage.maximum_filter(rgb, footprint=ring)
    noise = np.std(rgb[::8, ::8]) + 1e-6  # a sample is plenty
    return np.where(rgb - neighbors > HOT_SIGMA * noise, neighbors, rgb)


def render(raw: np.ndarray, bayer: str, zoom: int = 1, rotate: int = 180,
           night: bool | None = None, center: tuple[float, float] = (0.0, 0.0),
           mask: bool = False) -> tuple[bytes, dict]:
    """A raw frame as a JPEG for a camera view, and its numbers: focus (higher is sharper, see
    astro/capture/focus.py measure_focus), stars measured, focus_mode (stars, planet, none, or
    with `mask` looked for, bahtinov: astro/capture/bahtinov.py, and mask_offset).

    Turned by `rotate` degrees (ROTATE: each camera as mounted) so the page shows it upright;
    `zoom` crops around `center` (view fractions from the middle, e.g. the main camera's box on
    the finder), kept inside the frame; metrics["center"] is where the crop really is. At `night` (None: guess from the brightness) the sky's glow is
    taken off and the stars stretched; by day the view is as it is. Big frames are
    binned before the display work (the main camera's full frame cost ~0.4 s per frame)."""
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    ch, cw = h // zoom // 2 * 2, w // zoom // 2 * 2
    sx, sy = (-c for c in center) if rotate == 180 else center  # view -> sensor fractions
    x = int(np.clip(w / 2 + sx * w - cw / 2, 0, w - cw)) // 2 * 2  # even offsets keep the Bayer
    y = int(np.clip(h / 2 + sy * h - ch / 2, 0, h - ch)) // 2 * 2  # pattern
    used = ((x + cw / 2) / w - 0.5, (y + ch / 2) / h - 0.5)
    rgb = remove_hot_pixels(superpixel_rgb(raw[y:y + ch, x:x + cw], bayer).astype(np.float32))
    lum = rgb.mean(axis=2)
    lo = float(np.percentile(lum, 50))  # the sky background is the typical pixel
    metrics = {"center": tuple(-c for c in used) if rotate == 180 else used}
    # The mask first (0.1 s on the Mele): when it's on it decides, and the star fits (~1 s) wait.
    if mask and (found := measure_bahtinov(lum)) is not None:
        metrics |= {"focus": focus_number(found), "stars": 0, "focus_mode": "bahtinov",
                    "mask_offset": round(found.offset_px, 2)}
    else:
        focus = measure_focus(lum)  # also through bright cloud glow (it measures from the sky level)
        metrics |= {"focus": focus.score, "stars": focus.stars, "focus_mode": focus.mode}
    if night is None:
        night = lo <= DAY_MEDIAN
    while rgb.shape[1] > 2 * MAX_WIDTH:  # display needs no more than this; binned first, the
        rgb = _bin2(rgb)  # night stretch took 1.2 s of the Mele's 2.7 s telescope frame (2026-10-12)
    if night:
        rgb = _night_stretch(rgb, rgb.mean(axis=2))
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))  # by day: as it is
    img = img.rotate(rotate)
    width = min(MAX_WIDTH, w // 2)  # zoomed views are scaled up to the same size
    img = img.resize((width, round(img.height * width / img.width)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80, subsampling=0)  # full-resolution color: stars are tiny
    return out.getvalue(), metrics


def _night_stretch(rgb: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """Glow and gradients off the brightness, stretched on the brightness alone, and each star
    colored by its own light (above each color's sky level): stretching each color separately
    clipped every star to white, and the sky's purple cast colored the faint ones (2026-10-10).
    Taking a sky *map* off each color turned clouds green, so the colors use one level each.
    The colors are blurred (not the brightness): per pixel, a star 1-2 pixels wide took the
    color of the Bayer filter it fell on (2026-10-10 frames)."""
    flat = lum - sky_map(lum)
    lo = float(np.percentile(flat[::4, ::4], 50))
    noise = MAD_SIGMA * float(np.median(np.abs(flat[::4, ::4] - lo))) + 1e-6
    hi = max(float(np.percentile(flat, 99.8)), lo + MIN_RANGE, lo + MIN_RANGE_NOISE * noise)
    bright = ((flat - lo) / (hi - lo)).clip(0, 1) ** 0.6
    excess = cv2.GaussianBlur(rgb - np.median(rgb[::4, ::4].reshape(-1, 3), axis=0), (0, 0), COLOR_BLUR_PX)
    sure = ((flat - lo) / (COLOR_SIGMA * noise)).clip(0, 1)[..., None]  # the sky's noise stays gray
    # A clipped star's color is lost (its core reads the sky's color backwards, which the boost
    # made cyan): white, with the pixels its blurred color reaches.
    clipped = cv2.dilate((rgb.max(axis=2) >= SATURATED_RAW).astype(np.uint8), np.ones((5, 5), np.uint8))
    stars = (sure[..., 0] >= 1) & (clipped == 0)  # clear and not clipped
    starlight = excess[stars].sum(axis=0)  # the sensors see more green: the average star is
    if stars.sum() >= MIN_STAR_PX and (starlight > 0).all():  # made white, as in a finished picture
        excess = excess * (starlight.mean() / starlight)
    # Stars run from red to blue with their temperature; green or magenta is the finder lens's
    # halo and the Bayer filter a tiny star fell on, so only the red-blue difference is kept.
    ratio = excess / np.maximum(excess.mean(axis=2), 1e-6)[..., None]
    warm = COLOR_BOOST * (ratio[..., 0] - ratio[..., 2]) / 2
    color = (1 + warm[..., None] * np.array([1.0, 0.0, -1.0])).clip(0, 3)
    out = bright[..., None] * (1 + sure * (1 - clipped[..., None]) * (color - 1))
    return out / np.maximum(out.max(axis=2), 1)[..., None] * 255  # a bright star keeps its hue


def _bin2(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
    return rgb[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
