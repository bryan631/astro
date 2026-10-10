"""A raw Bayer frame as a small JPEG for the tablet."""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

from astro.capture.focus import measure_focus
from astro.process.planet import superpixel_rgb

MAX_WIDTH = 960
# Display rotation per camera, as mounted. The finder optics flip the view; the main camera was
# turned 180 degrees in its holder (2026-10-10) to undo the same flip.
ROTATE = {"finder": 180, "main": 0}
DAY_MEDIAN = 50  # 8-bit: a typical pixel this bright is no night sky
HOT_SIGMA = 4  # this far above all its neighbors (in noise units) is a hot pixel, not a star
MIN_RANGE = 40  # 8-bit counts: a dark frame stays dark instead of stretching its noise to full scale


def remove_hot_pixels(rgb: np.ndarray) -> np.ndarray:
    """Hot pixels look like stars that never move. A star spreads over several pixels; a hot
    pixel is one bright pixel with dark neighbors, so it gets its brightest neighbor's value."""
    ring = np.ones((3, 3, 1), bool)
    ring[1, 1, 0] = False
    neighbors = ndimage.maximum_filter(rgb, footprint=ring)
    noise = np.std(rgb[::8, ::8]) + 1e-6  # a sample is plenty
    return np.where(rgb - neighbors > HOT_SIGMA * noise, neighbors, rgb)


def render(raw: np.ndarray, bayer: str, zoom: int = 1, rotate: int = 180) -> tuple[bytes, dict]:
    """A raw frame as a JPEG for a camera view, and its numbers: focus (higher is sharper, see
    astro/capture/focus.py measure_focus), stars measured, focus_mode (stars, planet, none).

    Turned by `rotate` degrees (ROTATE: each camera as mounted) so the page shows it upright;
    `zoom` crops the center. The page draws its own labels over the picture. Big frames are
    binned before the display work (the main camera's full frame cost ~0.4 s per frame)."""
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    ch, cw = h // zoom // 2 * 2, w // zoom // 2 * 2
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2  # even offsets keep the Bayer pattern
    rgb = remove_hot_pixels(superpixel_rgb(raw[y:y + ch, x:x + cw], bayer).astype(np.float32))
    lum = rgb.mean(axis=2)
    lo = float(np.percentile(lum, 50))  # the sky background is the typical pixel
    focus = measure_focus(lum)  # also through bright cloud glow (it measures from the sky level)
    metrics = {"focus": focus.score, "stars": focus.stars, "focus_mode": focus.mode}
    while rgb.shape[1] > 2 * MAX_WIDTH:  # display needs no more than this
        rgb = _bin2(rgb)
    if lo > DAY_MEDIAN:  # day mode (a room or daylight): show it as it is, no sky stretch
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        hi = max(float(np.percentile(rgb, 99.8)), lo + MIN_RANGE)
        img = Image.fromarray((((rgb - lo) / (hi - lo)).clip(0, 1) ** 0.6 * 255).astype(np.uint8))
    img = img.rotate(rotate)
    width = min(MAX_WIDTH, w // 2)  # zoomed views are scaled up to the same size
    img = img.resize((width, round(img.height * width / img.width)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue(), metrics


def jpeg(raw: np.ndarray, bayer: str, zoom: int = 1, rotate: int = 180) -> bytes:
    """The JPEG alone (see render)."""
    return render(raw, bayer, zoom, rotate)[0]


def _bin2(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
    return rgb[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
