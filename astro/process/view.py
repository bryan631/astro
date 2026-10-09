"""A raw Bayer frame as a small JPEG for the tablet."""
import io

import numpy as np
from PIL import Image

from astro.process.planet import superpixel_rgb

MAX_WIDTH = 960
DAY_MEDIAN = 50  # 8-bit: a typical pixel this bright is no night sky
MIN_RANGE = 40  # 8-bit counts: a dark frame stays dark instead of stretching its noise to full scale


def jpeg(raw: np.ndarray, bayer: str) -> bytes:
    rgb = superpixel_rgb(raw, bayer)
    lo = np.percentile(rgb, 50)  # the sky background is the typical pixel
    if lo > DAY_MEDIAN:  # day mode (a room or daylight): show it as it is, no sky stretch
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        hi = max(np.percentile(rgb, 99.8), lo + MIN_RANGE)
        img = Image.fromarray((((rgb - lo) / (hi - lo)).clip(0, 1) ** 0.6 * 255).astype(np.uint8))
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue()
