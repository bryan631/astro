"""A raw Bayer frame as a small JPEG for the tablet."""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from astro.capture.focus import laplacian_variance
from astro.process.planet import superpixel_rgb

MAX_WIDTH = 960
DAY_MEDIAN = 50  # 8-bit: a typical pixel this bright is no night sky
MIN_RANGE = 40  # 8-bit counts: a dark frame stays dark instead of stretching its noise to full scale


def jpeg(raw: np.ndarray, bayer: str, zoom: int = 1) -> bytes:
    """Turned 180 degrees: the optics show the world upside down, the page shows it upright.
    `zoom` crops the center; the sharpness of what's shown is printed in the corner."""
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    ch, cw = h // zoom // 2 * 2, w // zoom // 2 * 2
    y, x = (h - ch) // 4 * 2, (w - cw) // 4 * 2  # even offsets keep the Bayer pattern
    rgb = superpixel_rgb(raw[y:y + ch, x:x + cw], bayer)
    sharpness = laplacian_variance(rgb.mean(axis=2))
    lo = np.percentile(rgb, 50)  # the sky background is the typical pixel
    if lo > DAY_MEDIAN:  # day mode (a room or daylight): show it as it is, no sky stretch
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        hi = max(np.percentile(rgb, 99.8), lo + MIN_RANGE)
        img = Image.fromarray((((rgb - lo) / (hi - lo)).clip(0, 1) ** 0.6 * 255).astype(np.uint8))
    img = img.rotate(180)
    width = min(MAX_WIDTH, w // 2)  # zoomed views are scaled up to the same size: same text size
    img = img.resize((width, round(img.height * width / img.width)))
    text = f"sharpness {sharpness:.0f}" + (f"  {zoom}x" if zoom > 1 else "")
    draw, font = ImageDraw.Draw(img), ImageFont.load_default(size=28)
    draw.text((10, 8), text, fill=(255, 60, 60), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue()
