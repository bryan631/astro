"""A raw Bayer frame as a small JPEG for the tablet."""
import io

import numpy as np
from PIL import Image

from astro.process.livestack import stretch
from astro.process.planet import superpixel_rgb

MAX_WIDTH = 960


def jpeg(raw: np.ndarray, bayer: str) -> bytes:
    rgb = stretch(superpixel_rgb(raw, bayer), black_pct=5, gamma=0.6)
    img = Image.fromarray(rgb)
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue()
