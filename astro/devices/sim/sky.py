"""Render a synthetic star field from tetra3's own star table (simulators and tests)."""

import numpy as np


def render(star_table, ra_deg, dec_deg, roll_deg, fov_deg, size=(640, 480), seed=0,
           mag_limit=7.5, sigma_px=1.2):
    w, h = size
    ra0, dec0, roll = np.radians([ra_deg, dec_deg, -roll_deg])  # tetra3 roll convention
    b = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.cross(b, east)
    v = star_table[:, 2:5]
    mag = star_table[:, 5]
    z = v @ b
    keep = (z > 0.5) & (mag < mag_limit)
    x_e, y_n = (v[keep] @ east) / z[keep], (v[keep] @ north) / z[keep]
    # Roll rotates the field; image x grows to the west (east left), y grows downward.
    xr = x_e * np.cos(roll) - y_n * np.sin(roll)
    yr = x_e * np.sin(roll) + y_n * np.cos(roll)
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    px, py = w / 2 - xr * f, h / 2 - yr * f
    rng = np.random.default_rng(seed)
    img = rng.normal(20, 3, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y, m in zip(px, py, mag[keep], strict=True):
        if -5 < x < w + 5 and -5 < y < h + 5:
            flux = 3000 * 10 ** (-0.4 * (m - 3))
            x0, x1, y0, y1 = int(max(x - 6, 0)), int(min(x + 7, w)), int(max(y - 6, 0)), int(min(y + 7, h))
            g = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * sigma_px**2))
            img[y0:y1, x0:x1] += flux * g / (2 * np.pi * sigma_px**2)
    return np.clip(img, 0, 255).astype(np.uint8)
