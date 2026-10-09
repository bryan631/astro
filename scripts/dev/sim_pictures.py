"""Made-up pictures for GUI design: live-stack previews and finished gallery pictures.

Synthetic targets go through the app's real processing (stretch for live previews, finish for
the gallery), so they look the way a real night would. Run, then open the GUI:
    .venv/bin/python scripts/dev/sim_pictures.py
    ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --port 8000
    http://localhost:8000/?demo=m42   (plays a live stack, then the picture lands in Pictures)
Files are named sim_*, so `rm data/gallery/sim_* data/live/sim_*` removes them.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from astro.process import planet
from astro.process.finish import finish
from astro.process.livestack import stretch

ROOT = Path(__file__).resolve().parents[2]
GALLERY, LIVE = ROOT / "data" / "gallery", ROOT / "data" / "live"
H, W = 540, 720  # main camera after 2x2 superpixel debayer, roughly
FRAMES, PREVIEW_EVERY = 60, 4  # 0.2 s subs; a preview every few frames, like every 3 s
rng = np.random.default_rng(1)
yy, xx = np.mgrid[0:H, 0:W].astype(float)


def blob(cx, cy, sx, sy, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def stars(n, bright=1.0, sigma=1.3):
    img = np.zeros((H, W))
    for x, y, f in zip(rng.uniform(0, W, n), rng.uniform(0, H, n), rng.pareto(1.5, n) * 0.05 * bright):
        x0, y0 = int(x), int(y)
        sl = np.s_[max(y0 - 8, 0):y0 + 9, max(x0 - 8, 0):x0 + 9]
        img[sl] += min(f, 3) * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * sigma**2))
    return np.repeat(img[..., None], 3, axis=2) * rng.uniform(0.85, 1.15, 3)


def noise_field(n):  # lumpy structure for nebulae
    f = np.fft.fft2(rng.normal(size=(H, W)))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H)))
    return np.real(np.fft.ifft2(f / (k + 0.01) ** n))


def m42():
    lumps = noise_field(1.4)
    lumps = (lumps - lumps.min()) / np.ptp(lumps)
    neb = blob(360, 270, 140, 110, 0.4) * (0.4 + lumps) + 2 * blob(360, 260, 30, 25)
    color = np.dstack([neb * 1.0, neb * 0.45, neb * 0.55]) + np.dstack([blob(360, 262, 12, 12)] * 3) * 3
    return 0.08 * color + stars(250)


def m31():
    gal = blob(360, 270, 260, 60, -0.6) * 0.5 + blob(360, 270, 60, 25, -0.6) * 2 + blob(360, 270, 8, 6, -0.6) * 4
    dust = 1 - 0.5 * blob(330, 250, 220, 6, -0.6)
    gal *= dust
    return 0.08 * np.dstack([gal * 1.05, gal * 0.95, gal * 0.8]) + 0.06 * np.dstack([blob(160, 120, 25, 18)] * 3) + stars(200)


def m45():
    img = stars(150)
    for x, y in [(300, 200), (380, 240), (420, 320), (330, 330), (260, 290), (470, 210), (350, 270)]:
        img += np.dstack([blob(x, y, 1.6, 1.6)] * 3) * [2.6, 2.8, 3.2]
        img += np.dstack([blob(x, y, 40, 30) * 0.03] * 3) * [0.5, 0.7, 1.2]  # reflection nebula
    return img


def live_stack(name, truth, sky=0.05, read_noise=0.06):
    """Running mean of noisy drifting frames; previews written as the app does, then finished."""
    total = np.zeros_like(truth)
    for n in range(1, FRAMES + 1):
        frame = rng.poisson((truth + sky) * 40) / 40 + rng.normal(0, read_noise, truth.shape)
        frame += np.linspace(0, 0.02, W)[None, :, None]  # light-pollution gradient
        total += frame
        if n % PREVIEW_EVERY == 0:
            Image.fromarray(stretch(total / n)).save(LIVE / f"sim_{name}_{n // PREVIEW_EVERY:02d}.png")
    Image.fromarray(finish(total / FRAMES)).save(GALLERY / f"sim_{name}.png")


def disk(cx, cy, r):
    return np.clip(r - np.hypot(xx - cx, yy - cy), 0, 1)


def jupiter():
    r = 90
    lat = (yy - 270) / r
    bands = 0.75 + 0.2 * np.sin(lat * 9) + 0.05 * np.sin(lat * 23 + xx / 40)
    limb = np.sqrt(np.clip(1 - ((xx - 360) ** 2 + (yy - 270) ** 2) / r**2, 0, 1)) ** 0.4
    b = disk(360, 270, r) * bands * limb
    spot = blob(400, 320, 14, 8)
    img = np.dstack([b * 1.0 + spot * 0.3, b * 0.85, b * 0.65 - spot * 0.2])
    for x in (150, 220, 560):  # moons
        img += np.dstack([disk(x, 272, 3)] * 3) * 0.8
    return img


def saturn():
    r = 60
    rho = np.hypot(xx - 360, (yy - 270) / 0.3)  # ring plane seen at a tilt
    ring = ((rho > 1.25 * r) & (rho < 2.2 * r)) * (0.8 - 0.6 * ((rho > 1.85 * r) & (rho < 1.92 * r)))
    ball = disk(360, 270, r) * (0.8 + 0.1 * np.sin((yy - 270) / r * 6))
    b = np.where(ball > 0, ball, ring)  # (the ring in front would cross the ball; close enough)
    return np.dstack([b * 1.0, b * 0.9, b * 0.7])


def moon():
    r = 240
    craters = 0.15 * noise_field(2.0)
    craters /= np.abs(craters).max()
    lit = (xx - 360) > -60 * np.sqrt(np.clip(1 - ((yy - 270) / r) ** 2, 0, 1))  # gibbous terminator
    b = disk(360, 270, r) * (0.7 + craters) * lit
    return np.dstack([b] * 3)


def planet_picture(name, truth, frames=40):
    """Seeing-blurred, jittered, noisy frames -> the app's planet finish."""
    stack = np.zeros_like(truth)
    for _ in range(frames):
        f = ndimage.shift(ndimage.gaussian_filter(truth, (2.5, 2.5, 0)), (*rng.normal(0, 3, 2), 0), order=1)
        stack += f + rng.normal(0, 0.05, truth.shape)
    Image.fromarray(planet.finish(stack / frames)).save(GALLERY / f"sim_{name}.png")


if __name__ == "__main__":
    GALLERY.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    for name, make in [("m42", m42), ("m31", m31), ("m45", m45)]:
        live_stack(name, make())
        print("live stack + picture:", name)
    for name, make in [("jupiter", jupiter), ("saturn", saturn), ("moon", moon)]:
        planet_picture(name, make())
        print("picture:", name)
