"""Finder -> main offset from a planet drifting through both cameras (scope held still).

Each camera's frames are time-stamped (mid-exposure) and the planet's centroid measured; the
planet's own alt/az at each moment is known, so per camera a similarity fit (known handedness:
neither camera mirrors the sky) gives its rotation, scale and where it points. No plate solve,
no encoders. Run on the Mele while the server runs:
    .venv/bin/python scripts/dev/drift_offset.py saturn [--save]
"""

import asyncio
import io
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import astropy.units as u
import numpy as np
import websockets
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time
from PIL import Image
from scipy import ndimage

URL = "https://localhost:8443"
CTX = ssl.create_default_context()
CTX.check_hostname, CTX.verify_mode = False, ssl.CERT_NONE
ROUNDS, SLICE_S = 4, 6.0  # main/finder turns; the planet crosses the main camera in ~2 min
ARCSEC_PX = {"main": 1928 * 0.5 / 960, "finder": 10.38 * 3600 / 640}  # page JPEG pixels
TEXT_BOX = (slice(0, 50), slice(0, 300))  # the sharpness label


def get(path):
    with urllib.request.urlopen(URL + path, context=CTX, timeout=5) as r:
        return r.read(), r.headers


def frames(camera, seconds):
    """[(utc unix time at mid-exposure, x, y)] of the brightest blob, one per new frame."""
    debug = json.loads(get("/api/debug")[0])
    exposure = debug[f"{camera}_camera"]["exposure_s"]
    out, last, end = [], None, time.time() + seconds
    while time.time() < end:
        data, headers = get(f"/api/camera/{camera}.jpg")
        when = time.time() - float(headers["X-Frame-Age"]) - exposure / 2
        if last is None or abs(when - last) > 0.05:
            last = when
            g = np.asarray(Image.open(io.BytesIO(data)).convert("L")).astype(float)
            g[TEXT_BOX] = np.median(g)
            s = ndimage.gaussian_filter(g, 2)
            if s.max() - np.median(s) > 40:
                region = s > (s.max() + np.median(s)) / 2
                lab, _ = ndimage.label(region)
                y, x = ndimage.center_of_mass(s * (lab == lab[np.unravel_index(s.argmax(), s.shape)]))
                out.append((when, x - g.shape[1] / 2, y - g.shape[0] / 2))
        time.sleep(0.2)
    return out


async def send(*messages):
    async with websockets.connect(URL.replace("https", "wss") + "/ws", ssl=CTX) as ws:
        for m in messages:
            await ws.send(json.dumps(m))
        await asyncio.sleep(0.5)


async def capture():
    """Main and finder interleaved (the SVBony SDK crashes capturing both at once)."""
    got = {"main": [], "finder": []}
    for _ in range(ROUNDS):
        for camera in ("main", "finder"):
            await send({"type": "video", "camera": camera})
            await asyncio.sleep(2)  # the camera reopens with its video settings
            got[camera] += await asyncio.to_thread(frames, camera, SLICE_S)
    print("frames with the planet: main", len(got["main"]), "finder", len(got["finder"]), flush=True)
    return got


def fit(samples, planet, loc, ref):
    """Similarity fit p = s R D (q - aim): q the planet's sky offset from `ref` (deg)."""
    t = np.array([s[0] for s in samples])
    p = np.array([s[1:] for s in samples])
    aa = get_body(planet, Time(t, format="unix"), loc).transform_to(AltAz(obstime=Time(t, format="unix"), location=loc))
    q = np.c_[((aa.az.deg - ref[1] + 180) % 360 - 180) * np.cos(np.radians(ref[0])), aa.alt.deg - ref[0]]
    q = q * [1, -1]  # D: alt up is image y down; sky as seen (az to the right), not mirrored
    # p = [[a,-b],[b,a]] q + c  ->  linear in a, b, cx, cy
    rows = np.zeros((2 * len(q), 4))
    rows[0::2] = np.c_[q[:, 0], -q[:, 1], np.ones(len(q)), np.zeros(len(q))]
    rows[1::2] = np.c_[q[:, 1], q[:, 0], np.zeros(len(q)), np.ones(len(q))]
    (a, b, cx, cy), *_ = np.linalg.lstsq(rows, p.ravel(), rcond=None)
    m = np.array([[a, -b], [b, a]])
    aim = -np.linalg.solve(m, [cx, cy]) * [1, -1]  # back to (d_az_sky, d_alt)
    resid = np.sqrt(np.mean((rows @ [a, b, cx, cy] - p.ravel()) ** 2))
    return aim, 3600 / np.hypot(a, b), np.degrees(np.arctan2(b, a)), resid, np.ptp(p, axis=0)


def main():
    planet = sys.argv[1] if len(sys.argv) > 1 else "saturn"
    site = json.loads(get("/api/debug")[0])["site"]
    loc = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg, height=site["elevation_m"] * u.m)
    got = asyncio.run(capture())
    t0 = got["main"][0][0] if got["main"] else time.time()
    r = get_body(planet, Time(t0, format="unix"), loc).transform_to(AltAz(obstime=Time(t0, format="unix"), location=loc))
    ref = (r.alt.deg, r.az.deg)
    aims = {}
    for camera, samples in got.items():
        if len(samples) < 5:
            print(f"{camera}: too few frames with the planet ({len(samples)})")
            return
        aim, scale, rot, resid, spread = fit(samples, planet, loc, ref)
        aims[camera] = aim
        print(f"{camera}: aim {aim * 60} arcmin, scale {scale:.2f}\"/px (expect {ARCSEC_PX[camera]:.2f}), "
              f"rotation {rot:.1f} deg, residual {resid:.1f} px, drift {spread} px")
    off = aims["main"] - aims["finder"]
    print(f"OFFSET main - finder: d_az_sky {off[0]:+.3f} deg, d_alt {off[1]:+.3f} deg ({np.hypot(*off):.3f} deg)")
    if "--save" in sys.argv:
        path = Path("data/calibration.json")
        d = json.loads(path.read_text())
        d["main_offset"] = {"d_az_sky_deg": round(float(off[0]), 4), "d_alt_deg": round(float(off[1]), 4),
                            "observations": 1}
        path.write_text(json.dumps(d, indent=1))
        print("saved to", path, "(restart the server to use it)")


if __name__ == "__main__":
    main()
