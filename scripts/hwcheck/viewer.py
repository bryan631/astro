#!/usr/bin/env python3
"""Live camera window for focusing and hardware checks (matplotlib).

    .venv/bin/python scripts/hwcheck/viewer.py finder --exp 0.5 --gain 100
Keys: up/down exposure x2 / /2, right/left gain +/-20, s save raw frame (.npy), q quit.
Title shows fps, max pixel, sharpness (higher = sharper) and star count / HFR (lower = sharper).
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from astro.capture.preview import focus_numbers, superpixel_rgb
from astro.devices.svbony import SvbonyCamera

MODELS = {"finder": "SV905C", "main": "SV705C"}

ap = argparse.ArgumentParser()
ap.add_argument("cam", choices=MODELS, help="finder (SV905C) or main (SV705C)")
ap.add_argument("--exp", type=float, default=0.05, help="exposure, seconds")
ap.add_argument("--gain", type=int, default=100)
args = ap.parse_args()
if args.exp > 10:
    ap.error(f"--exp is in seconds; {args.exp:g} s is very long. Did you mean {args.exp / 1000:g}?")

cam = SvbonyCamera(MODELS[args.cam])
cam.connect()
cam.set_exposure(args.exp)
cam.set_gain(args.gain)
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_axis_off()
img = ax.imshow(superpixel_rgb(cam.capture(), cam.bayer))
state = {"t": time.time(), "fps": 0.0, "raw": None}


def update(_):
    raw = cam.capture()
    state["raw"] = raw
    now = time.time()
    state["fps"] = 0.8 * state["fps"] + 0.2 / max(now - state["t"], 1e-6)
    state["t"] = now
    img.set_data(superpixel_rgb(raw, cam.bayer))
    f = focus_numbers(raw)
    ax.set_title(f"{args.cam}  exp {cam.exposure_s:g}s  gain {cam.gain}  {state['fps']:.1f} fps  "
                 f"max {raw.max()}  sharpness {f.sharpness:.0f}  stars {f.stars}  hfr {f.hfr_px:.2f}")
    return (img,)


def on_key(event):
    if event.key == "up":
        cam.set_exposure(cam.exposure_s * 2)
    elif event.key == "down":
        cam.set_exposure(cam.exposure_s / 2)
    elif event.key == "right":
        cam.set_gain(cam.gain + 20)
    elif event.key == "left":
        cam.set_gain(max(cam.gain - 20, 0))
    elif event.key == "s" and state["raw"] is not None:
        name = f"{args.cam}_{int(time.time())}.npy"
        np.save(name, state["raw"])
        print("saved", name)


fig.canvas.mpl_connect("key_press_event", on_key)
anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    cam.close()
