#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Live camera view in a matplotlib window.

    scripts/hwcheck/viewer_plot.py finder --exp 0.5 --gain 100
Keys: up/down exposure x2 / /2, right/left gain +/-20, s save raw frame (.npy), q quit.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from live import LiveCamera, parse_args
from matplotlib.animation import FuncAnimation

REFRESH_MS = 50

args = parse_args(__doc__)
live = LiveCamera(args).start()
while live.latest is None:
    time.sleep(0.05)
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_axis_off()
fig.tight_layout()
img = ax.imshow(live.latest[..., ::-1])  # BGR -> RGB


def update(_):
    img.set_data(live.latest[..., ::-1])
    return (img,)


def on_key(event):
    cam = live.cam  # read only here; changes go through live.request (capture thread applies)
    if event.key == "up":
        live.request(exposure_s=cam.exposure_s * 2)
    elif event.key == "down":
        live.request(exposure_s=cam.exposure_s / 2)
    elif event.key == "right":
        live.request(gain=cam.gain + 20)
    elif event.key == "left":
        live.request(gain=max(cam.gain - 20, 0))
    elif event.key == "s" and live.raw is not None:
        name = f"{args.cam}_{int(time.time())}.npy"
        np.save(name, live.raw)
        print("saved", name)


fig.canvas.mpl_connect("key_press_event", on_key)
# No blitting: the image fills the figure anyway, and blit crashes when Tk swaps the canvas.
anim = FuncAnimation(fig, update, interval=REFRESH_MS, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    live.stop()
