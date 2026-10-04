#!/usr/bin/env python3
"""Real-sky finder test: capture, focus-check and plate-solve every few seconds.

Usage (finder on a tripod, pointed at clear sky):
    .venv/bin/python scripts/hwcheck/solve_sky.py --exp 0.5 --gain 100 --save tests/data/sky
Prints one line per frame and a summary (solve rate, time, failure reasons) on Ctrl+C.
"""

import argparse
from pathlib import Path

from astro.devices.svbony import SvbonyCamera
from astro.pointing import sky_test
from astro.pointing.platesolve import FinderSolver

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=float, default=0.5, help="exposure, seconds")
ap.add_argument("--gain", type=int, default=100)
ap.add_argument("--every", type=float, default=3, help="seconds between solves")
ap.add_argument("--save", type=Path, default=None, help="directory for sample raw frames")
ap.add_argument("--save-every", type=int, default=10, help="save every Nth frame")
args = ap.parse_args()

cam = SvbonyCamera("SV905C")
cam.connect()
try:
    cam.set_exposure(args.exp)
    cam.set_gain(args.gain)
    sky_test.run(cam, FinderSolver(), args.every, args.save, args.save_every)
finally:
    cam.close()
