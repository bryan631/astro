#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Real-sky finder test: capture, focus-check and plate-solve every few seconds.

Usage (finder on a tripod, pointed at clear sky):
    scripts/hwcheck/solve_sky.py --save tests/data/sky        (defaults: 0.8 s, gain 200)
Prints one line per frame and a summary (solve rate, time, failure reasons) on Ctrl+C.
"""

import argparse
from pathlib import Path

from astro.devices.svbony import SvbonyCamera
from astro.pointing import sky_test
from astro.pointing.platesolve import FinderSolver

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=float, default=0.8, help="exposure, seconds (sweep: >=0.8 s solves)")
ap.add_argument("--gain", type=int, default=200)
ap.add_argument("--every", type=float, default=3, help="seconds between solves")
ap.add_argument("--save", type=Path, default=None, help="directory for sample raw frames")
ap.add_argument("--save-every", type=int, default=10, help="save every Nth frame")
args = ap.parse_args()
if args.exp > 10:
    ap.error(f"--exp is in seconds; {args.exp:g} s is very long. Did you mean {args.exp / 1000:g}?")

cam = SvbonyCamera("SV905C")
cam.connect()
try:
    cam.set_exposure(args.exp)
    cam.set_gain(args.gain)
    sky_test.run(cam, FinderSolver(), args.every, args.save, args.save_every)
finally:
    cam.close()
