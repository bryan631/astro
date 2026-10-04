#!/usr/bin/env python3
"""Real-sky finder test: capture, focus-check and plate-solve every few seconds.

Usage (finder on a tripod, pointed at clear sky):
    .venv/bin/python scripts/hwcheck/solve_sky.py --exp 0.5 --gain 100 --save tests/data/sky
Prints one line per frame and a summary (solve rate, time, failure reasons) on Ctrl+C.
Saves every Nth frame as .npy (raw Bayer) for regression tests.
"""

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from astro.devices.svbony import SvbonyCamera
from astro.pointing.finder_sync import check_focus
from astro.pointing.platesolve import FinderSolver, bin2x2

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=float, default=0.5, help="exposure, seconds")
ap.add_argument("--gain", type=int, default=100)
ap.add_argument("--every", type=float, default=3, help="seconds between solves")
ap.add_argument("--save", type=Path, default=None, help="directory for sample frames")
ap.add_argument("--save-every", type=int, default=10)
args = ap.parse_args()

cam, solver = SvbonyCamera("SV905C"), FinderSolver()
cam.connect()
cam.set_exposure(args.exp)
cam.set_gain(args.gain)
if args.save:
    args.save.mkdir(parents=True, exist_ok=True)
stats, n = collections.Counter(), 0
times = []
try:
    while True:
        raw = cam.capture()
        n += 1
        focus = check_focus(bin2x2(raw))
        sol = solver.solve(raw)
        if sol:
            stats["solved"] += 1
            times.append(sol.ms)
            print(f"{n}: RA {sol.ra_deg:.3f} Dec {sol.dec_deg:+.3f} roll {sol.roll_deg:.1f} "
                  f"fov {sol.fov_deg:.2f} matches {sol.matches} {sol.ms:.0f} ms | "
                  f"stars {focus.stars} hfr {focus.hfr_px:.2f}", flush=True)
        else:
            stats[focus.reason or "no pattern match"] += 1
            print(f"{n}: no solve | stars {focus.stars} hfr {focus.hfr_px:.2f} | {focus.reason}",
                  flush=True)
        if args.save and n % args.save_every == 1:
            np.save(args.save / f"finder_{int(time.time())}.npy", raw)
        time.sleep(args.every)
except KeyboardInterrupt:
    pass
finally:
    cam.close()
    print(f"\n{n} frames, solved {stats['solved']} ({100 * stats['solved'] / max(n, 1):.0f}%), "
          f"median {np.median(times) if times else float('nan'):.0f} ms")
    for reason, k in stats.most_common():
        if reason != "solved":
            print(f"  {k}x {reason}")
