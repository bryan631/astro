#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Record labeled real finder frames as regression cases (tests/test_real_frames.py).

    scripts/hwcheck/capture_cases.py lens_cap --expect no_stars --notes "cap on"
    scripts/hwcheck/capture_cases.py trees_left --expect few_stars -n 2
    scripts/hwcheck/capture_cases.py zenith --expect solves --notes "straight up"

Expect one of: solves, no_match, no_stars, not_sky, few_stars, out_of_focus.
Each frame is classified right away, so you see whether reality matched the label.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from astro.devices.svbony import SvbonyCamera
from astro.pointing.platesolve import FinderSolver
from astro.pointing.sky_test import OUTCOMES, classify

FINDER_DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "finder"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("label", help="short name, e.g. lens_cap, trees_left, defocus_in, zenith")
ap.add_argument("--expect", required=True, choices=OUTCOMES)
ap.add_argument("--exp", type=float, default=0.8, help="exposure, seconds")
ap.add_argument("--gain", type=int, default=200)
ap.add_argument("-n", type=int, default=1, help="frames to record")
ap.add_argument("--notes", default="")
args = ap.parse_args()
if args.exp > 10:
    ap.error(f"--exp is in seconds; {args.exp:g} s is very long. Did you mean {args.exp / 1000:g}?")

manifest_path = FINDER_DATA / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
solver, cam = FinderSolver(), SvbonyCamera("SV905C")
cam.connect()
try:
    cam.set_exposure(args.exp)
    cam.set_gain(args.gain)
    for i in range(args.n):
        raw = cam.capture()
        got = classify(raw, solver)
        name = f"{args.label}_{i}" if args.n > 1 else args.label
        if (FINDER_DATA / f"{name}.npz").exists():
            name += datetime.now().astimezone().strftime("_%H%M%S")
        np.savez_compressed(FINDER_DATA / f"{name}.npz", raw=raw)
        manifest.append({"file": f"{name}.npz", "expect": args.expect, "exposure_s": args.exp,
                         "gain": args.gain, "taken": datetime.now().astimezone().isoformat(timespec="minutes"),
                         "notes": args.notes})
        mark = "ok" if got == args.expect else f"MISMATCH (expected {args.expect})"
        print(f"{name}: {got}  {mark}", flush=True)
finally:
    cam.close()
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
print("Saved to tests/data/finder. A MISMATCH is still a useful case: tell Claude about it.")
