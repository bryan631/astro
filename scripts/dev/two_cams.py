"""Hardware check: both SVBony cameras capturing at once, each in its own process.

One process per camera (two capture threads in one process segfaulted the SDK, 2026-10-10).
Each prints its frame rate and errors, and saves frames with their mid-exposure time, so the
pairs can serve as test data. Stop the astro service first (it holds the cameras):
    sudo systemctl stop astro
    .venv/bin/python scripts/dev/two_cams.py 30 data/fixtures/night-2026-10-10
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

CAMERAS = {"finder": ("SV905C", 0.8, 200), "main": ("SV705C", 0.25, 480)}
SAVE_EVERY_S = 2.0


def run(name: str, seconds: float, out: Path) -> None:
    from astro.devices.svbony import SvbonyCamera  # noqa: PLC0415 - the SDK loads in this process only

    model, exposure, gain = CAMERAS[name]
    cam = SvbonyCamera(model)
    cam.connect()
    cam.set_exposure(exposure)
    cam.set_gain(gain)
    frames = errors = 0
    start = saved_at = time.time()
    while time.time() - start < seconds:
        try:
            frame = cam.capture()
        except Exception as e:  # report and keep going: this is what the test is about
            errors += 1
            print(f"{name}: error {e}", flush=True)
            continue
        when = time.time() - exposure / 2  # mid-exposure
        frames += 1
        if when - saved_at >= SAVE_EVERY_S:
            saved_at = when
            np.savez_compressed(out / f"{name}_{when:.3f}.npz", frame=frame, t=when,
                                exposure_s=exposure, gain=gain, bayer=cam.bayer)
    cam.close()
    print(f"{name}: {frames} frames in {seconds:.0f} s ({frames / seconds:.1f}/s), {errors} errors",
          flush=True)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/fixtures/two_cams")
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")  # a fresh interpreter per camera: no SDK state shared
    procs = [ctx.Process(target=run, args=(name, seconds, out)) for name in CAMERAS]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        print(f"process exit code {p.exitcode} (negative = killed by a signal, -11 = segfault)")
