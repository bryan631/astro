"""Cut a two_cams.py recording down to a CI fixture for the Align step.

Keeps a few full finder frames (they must plate-solve) and, for every main frame, a crop around
the brightest object with its offset, all with mid-exposure times:
    .venv/bin/python scripts/dev/pack_align_fixture.py data/fixtures/align-2026-10-10 \
        tests/data/align/bright-star-2026-10-10.npz
"""

import json
import sys
from pathlib import Path

import numpy as np

from astro.capture.roi import brightest_blob

CROP = 128  # px each way from the object: room for a planet and seeing jitter
FINDER_FRAMES = 3


def main(src: Path, dest: Path) -> None:
    finder = sorted(src.glob("finder_*.npz"))
    pick = [finder[i] for i in np.linspace(0, len(finder) - 1, FINDER_FRAMES).astype(int)]
    out = {"finder": np.stack([np.load(f)["frame"] for f in pick]),
           "finder_t": np.array([float(np.load(f)["t"]) for f in pick])}
    crops, origins, times = [], [], []
    for f in sorted(src.glob("main_*.npz")):
        d = np.load(f)
        blob = brightest_blob(d["frame"])
        if blob is None:
            continue
        h, w = d["frame"].shape
        x0 = int(np.clip(blob[0] - CROP, 0, w - 2 * CROP)) // 2 * 2  # even: keeps the Bayer phase
        y0 = int(np.clip(blob[1] - CROP, 0, h - 2 * CROP)) // 2 * 2
        crops.append(d["frame"][y0:y0 + 2 * CROP, x0:x0 + 2 * CROP])
        origins.append((x0, y0))
        times.append(float(d["t"]))
        out["main_shape"], out["main_bayer"] = np.array([h, w]), str(d["bayer"])
    out |= {"main": np.stack(crops), "main_origin": np.array(origins), "main_t": np.array(times),
            "finder_bayer": str(np.load(pick[0])["bayer"])}
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(json.dumps({"finder": len(pick), "main": len(crops), "bytes": dest.stat().st_size}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
