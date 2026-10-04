import os

import numpy as np
import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.devices.sim.sky import render
from astro.pointing import sky_test
from astro.pointing.platesolve import FinderSolver


class Frames:
    """Camera stub: returns queued frames, then interrupts like Ctrl+C."""

    def __init__(self, frames):
        self.frames = list(frames)

    def capture(self):
        if not self.frames:
            raise KeyboardInterrupt
        return self.frames.pop(0)


def test_run_tallies_and_saves_every_nth(tmp_path):
    solver = FinderSolver()
    sky = render(solver._t3.star_table, 279.2, 38.8, 0, 11.0, size=(1280, 960), sigma_px=2.0,
                 mag_limit=6.5)  # same as SimFinderCamera
    dark = np.full((960, 1280), 20, np.uint8)
    lines = []
    stats = sky_test.run(Frames([sky, dark, sky, sky]), solver, 0, tmp_path, 2, lines.append)
    assert stats.frames == 4 and stats.outcomes[sky_test.SOLVED] == 3
    assert len(list(tmp_path.glob("*.npy"))) == 2  # frames 2 and 4
    assert "solved 3 (75%)" in lines[-1] and "can't see any stars" in lines[-1]
