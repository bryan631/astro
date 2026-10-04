"""Real-sky finder test loop: capture -> focus check -> solve, with a running tally.

Used by scripts/hwcheck/solve_sky.py; kept here so the CLI is just argument parsing.
"""

import collections
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from astro.devices.base import Camera
from astro.pointing.finder_sync import check_focus
from astro.pointing.platesolve import FinderSolver, finder_gray

SOLVED = "solved"
NO_MATCH = "stars visible but no pattern match"
OUTCOMES = ("solves", "no_match", "no_stars", "not_sky", "few_stars", "out_of_focus")


def classify(raw: np.ndarray, solver: FinderSolver) -> str:
    """One finder frame -> outcome code: solves, no_match, or the focus check's outcome.

    The solver gets every frame, even ones the focus check rejects, so "solves" wins whenever
    the stars are good enough for it."""
    focus = check_focus(finder_gray(raw))
    if solver.solve(raw) is not None:
        return "solves"
    return "no_match" if focus.ok else focus.outcome


@dataclass
class SkyTestStats:
    frames: int = 0
    outcomes: collections.Counter = field(default_factory=collections.Counter)
    solve_ms: list[float] = field(default_factory=list)

    def summary(self) -> str:
        solved = self.outcomes[SOLVED]
        median = f"{np.median(self.solve_ms):.0f} ms" if self.solve_ms else "n/a"
        rate = 100 * solved / max(self.frames, 1)
        lines = [f"{self.frames} frames, solved {solved} ({rate:.0f}%), median solve {median}"]
        lines += [f"  {k}x {reason}" for reason, k in self.outcomes.most_common() if reason != SOLVED]
        return "\n".join(lines)


def test_frame(raw: np.ndarray, solver: FinderSolver, stats: SkyTestStats) -> str:
    """Focus-check and solve one raw finder frame; update `stats`; return a report line."""
    stats.frames += 1
    focus = check_focus(finder_gray(raw))
    seen = f"stars {focus.stars} hfr {focus.hfr_px:.2f}"
    sol = solver.solve(raw)
    if sol is None:
        stats.outcomes[focus.reason or NO_MATCH] += 1
        return f"{stats.frames}: no solve | {seen} | {focus.reason or NO_MATCH}"
    stats.outcomes[SOLVED] += 1
    stats.solve_ms.append(sol.ms)
    return (f"{stats.frames}: RA {sol.ra_deg:.3f} Dec {sol.dec_deg:+.3f} "
            f"rotation {sol.roll_deg:.1f} deg scale {sol.scale_arcsec_px:.2f}\"/px "
            f"fov {sol.fov_deg:.2f} matches {sol.matches} conf {sol.confidence:.1f} "
            f"{sol.ms:.0f} ms | {seen}")


def run(camera: Camera, solver: FinderSolver, every_s: float, save_dir: Path | None = None,
        save_every: int = 10, out: Callable[[str], None] = print) -> SkyTestStats:
    """Loop until Ctrl+C; saves every `save_every`-th raw frame to `save_dir` as .npy."""
    stats = SkyTestStats()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            raw = camera.capture()
            out(test_frame(raw, solver, stats))
            if save_dir and stats.frames % save_every == 0:
                np.save(save_dir / f"finder_{int(time.time())}_{stats.frames:04d}.npy", raw)
            time.sleep(every_s)
    except KeyboardInterrupt:
        pass
    out(stats.summary())
    return stats
