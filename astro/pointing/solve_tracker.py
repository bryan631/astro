"""Pointing from the finder alone: plate-solve continuously, no encoders.

A drop-in for FinderSync (same position/sync/focus_report/reset), for the Phase 2 tripod
test before the Arduino arrives, and as a fallback if the encoders fail. A solve takes about
one finder exposure (~0.8 s), so guidance cues update about once a second instead of 10x.
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime

from astro.devices.base import Camera
from astro.pointing.coords import Site, radec_to_altaz
from astro.pointing.finder_sync import FocusReport, check_focus
from astro.pointing.platesolve import FinderSolver, finder_gray

RETRY_S = 1.0  # pause after a failed solve (clouds, slewing) before trying again
FRESH_S = 5.0  # a solve this recent counts for "sync" without solving again
SAFETY_IDLE_S = 10.0  # re-check this often while exposures aren't allowed


class SolveTracker:
    def __init__(self, camera: Camera, solver: FinderSolver, site: Site,
                 clock: Callable[[], datetime]):
        self.camera, self.solver, self.site, self.clock = camera, solver, site, clock
        self.synced = False
        self.safety: Callable[[], str | None] | None = None  # exposure gate, set by the session
        self._altaz = (0.0, 0.0)
        self._solved_at = -1e9
        self._last_reason = "I haven't looked at the sky yet."
        self._camera_lock = threading.Lock()  # one capture at a time (tracker vs focus coach)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "SolveTracker":
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop and wait for the worker, so nothing captures or solves after this returns."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def alignment(self) -> tuple[int, float | None]:
        """Plate solving alone needs no model: aligned once it has solved."""
        return (1 if self.synced else 0), None

    def position(self) -> tuple[float, float]:
        """Last solved (alt, az). The scope sits still between pushes, so this stays valid."""
        return self._altaz

    def sync(self) -> tuple[bool, str]:
        if self.synced and time.monotonic() - self._solved_at < FRESH_S:
            return True, "Got it, I know where we're pointing."
        if self.safety and (reason := self.safety()):  # every finder exposure is gated (S2/S3)
            return False, f"I can't look at the sky right now: {reason}."
        ok, reason = self._solve_once()
        return (True, "Got it, I know where we're pointing.") if ok else (False, reason)

    def focus_report(self) -> FocusReport:
        with self._camera_lock:
            return check_focus(finder_gray(self.camera.capture()))

    def reset(self, site: Site) -> None:
        self.site, self.synced = site, False

    def _solve_once(self) -> tuple[bool, str]:
        with self._camera_lock:
            raw = self.camera.capture()
        focus = check_focus(finder_gray(raw))
        if not focus.ok:
            self._last_reason = focus.reason
            return False, focus.reason
        sol = self.solver.solve(raw)
        if sol is None:
            self._last_reason = ("I can see stars but couldn't recognize the pattern. "
                                 "Something may be blocking part of the view.")
            return False, self._last_reason
        self._altaz = radec_to_altaz(sol.ra_deg, sol.dec_deg, self.site, self.clock())
        self._solved_at, self.synced = time.monotonic(), True
        return True, ""

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.safety and self.safety():  # e.g. daytime: don't expose, just wait
                self._stop.wait(SAFETY_IDLE_S)
                continue
            try:
                ok, _ = self._solve_once()
            except (RuntimeError, OSError) as e:  # camera hiccup: keep going
                ok, self._last_reason = False, f"The finder camera isn't responding: {e}"
            if not ok:
                self._stop.wait(RETRY_S)
