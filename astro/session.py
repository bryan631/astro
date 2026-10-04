"""One observing session: ties planner, safety, mount position and guidance together.

`handle(text)` answers offline intents; `tick()` runs guidance at ~10 Hz. Both return
messages for the tablet: {"type": "say", "text": ...} and {"type": "state", ...}.
"""

import threading
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from astro.capture.focus import FocusCoach, laplacian_variance
from astro.capture.recorder import Recorder
from astro.capture.roi import brightest_blob, roi_around
from astro.devices.base import Camera
from astro.guidance.engine import Guide
from astro.intents import match_name, parse
from astro.planner.catalog import load_targets
from astro.planner.tonight import PLANETS, plan
from astro.pointing.coords import Site, body_altaz, radec_to_altaz
from astro.pointing.finder_sync import FinderSync, check_focus
from astro.pointing.geometry import separation_deg
from astro.safety import check_target

Clock = Callable[[], datetime]
TARGET_REFRESH_S = 1.0  # targets drift ~15"/s, so re-resolve their alt/az once a second
FOCUS_STEP_S = 1.0  # one finder focus measurement per second while coaching
MIN_HFR_PX = 0.5  # floor so a perfectly sharp (tiny) star can't blow up the focus score
TOLERANCE_ARCMIN = {False: 4.0, True: 2.0}  # guidance tolerance without / with the 2x Barlow
RECORD_SECONDS = 60  # planetary video length
FOCUS_CROP_PX = 256  # sharpness measured on a crop around the planet


def utcnow() -> datetime:
    return datetime.now(UTC)


class Session:
    def __init__(self, site: Site, position: Callable[[], tuple[float, float]] | None = None,
                 clock: Clock = utcnow, developer_override: bool = False,
                 finder: FinderSync | None = None, main_camera: Camera | None = None,
                 main_sensor: tuple[int, int] = (3856, 2180), data_dir: Path = Path("data")):
        """Pointing comes from `finder` (encoders + mount model + plate solving), or, for
        tests without a finder, from `position()` returning true (alt, az)."""
        self.site, self.clock, self.finder = site, clock, finder
        self.position = finder.position if finder else position
        self.override = developer_override
        self.catalog = {t.name: t for t in load_targets()}
        for t in list(self.catalog.values()):
            self.catalog.setdefault(t.id, t)
        self.target: str | None = None
        self.guide: Guide | None = None
        self._suggestions: list[str] = []
        self._resolved_at = -1e9
        self._focus_coach: FocusCoach | None = None
        self._focus_mode = ""  # "finder" or "main"
        self.main_camera, self.main_sensor = main_camera, main_sensor
        self.barlow = False
        self.main_focus_ok = False  # pre-flight gate: reset per session and on Barlow change
        self.record_seconds = RECORD_SECONDS
        self.recorder = Recorder(main_camera, main_sensor, data_dir / "captures") if main_camera else None
        self._announced_done = True
        self._focus_at = -1e9
        # Commands and the guidance tick run on worker threads (camera calls block), so
        # serialize them: one camera capture or state change at a time.
        self._lock = threading.Lock()

    # --- target resolution -------------------------------------------------------------
    def names(self) -> list[str]:
        return [p.capitalize() for p in PLANETS] + ["Moon", *self.catalog]

    def altaz_of(self, name: str) -> tuple[float, float]:
        if name.lower() in PLANETS or name == "Moon":
            return body_altaz(name.lower(), self.site, self.clock())
        t = self.catalog[name]
        return radec_to_altaz(t.ra, t.dec, self.site, self.clock())

    # --- commands ------------------------------------------------------------------------
    def handle(self, text: str) -> list[dict]:
        with self._lock:
            return self._handle(text)

    def _handle(self, text: str) -> list[dict]:
        intent = parse(text)
        if intent is None:
            return [say("Sorry, I didn't catch that. Try 'what's good tonight' or 'go to Saturn'.")]
        if intent.name == "goto":
            name = match_name(intent.target or "", self.names())
            return self.goto(name) if name else [say(f"I don't know {intent.target}.")]
        if intent.name == "stop":
            if self._focus_coach is not None:
                if self._focus_mode == "main":
                    self.main_focus_ok = True
                self._focus_coach = None
                return [say("OK, focus is set.")]
            self.target, self.guide = None, None
            return [say("Stopped.")]
        if intent.name in ("barlow_on", "barlow_off"):
            self.barlow = intent.name == "barlow_on"
            self.main_focus_ok = False
            return [say("Got it. The Barlow changes focus, so we'll refocus before taking pictures.")]
        if intent.name == "focus":
            return self.start_main_focus()
        if intent.name == "capture":
            return self.capture()
        if intent.name == "stop_capture":
            if self.recorder is None or self.recorder.current is None:
                return [say("We're not recording.")]
            self.recorder.stop()
            return [say("Stopping the recording.")]
        if intent.name == "sync":
            return self.sync()
        if intent.name == "finder_focus":
            return self.start_finder_focus()
        if intent.name == "tonight":
            return self.tonight()
        if intent.name == "next":
            if not self._suggestions:
                return [say("Ask me what's good tonight first.")]
            return self.goto(self._suggestions.pop(0))
        if intent.name == "where":
            return self.where()
        return [say(f"{intent.name.replace('_', ' ').capitalize()} isn't ready yet.")]

    def sync(self) -> list[dict]:
        if self.finder is None:
            return [say("There's no finder camera connected.")]
        return [say(self.finder.sync()[1])]

    def start_finder_focus(self) -> list[dict]:
        if self.finder is None:
            return [say("There's no finder camera connected.")]
        self.target, self.guide = None, None
        self._focus_coach, self._focus_mode = FocusCoach(), "finder"
        return [say("Point at some stars, then turn the finder's focus ring slowly. "
                    "I'll tell you when it gets sharper. Say stop when I say it's the sharpest.")]

    def start_main_focus(self) -> list[dict]:
        if self.main_camera is None:
            return [say("There's no main camera connected.")]
        self.guide = None  # keep the target; we're on it
        self._focus_coach, self._focus_mode = FocusCoach(), "main"
        return [say("Turn the telescope's focus knob slowly. I'll tell you when it gets sharper. "
                    "Say stop when I say it's the sharpest.")]

    def capture(self) -> list[dict]:
        if self.recorder is None:
            return [say("There's no main camera connected.")]
        if not self.main_focus_ok:  # pre-flight gate (plan Phase 1 step 8)
            return [say("Let's make sure it's sharp first."), *self.start_main_focus()]
        rec = self.recorder.start(self.target or "capture", self.record_seconds)
        if rec is None:
            return [say("I don't see anything bright in the main camera. Let's center it first.")]
        self._announced_done = False
        return [say(f"Recording for {self.record_seconds:g} seconds. Try not to touch the telescope.")]

    def goto(self, name: str) -> list[dict]:
        pre: list[dict] = []
        if self.finder is not None and not self.finder.synced:
            ok, msg = self.finder.sync()  # need to know where we point before guiding
            if not ok:
                return [say(f"Before we go, I need to see the stars. {msg}")]
            pre = [say(msg)]
        alt, az = self.altaz_of(name)
        safe = check_target(alt, az, self.site, self.clock(), self.override)
        if not safe.ok:
            return [say(f"I can't go to {name}: it's {safe.reason}.")]
        guide = Guide(alt, az, tolerance_arcmin=TOLERANCE_ARCMIN[self.barlow])
        self.target, self.guide, self._focus_coach = name, guide, None
        return [*pre, say(f"Let's find {name}.")]

    def tonight(self) -> list[dict]:
        choices = plan(self.site, self.clock())
        flat = sorted((c for cs in choices.values() for c in cs), key=lambda c: -c.score)
        if not flat:
            return [say("Nothing good is up right now.")]
        self._suggestions = [c.name for c in flat[1:6]]
        best = flat[0]
        others = ", ".join(c.name for c in flat[1:3])
        return [say(f"{best.name} is the best right now. {best.note} "
                    f"Other good ones: {others}. Say 'go to' a name, or 'next'.")]

    def tonight_by_category(self) -> str:
        """Compact text for the agent: best target per category."""
        choices = plan(self.site, self.clock())
        lines = [f"{cat}: {cs[0].name} (best around {cs[0].best_time:%H:%M}). {cs[0].note}"
                 for cat, cs in choices.items()]
        return "\n".join(lines) or "Nothing good is up right now."

    def where(self) -> list[dict]:
        if self.finder is not None and not self.finder.synced:
            ok, msg = self.finder.sync()  # encoders mean nothing until the first solve
            if not ok:
                return [say(f"I don't know where we're pointing yet. {msg}")]
        alt, az = self.position()
        nearest = min(self.names(), key=lambda n: separation_deg(alt, az, *self.altaz_of(n)))
        d = separation_deg(alt, az, *self.altaz_of(nearest))
        if d < 1:
            return [say(f"You're on {nearest}.")]
        return [say(f"You're about {d:.0f} degrees from {nearest}.")]

    # --- guidance loop -------------------------------------------------------------------
    def tick(self, t: float) -> list[dict]:
        with self._lock:
            return self._tick(t)

    def _tick(self, t: float) -> list[dict]:
        rec = self.recorder.current if self.recorder else None
        if rec is not None and rec.done.is_set() and not self._announced_done:
            self._announced_done = True
            why = " The planet drifted out of view, so I stopped early." if rec.lost else ""
            return [say(f"Done. I saved {rec.frames} frames.{why}")]
        if self._focus_coach is not None:
            if self._focus_mode == "main":
                return self._main_focus_step(t)
            return self._finder_focus_step(t)
        if self.guide is None or self.target is None:
            return []
        if t - self._resolved_at >= TARGET_REFRESH_S:
            self._resolved_at = t
            alt, az = self.altaz_of(self.target)
            if not check_target(alt, az, self.site, self.clock(), self.override).ok:
                self.target, self.guide = None, None
                return [say("Stopping: the target is no longer safe to point at.")]
            self.guide.target = (alt, az)
        state, cue = self.guide.update(*self.position(), t)
        out = [{"type": "state", "target": self.target, **asdict(state)}]
        if cue:
            out.append(say(cue.text))
        return out


    def _finder_focus_step(self, t: float) -> list[dict]:
        if t - self._focus_at < FOCUS_STEP_S or self.finder is None or self._focus_coach is None:
            return []
        self._focus_at = t
        report = self.finder.focus_report()
        if report.stars == 0:
            return [say("I can't see any stars yet.")]
        # Fewer visible stars also means softer focus, so fold the count into the score.
        cue = self._focus_coach.update(report.stars / max(report.hfr_px, MIN_HFR_PX))
        return [say(cue)] if cue else []


    def _main_focus_step(self, t: float) -> list[dict]:
        if t - self._focus_at < FOCUS_STEP_S or self.main_camera is None or self._focus_coach is None:
            return []
        self._focus_at = t
        frame = self.main_camera.capture()
        if self.target is None or self.target in self._extended_targets():
            center = brightest_blob(frame)
            if center is None:
                return [say("I don't see anything bright in the main camera.")]
            h, w = frame.shape
            r = roi_around(center, FOCUS_CROP_PX, (w, h))
            score = laplacian_variance(frame[r.y:r.y + r.height, r.x:r.x + r.width])
        else:  # stars: smaller is sharper
            report = check_focus(frame.astype(float))
            if report.stars == 0:
                return [say("I don't see any stars in the main camera.")]
            score = 1 / max(report.hfr_px, MIN_HFR_PX)
        cue = self._focus_coach.update(score)
        return [say(cue)] if cue else []

    def _extended_targets(self) -> set[str]:
        return {p.capitalize() for p in PLANETS} | {"Moon"}


def say(text: str) -> dict:
    return {"type": "say", "text": text}
