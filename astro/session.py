"""One observing session: ties planner, safety, mount position and guidance together.

`handle(text)` answers offline intents; `tick()` runs guidance at ~10 Hz. Both return
messages for the tablet: {"type": "say", "text": ...} and {"type": "state", ...}.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from astro import calibration_store
from astro.capture.collimation import CollimationCoach
from astro.capture.exposure import SATURATED_RAW, ViewSettings
from astro.capture.live_stacker import LiveStacker
from astro.capture.recorder import CaptureRefused, Recorder, prune
from astro.capture.roi import brightest_blob, has_disk
from astro.devices.base import Camera
from astro.devices.tap import frame_with_time, tap
from astro.guidance.centering import CALIBRATED, Centerer
from astro.guidance.engine import CueLimiter, DirectionLearner, Guide, wrap180
from astro.intents import Intent, match_name, parse
from astro.messages import notice, say
from astro.optics import MAIN_SENSOR_PX
from astro.planner.horizon import HorizonMask
from astro.planner.sky import MOON_FEATURES, Sky, by_category_text, tonight_text
from astro.planner.tonight import PLANETS
from astro.pointing.align import (
    ALIGN_S,
    AlignRun,
    MainInFinder,
    box_on_finder_view,
    finder_offset_to_sky,
    sky_offset_to_altaz,
)
from astro.pointing.coords import Site, altaz_to_radec, body_altaz, radec_to_altaz
from astro.pointing.finder_sync import FinderSync
from astro.pointing.geometry import separation_deg
from astro.pointing.labels import FinderLabels
from astro.pointing.main_offset import MainOffset
from astro.process.planet import MIN_FRAMES, StackResult, process_ser
from astro.process.view import ROTATE
from astro.safety import DAYTIME_SUN_ALT_DEG, SafetyResult, check_target
from astro.spots import Spot, upsert, with_mask
from astro.wizard import SetupWizard

Clock = Callable[[], datetime]
log = logging.getLogger(__name__)

TARGET_REFRESH_S = 1.0  # targets drift ~15"/s, so re-resolve their alt/az once a second
ZOOMS = (1, 2, 4)  # the page's digital zoom per camera view
NOT_UNDERSTOOD = "Sorry, I didn't catch that. Say 'what's good tonight' or 'go to Saturn'."
REPEAT_S = 6.0  # the same spoken cue from the guidance or coaching loop, at most this often
CENTER_STEP_S = 0.5  # main-camera centering cue rate
DIRECTION_PROBE_S = 1.5  # after a left/right cue, look this long for the azimuth to move
FIX_STALE_S = 2.0  # a plate-solve fix older than this is too old to steer by
ENCODER_STALE_S = 1.0  # encoder positions older than this mean the board or cable is gone
PLANET_DISK_PX = 2000  # one clipped blob this big (raw px) is a disk: Jupiter ~30000, Sirius ~700
UNNAMED_PLANET = "the planet"  # a disk in the picture when the pointing can't name it
FIELD_RADIUS_DEG = 0.3  # the telescope's 32' x 18' view, center to corner
PLANET_NEAR_DEG = 3.0  # ... and it's the planet within this of where the scope points
AUTO_SOLVE_EVERY_S = 5.0  # background plate solves at night (back to back while not synced)
AUTO_SOLVE_NEAR_DEG = 5.0  # a background solve replaces model syncs this close (keeps the spread)
DARK_SUN_ALT_DEG = -10.0  # Sun lower than this: the views get the night stretch (twilight: as is)
# Guidance "on target" tolerance. 4' kept the user nudging in the field (encoder steps are 2.3');
# 8' still lands the target well inside the main camera's 32' x 18' view.
TOLERANCE_ARCMIN = 8.0
TOLERANCE_BARLOW_ARCMIN = 4.0  # half that with the 2x Barlow (half the field)
EXTENDED_TARGETS = {p.capitalize() for p in PLANETS} | {"Moon", *MOON_FEATURES}  # SER video
# Captures run until the page's Stop capture (the user sees the target near the frame edge);
# these are only safety limits.
RECORD_SECONDS = 300  # planetary video
STACK_SECONDS = 300  # deep-sky live stack
PROGRESS_S = 1.0  # capture progress to the page this often
COLLIMATION_STEP_S = 2.0  # time to turn a screw and let the image settle between checks
COLLIMATING = "We're checking collimation. Tap STOP to finish that first."
MIN_HORIZON_MARKS = 3
MIN_HORIZON_COVERAGE_DEG = 270  # less: a big unmarked gap gets a straight-line guess
SITE_MOVE_KM = 1.0  # moving farther than this from the model's site invalidates the mount model
EARTH_RADIUS_KM = 6371.0


def utcnow() -> datetime:
    return datetime.now(UTC)


class Session:
    def __init__(self, site: Site, position: Callable[[], tuple[float, float]] | None = None,
                 clock: Clock = utcnow, developer_override: bool = False,
                 finder: FinderSync | None = None, main_camera: Camera | None = None,
                 main_sensor: tuple[int, int] = MAIN_SENSOR_PX, data_dir: Path = Path("data"),
                 on_site_change: Callable[[Site], None] | None = None,
                 horizon: HorizonMask | None = None,
                 on_horizon_change: Callable[[HorizonMask], None] | None = None,
                 weather: Callable[[float, float, datetime], float | None] | None = None,
                 calibration: dict | None = None,
                 on_calibration_change: Callable[[dict], None] | None = None,
                 spots: list[Spot] | None = None, spot: str | None = None,
                 on_spots_change: Callable[[list[Spot], str | None], None] | None = None):
        """Pointing comes from `finder` (encoders + mount model + plate solving), or, for
        tests without a finder, from `position()` returning true (alt, az)."""
        if finder is not None and hasattr(finder, "camera"):
            finder.camera = tap(finder.camera)  # the tablet can show the last frame of each
        main_camera = tap(main_camera)
        self.site, self.clock, self.finder = site, clock, finder
        self.on_site_change = on_site_change  # e.g. persist the GPS fix, update simulators
        self.horizon = horizon or HorizonMask()  # treeline for the planner
        self.on_horizon_change = on_horizon_change
        self.spots, self.spot = list(spots or []), spot  # named places; the one we're at
        self.on_spots_change = on_spots_change
        self._horizon: list[tuple[float, float]] | None = None  # points during a horizon walk
        self.wizard: SetupWizard | None = None  # first-time setup at a location
        self._location_request = 0  # id of the GPS request whose answer we'd accept
        self._model_site = site  # site the current mount model was built for
        self.sky = Sky(weather)  # targets, plans and the cloud forecast
        self.position = finder.position if finder else position
        if finder is not None and hasattr(finder, "start"):  # background plate solver
            finder.start()
        self.override = developer_override
        self.catalog = self.sky.catalog
        self.target: str | None = None
        self.guide: Guide | None = None
        self._suggestions: list[str] = []
        self._resolved_at = -1e9
        self._last_said, self._last_said_at = "", -1e9  # tick speech, for _no_repeats
        self._collimation: CollimationCoach | None = None
        self.labels = FinderLabels(self.sky.targets())  # names on the finder view
        self._solved_try_at, self._solving = -1e9, False
        self._dark = (True, -1e9)  # dark(): (answer, when)
        self._said_later: list[dict] = []  # from background work, said on the next tick
        self._collimation_at = -1e9
        self.main_camera, self.main_sensor = main_camera, main_sensor
        self.zoom = {"finder": 1, "main": 1}  # the page's digital zoom per camera view
        self.barlow = False
        self.record_seconds = RECORD_SECONDS
        self.recorder = (Recorder(main_camera, main_sensor, data_dir / "captures") if main_camera else None)
        self._announced_done = True
        self.gallery_dir = data_dir / "gallery"
        self.stack_seconds = STACK_SECONDS
        self.stacker = (LiveStacker(main_camera, self.gallery_dir,
                                    preview_dir=data_dir / "live")
                        if main_camera else None)
        self._stack_done_announced = True
        self._preview_seen = 0
        self._processor = ThreadPoolExecutor(max_workers=1)  # one stacking job at a time
        self._holding = False  # asked the user to hold still for a fresh fix
        self._jobs: list[tuple[str, Future[StackResult]]] = []  # pictures being made, in order
        self.centerer = Centerer((main_sensor[0], main_sensor[1]))  # learns finder->main offset
        self._centering = False
        self._center_at = -1e9
        self._center_gave_up: str | None = None  # main camera lost this target: finder only
        self.main_box: list | None = None  # main field on the finder view (Align)
        self._align: AlignRun | None = None  # Align in progress
        self.main_in_finder: MainInFinder | None = None  # the last Align's fit
        self._progress_at, self._capture_at = -1e9, 0.0  # capture progress messages
        self._capture_radec = (0.0, 0.0)  # where a capture began (Recenter's target if unnamed)
        self._recentering = False  # a paused capture: guidance leads back to its target
        self.views = ViewSettings({"finder": getattr(finder, "camera", None), "main": main_camera})
        # G3: which way "right" turns the scope, learned from the first left/right push.
        self.right_is_plus_az = True
        self._direction_learner = DirectionLearner()
        self._direction_known = False
        self._direction_probe: tuple[str, float, float] | None = None  # (word, az, t)
        self.on_calibration_change = None  # set after restoring, so restoring doesn't save
        self._restore_calibration(calibration or {})
        self.on_calibration_change = on_calibration_change
        if finder is not None:
            finder.on_change = self._save_calibration
        self._center_limiter = CueLimiter()  # centering cues obey the guide's pacing
        # Commands and the guidance tick run on worker threads (camera calls block), so
        # serialize them: one camera capture or state change at a time.
        self._lock = threading.Lock()

    # --- target resolution -------------------------------------------------------------
    def describe(self, spoken: str) -> list[dict]:
        """What a target is and where it is right now (agent tool and "tell me about ...")."""
        return [say(self.sky.describe(spoken, self.site, self.clock(), self.horizon))]

    # --- spots ---------------------------------------------------------------------------
    def use_spot(self, spoken: str) -> list[dict]:
        """We're at a named spot: use its place and treeline (agent tool)."""
        with self._lock:
            name = match_name(spoken, [s.name for s in self.spots])
            if name is None:
                known = ", ".join(s.name for s in self.spots) or "none yet"
                return [say(f"I don't know a spot called {spoken}. Known spots: {known}.")]
            self._use(next(s for s in self.spots if s.name == name))
            return [say(f"OK, we're at {name}. I'll plan with its treeline.")]

    def save_spot(self, spot: Spot) -> list[dict]:
        """A treeline recorded with the tablet. We're standing there, so it becomes current."""
        with self._lock:
            self.spots = upsert(self.spots, spot)
            self._use(spot)
            return [say(f"Saved {spot.name} with {len(spot.mask.points)} treeline marks.")]

    def _use(self, spot: Spot) -> None:
        self._relocate(replace(self.site, lat_deg=spot.site.lat_deg, lon_deg=spot.site.lon_deg,
                               elevation_m=spot.site.elevation_m))  # a far spot re-syncs
        self.horizon, self.spot, self._suggestions = spot.mask, spot.name, []
        if self.on_horizon_change:
            self.on_horizon_change(self.horizon)
        self._spots_changed()

    def _spots_changed(self) -> None:
        if self.on_spots_change:
            self.on_spots_change(self.spots, self.spot)

    def timing(self, spoken: str) -> list[dict]:
        """When a target rises, clears the trees, is highest and sets (agent tool)."""
        return [say(self.sky.timing(spoken, self.site, self.clock(), self.horizon))]

    def names(self) -> list[str]:
        return self.sky.names()

    def altaz_of(self, name: str) -> tuple[float, float]:
        return self.sky.altaz_of(name, self.site, self.clock())

    # --- commands ------------------------------------------------------------------------
    def handle(self, text: str) -> list[dict]:
        intent = parse(text)
        if intent is not None and intent.name == "tonight":
            return self.tonight()  # planning is slow and read-only: keep it off the lock
        with self._lock:
            return self._handle(text)

    def action(self, do: str, target: str | None = None) -> list[dict]:
        """The page's buttons (Phase 3b): Go to, Capture, Focus, Sync, STOP. Answers are text."""
        if do == "tonight":
            return self.tonight()  # planning is slow and read-only: keep it off the lock
        with self._lock:
            if do == "goto":
                return self._goto_named(target or "")
            if do == "capture" and target:  # picked from "which one?": never a stop
                return self.capture(target) if not self._camera_busy() else [notice("Already capturing.")]
            handler = {"capture": self._toggle_capture, "stop": self._stop, "sync": self.sync,
                       "focus": self.focus_hint, "recenter": self._recenter,
                       "align": self.align}.get(do)
            return handler() if handler else [notice(f"{do.capitalize()} isn't ready yet.")]

    def _toggle_capture(self) -> list[dict]:
        """One button: start a capture, or stop the one running."""
        return self._stop_capture() if self._camera_busy() else self.capture()

    def _recenter(self) -> list[dict]:
        """Pause the capture and guide back to its target with the finder view's arrows; the
        same button resumes. Video and stack both continue where they left off."""
        job = next((j for j in (self.recorder, self.stacker) if j is not None and j.busy), None)
        if job is None:
            return [say("Recenter works during a capture. Start one first.")]
        if not job.current.paused.is_set():
            if self.finder is not None and not self.finder.synced:
                ok, msg = self.finder.sync()  # the arrows need to know where we point
                if not ok:
                    return [say(f"I can't guide back yet: {msg}")]
            job.pause()
            self._recentering = True
            alt, az = self._capture_altaz()
            self.guide = self._guide_to(alt, az)
            return [say("Paused. Follow the arrow on the finder view, then tap Resume."),
                    self._capture_state(job.current, "paused")]
        job.resume()
        self.guide, self._recentering = None, False
        return [say("Resumed."), self._capture_state(job.current, "recording")]

    def _capture_altaz(self) -> tuple[float, float]:
        """Where the capture's target is now: the named target, or the sky where it began."""
        if self.target in self.names():
            return self.altaz_of(self.target)
        return radec_to_altaz(*self._capture_radec, self.site, self.clock())

    def target_list(self, limit: int = 20) -> list[dict]:
        """The Go to list (Sky.target_list), planned outside the lock."""
        with self._lock:
            site, horizon = self.site, self.horizon
        return self.sky.target_list(site, horizon, self.clock(), limit)

    def goto_spoken(self, target: str) -> list[dict]:
        """Go to a target named in free text (agent tool): matched by name, never re-parsed."""
        with self._lock:
            return self._goto_named(target)

    def _handle(self, text: str) -> list[dict]:
        intent = parse(text)
        if intent is not None and intent.name in ("ready", "skip") and not self.wizard_active:
            # Outside setup, "okay" is conversation, and "next step" just means "next".
            intent = Intent("next") if "next" in text.lower() else None
        if intent is not None and intent.name in ("setup", "ready", "skip"):
            return self._wizard_command(intent.name)
        if intent is None:
            return [say(NOT_UNDERSTOOD)]
        if intent.name == "goto":
            return self._goto_named(intent.target or "")
        if intent.name == "describe":
            return self.describe(intent.target or "")
        if intent.name in ("barlow_on", "barlow_off"):
            return self._set_barlow(intent.name == "barlow_on")
        command = {
            "stop": self._stop, "focus": self.focus_hint, "capture": self.capture,
            "stop_capture": self._stop_capture, "sync": self.sync,
            "finder_focus": self.focus_hint, "tonight": self.tonight, "next": self._next,
            "horizon_start": self.start_horizon, "horizon_mark": self.mark_horizon,
            "location": self.request_location, "where": self.where,
            "collimate": self.start_collimation,
        }.get(intent.name)
        if command is None:
            return [say(f"{intent.name.replace('_', ' ').capitalize()} isn't ready yet.")]
        return command()

    def _stop(self) -> list[dict]:
        """'Stop' (or 'done') ends whatever is going on, most specific first."""
        if self.wizard is not None and self.wizard.active:
            self.wizard = None
            return [say("OK, setup stopped. Say 'set up the telescope' to start again.")]
        if self._horizon is not None:  # "done" / "stop" finishes the horizon walk
            return self.finish_horizon()
        if self._collimation is not None:
            self._collimation = None
            return [say("OK, collimation check stopped. Remember to refocus.")]
        if self._camera_busy():  # "stop" while taking a picture ends the picture
            return self._stop_capture()
        self._clear_guidance()
        return [say("Stopped.")]

    def _clear_guidance(self) -> None:
        """Forget the target and stop guiding, centering and any Align."""
        self.target, self.guide, self._centering, self._align = None, None, False, None

    def _set_barlow(self, inserted: bool) -> list[dict]:
        self.barlow = inserted
        if self.guide is not None:  # an active guide switches tolerance too
            self.guide.tol_deg = self._tolerance_arcmin() / 60
        return [say("Got it. The Barlow changes focus: refocus with the focus number before "
                    "capturing.")]

    def _stop_capture(self) -> list[dict]:
        if self._recentering:  # stopping a paused capture ends its guidance too
            self.guide, self._recentering = None, False
        if self.stacker is not None and self.stacker.busy:
            self.stacker.stop()
            return [say("Stopping. I'll keep what's stacked so far.")]
        if self.recorder is None or not self.recorder.busy:
            return [say("We're not recording."), {"type": "capture", "state": "idle"}]
        self.recorder.stop()
        return [say("Stopping the recording.")]

    def _next(self) -> list[dict]:
        if not self._suggestions:
            return [say("Ask me what's good tonight first.")]
        return self.goto(self._suggestions.pop(0))

    def sync(self) -> list[dict]:
        if self.finder is None:
            return [say("There's no finder camera connected.")]
        return [say(self.finder.sync()[1])]

    def focus_hint(self) -> list[dict]:
        """Focus is visual: the number on each camera view (higher is sharper). No coach."""
        return [say("Turn the focus knob to make the focus number on the camera view as high "
                    "as you can.")]

    def capture(self, choice: str | None = None) -> list[dict]:
        """Record or stack what's in the telescope view (_what_to_capture), or `choice` when
        the page asked which one. Planets and the Moon record a video; anything else stacks."""
        if self.recorder is None:
            return [say("There's no telescope camera connected.")]
        if self._collimation is not None:
            return [say(COLLIMATING)]
        if self._camera_busy():
            return [say("I'm already recording." if self.recorder.busy else "I'm already stacking.")]
        if choice is None:
            found = self._what_to_capture()
            if len(found) > 1:
                return [say("I see more than one thing in the telescope. Which one?"),
                        {"type": "choose", "options": found}]
            # nothing recognized: the Go to target (pointing unknown) or the field
            choice = found[0] if found else (self.target or "the field")
        target = choice if choice in self.names() else None  # the session's, once it starts
        video = target in EXTENDED_TARGETS or choice == UNNAMED_PLANET  # planets, Moon
        try:
            if video:
                self.recorder.start(choice, self.record_seconds)
            else:
                self.stacker.start(choice, self.stack_seconds)  # deep-sky: live stack short subs
        except CaptureRefused as e:
            return [say(str(e))]  # nothing started: guidance carries on as before
        self.target = target
        self._picture_started()
        if video:
            self._announced_done = False
            return [say("Recording. Tap Stop capture when it nears the edge of the picture.")]
        self._stack_done_announced, self._preview_seen = False, 0
        return [say(f"Stacking short pictures of {choice}. Watch it build up on the screen. "
                    "Try not to touch the telescope.")]

    def _what_to_capture(self) -> list[str]:
        """What's in the telescope view, by name: a planet's disk seen in the picture, then the
        planets, Moon and Go to targets inside its field when the pointing is known."""
        found = []
        disk = self._planet_in_view()
        if disk:
            found.append(UNNAMED_PLANET if disk == "planet" else disk)
        if getattr(self.finder, "synced", False):
            alt, az = self.centerer.offset.main_center(*self.position())
            found += self.sky.in_field(alt, az, FIELD_RADIUS_DEG, self.site, self.clock())
        if UNNAMED_PLANET in found and len(found) > 1 and any(f in EXTENDED_TARGETS for f in found):
            found.remove(UNNAMED_PLANET)  # the disk is the planet named by the pointing
        if self.target in MOON_FEATURES and "Moon" in found:  # Go to Tycho, then Capture: Tycho
            found[found.index("Moon")] = self.target
        return list(dict.fromkeys(found))

    def _planet_in_view(self) -> str | None:
        """If the main view shows a disk (one clipped blob far bigger than any star's): the
        planet or Moon nearest where the scope points, or "planet" when none is near (the
        pointing may be off: a disk is still a disk)."""
        frame = self.camera_frame("main")
        if frame is None or not has_disk(frame[0], PLANET_DISK_PX, SATURATED_RAW):
            return None
        sep, body = self.sky.nearest_body(*self.position(), self.site, self.clock())
        return body if sep < PLANET_NEAR_DEG else "planet"

    def _picture_started(self) -> None:
        """The picture has the camera: guidance goes quiet (no "right a little" while the user
        was asked not to touch the telescope). The target is kept, and the sky position too,
        so Recenter can lead back to an unnamed field."""
        self._capture_at = time.monotonic()
        self._centering, self.guide = False, None
        main_at = self.centerer.offset.main_center(*self.position())  # the field, not the finder
        self._capture_radec = altaz_to_radec(*main_at, self.site, self.clock())

    def _camera_busy(self) -> bool:
        return any(job is not None and job.busy for job in (self.recorder, self.stacker))

    def _goto_named(self, spoken: str) -> list[dict]:
        name = match_name(spoken, self.names())
        return self.goto(name) if name else [say(f"I don't know {spoken}.")]

    def goto(self, name: str) -> list[dict]:
        if self._camera_busy():
            return [say("I'm taking a picture. Stop it first (Capture or STOP), then we can move.")]
        pre: list[dict] = []
        if self.finder is not None and not self.finder.synced:
            ok, msg = self.finder.sync()  # need to know where we point before guiding
            if not ok:
                return [say(f"Before we go, I need to see the stars. {msg}")]
            pre = [say(msg)]
        alt, az = self.altaz_of(name)
        safe = self.target_safety(alt, az)
        if not safe.ok:
            return [say(f"I can't go to {name}: it's {safe.reason}.")]
        self._centering = False
        self.target, self.guide = name, self._guide_to(alt, az)
        self._center_gave_up = None  # a new "go to" tries the main camera again
        return [*pre, say(f"Let's find {name}.")]

    def _plan(self) -> tuple[dict, datetime | None, float | None]:
        """Tonight's plan (Sky.plan) for a consistent snapshot, planned outside the lock."""
        with self._lock:
            site, horizon = self.site, self.horizon
        return self.sky.plan(site, horizon, self.clock())

    def tonight(self) -> list[dict]:
        text, suggestions = tonight_text(*self._plan(), self.clock())
        if suggestions is not None:
            with self._lock:
                self._suggestions = suggestions
        return [say(text)]

    def tonight_by_category(self) -> str:
        choices, _, clouds = self._plan()
        return by_category_text(choices, clouds)

    # --- what the tablet can look at ------------------------------------------------------
    def camera(self, name: str):
        """The named camera (finder or main), which keeps its last frame, or None."""
        return getattr(self.finder, "camera", None) if name == "finder" else self.main_camera

    def camera_frame(self, name: str):
        """(raw frame, bayer, age in s) the named camera last captured, or None."""
        cam = self.camera(name)
        if cam is None or getattr(cam, "last", None) is None:
            return None
        return cam.last, cam.bayer, time.monotonic() - cam.last_at

    def view_frame(self, name: str):
        """camera_frame for the page's view: while a planet records only its ROI, that ROI is
        pasted into the full live view from just before, so the whole field stays on screen."""
        frame = self.camera_frame(name)
        rec = self.recorder if name == "main" else None
        bg, roi = (rec.background, rec.roi) if rec else (None, None)  # once: the recorder clears them
        if frame is None or bg is None or roi is None:
            return frame
        raw, bayer, age = frame
        bh, bw = bg.shape
        if raw.shape != (roi.height, roi.width) or raw.shape == (bh, bw) or roi.x + roi.width > bw \
                or roi.y + roi.height > bh:
            return frame  # paused (whole field), a frame from before the ROI changed, or no fit
        full = bg.copy()
        full[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width] = raw
        return full, bayer, age

    def finder_labels(self) -> list[list]:
        """Names on the finder view, [[name, kind, x, y], ...] with x, y as fractions from the
        view's center (kind: star, target or planet), from the last plate solve's map and where
        the finder points now (astro/pointing/labels.py). The map is saved, so names come back
        after a restart without a new solve; empty until the first solve ever."""
        sol = getattr(self.finder, "last_solution", None)
        frame = self.camera_frame("finder")
        if sol is None and self.labels.map is None:  # never solved: solve now, in the background
            self._solve_soon()
            return []
        if frame is None:
            return []
        h, w = frame[0].shape
        found, new_map = self.labels.labels(sol, (w, h), self.finder.position(), self.site,
                                            self.clock(), time.monotonic(), ROTATE["finder"])
        if new_map:
            self._save_calibration()
        return found

    def adjust_camera(self, name: str, zoom: int | None = None) -> None:
        """The page's digital zoom for a camera view (brightness is the page's own)."""
        if name in self.zoom and zoom in ZOOMS:
            self.zoom[name] = zoom

    def connections(self) -> dict:
        """What's plugged in and working, for the status line: True, False or None (not fitted)."""
        def cam(c):
            return None if c is None else bool(getattr(c, "connected", True))

        age = getattr(self.finder, "encoder_age", None)
        return {"finder": cam(getattr(self.finder, "camera", None)), "main": cam(self.main_camera),
                "encoders": None if age is None else age() < 2 * ENCODER_STALE_S}

    def debug_info(self) -> dict:
        """Everything the engineer would ask for, as sections of plain values (no camera calls:
        the SDK isn't safe to query while another thread captures)."""
        now = self.clock()
        info: dict = {"time": {"utc": now.isoformat(timespec="seconds"),
                               "local": now.astimezone().isoformat(timespec="seconds")},
                      "site": {"lat": self.site.lat_deg, "lon": self.site.lon_deg,
                               "elevation_m": self.site.elevation_m}}
        f = self.finder
        if f is not None:
            alt, az = self.position()
            syncs, rms = f.alignment()
            pointing = {"alt_deg": round(alt, 2), "az_deg": round(az, 2), "synced": f.synced,
                        "syncs": syncs, "model_rms_arcmin": rms}
            if (age := getattr(f, "fix_age", None)) is not None and f.synced:
                pointing["fix_age_s"] = round(age(), 1)  # seconds since the last good solve
            info["pointing"] = pointing
            if (counts := getattr(f, "raw_counts", None)) is not None:
                try:
                    az_c, alt_c = counts()
                    info["encoders"] = {"az_counts": az_c, "alt_counts": alt_c}
                except (OSError, RuntimeError, ValueError) as e:  # a dead serial port must not hide the rest
                    info["encoders"] = {"error": str(e)}
            if (sol := f.last_solution) is not None:
                info["plate_solve"] = {k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in asdict(sol).items()}
                info["plate_solve"]["confidence"] = round(sol.confidence, 1)
        for name, cam in (("finder", getattr(f, "camera", None)), ("main", self.main_camera)):
            if cam is None:
                info[f"{name}_camera"] = {"connected": False}
            else:
                frame = self.camera_frame(name)
                info[f"{name}_camera"] = {
                    "exposure_s": cam.exposure_s, "gain": cam.gain,
                    "frame_px": list(frame[0].shape[::-1]) if frame else None,
                    "frame_age_s": round(frame[2], 1) if frame else None}
        info["guidance"] = {"target": self.target, "guiding": self.guide is not None,
                            "centering": self._centering, "holding": self._holding}
        info["system"] = {"load_1_5_15": [round(x, 2) for x in os.getloadavg()]}
        return info

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
            later, self._said_later = self._said_later, []
            return self._no_repeats([*later, *self._capture_progress(t), *self._tick(t)], t)

    def _capture_progress(self, t: float) -> list[dict]:
        """While a capture runs, its frame count and time for the page, every PROGRESS_S."""
        if t - self._progress_at < PROGRESS_S:
            return []
        for job in (self.recorder and self.recorder.current, self.stacker and self.stacker.current):
            if job is not None and not job.done.is_set():
                self._progress_at = t
                return [self._capture_state(job, "paused" if job.paused.is_set() else "recording")]
        return []

    def _capture_state(self, job, state: str) -> dict:
        """A capture's progress for the page: frames, time, and when the target reaches the
        edge of the main camera (seconds, or None until the drift is measured)."""
        left = job.edge.seconds_left() if job.edge else None
        where = job.edge.position() if job.edge else None
        return {"type": "capture", "state": state, "kind": "video" if hasattr(job, "lost") else "stack",
                "name": job.name, "frames": job.frames,
                "seconds": round(time.monotonic() - self._capture_at),
                "edge_s": None if left is None else round(left),
                "target_xy": None if where is None else [round(where[0], 3), round(where[1], 3)]}

    def _no_repeats(self, out: list[dict], t: float) -> list[dict]:
        """Coaching said the same words every second in the field: say a phrase again only after
        REPEAT_S (state updates and new phrases go through)."""
        kept = []
        for m in out:
            if m.get("type") == "say":
                if m["text"] == self._last_said and t - self._last_said_at < REPEAT_S:
                    continue
                self._last_said, self._last_said_at = m["text"], t
            kept.append(m)
        return kept

    def _solve_soon(self) -> None:
        """A plate solve in the background: every AUTO_SOLVE_EVERY_S, back to back while not
        synced. Without one the encoders mean nothing (2026-10-10: the board rebooted, the saved model was dropped, and
        captures, names and Recenter all used a pointing 45 degrees off)."""
        synced = getattr(self.finder, "synced", True)
        if not hasattr(self.finder, "sync") or self._solving or (
                synced and time.monotonic() - self._solved_try_at < AUTO_SOLVE_EVERY_S):
            return
        self._solved_try_at, self._solving = time.monotonic(), True
        threading.Thread(target=self._auto_solve, args=(synced,), daemon=True).start()

    def _auto_solve(self, was_synced: bool) -> None:
        try:
            extra = {"replace_near_deg": AUTO_SOLVE_NEAR_DEG} if isinstance(self.finder, FinderSync) else {}
            ok, why = self.finder.sync(fresh=True, **extra)
        except Exception:  # a bug in one solve must not stop the next ones
            log.exception("auto solve failed")
            return
        finally:
            self._solving = False
        log.info("auto solve", extra={"data": {"ok": ok, "why": why}})
        if ok and not was_synced:
            with self._lock:  # tick swaps the list under it
                self._said_later.append(say("Found where the scope points (plate solve)."))

    def _tick(self, t: float) -> list[dict]:
        self.views.update(t, self.daytime(), self._camera_busy(), hold=self._align is not None)
        if self.finder is not None and not self.daytime() and not self._camera_busy():
            self._solve_soon()  # every 5 s; back to back while the encoders mean nothing
        if self._align is not None:
            return self._align_step()
        rec = self.recorder.current if self.recorder else None
        if rec is not None and rec.done.is_set() and not self._announced_done:
            self._announced_done = True
            if rec.frames < MIN_FRAMES:
                return [say(f"{rec.error or 'Done.'} I only got {rec.frames} frames, "
                            "not enough for a picture."), {"type": "capture", "state": "idle"}]
            job = self._processor.submit(process_ser, rec.path, self.gallery_dir)
            self._jobs.append((rec.name, job))
            done = rec.error or "Done."  # e.g. drifted out of view: still make the picture
            return [say(f"{done} I saved {rec.frames} frames. I'm making your picture now."),
                    {"type": "capture", "state": "processing", "frames": rec.frames}]
        if self._jobs and self._jobs[0][1].done():
            return self._announce_picture()
        if (live := self.stacker.current if self.stacker else None) is not None:
            if live.preview_version > self._preview_seen:  # tablet refreshes the live view
                self._preview_seen = live.preview_version
                if not live.done.is_set():
                    return [{"type": "live", "url": f"/live/{live.preview.name}",
                             "frames": live.frames}]
            if live.done.is_set() and not self._stack_done_announced:
                self._stack_done_announced = True
                return self._announce_stack(live)
        if self._collimation is not None:
            return self._collimation_step(t)
        if (self._centering or self.guide is not None) and (lost := self._pointing_lost()):
            self.guide, self._centering = None, False  # keep the target: "go to" it again later
            return [say(lost)]
        if self._centering or self.guide is not None:
            hold = self._hold_for_fix()
            if hold is not None:
                return hold
        if self._centering:
            return self._center_step(t)
        if self._recentering and not self._camera_busy():  # the capture ended while paused
            self.guide, self._recentering = None, False
        if self.guide is None or (self.target is None and not self._recentering):
            self._direction_probe, self._holding = None, False  # forget unrelated motion
            return []
        if t - self._resolved_at >= TARGET_REFRESH_S:
            self._resolved_at = t
            alt, az = self._capture_altaz() if self._recentering else self.altaz_of(self.target)
            if not self.target_safety(alt, az).ok:
                self.target, self.guide = None, None
                return [say("Stopping: the target is no longer safe to point at.")]
            self.guide.target = self._aim(alt, az)
        alt_now, az_now = self.position()
        learned = self._learn_direction(az_now, t)  # before the cue: a flip must apply to it
        state, cue = self.guide.update(alt_now, az_now, t)
        out = [{"type": "state", "target": self.target or "the capture", "right_is_plus_az": self.right_is_plus_az,
                **asdict(state)}, *learned]
        if cue and learned:
            cue = None  # let "Got it…" be heard; the next tick brings the (corrected) cue
        if cue:
            self._start_direction_probe(cue.text, az_now, t)
            out.append(say(cue.text))
            if cue.text == "stop" and state.on_target and self._should_center():
                self._centering, self.guide = True, None  # finish with the main camera
                self.centerer.restart(self.altaz_of(self.target))  # true, uncorrected target
                self._center_limiter = CueLimiter()
        return out

    def _hold_for_fix(self) -> list[dict] | None:
        """Plate solving only: while the fix is stale (the scope is moving), don't steer by it.
        Returns the messages for this tick, or None when the fix is fresh."""
        fix_age = getattr(self.finder, "fix_age", None)
        if fix_age is None or fix_age() <= FIX_STALE_S:
            self._holding = False
            return None
        self._direction_probe = None  # motion during the pause says nothing about left/right
        if self._holding:
            return []
        self._holding = True
        return [{"type": "hold"}, say("Hold still for a second so I can see where we are.")]

    def _pointing_lost(self) -> str | None:
        """Why guidance can't trust the pointing any more, or None."""
        if self.finder is None:
            return None
        age = getattr(self.finder, "encoder_age", None)
        if age is not None and age() > ENCODER_STALE_S:
            return "I lost the telescope's position sensors, so I stopped guiding. Check the cable."
        if not self.finder.synced:  # e.g. the encoder board restarted and was reset
            return ("I lost track of where the telescope points. Point the finder at clear sky and "
                    "tap More, then Sync now.")
        return None

    # --- calibration that survives a restart (CV7) ----------------------------------------
    def calibration(self) -> dict:
        """What's worth keeping: left/right, main-camera axes and offset, the mount model."""
        axes, offset = self.centerer.axes, self.centerer.offset
        data = {
            "right_is_plus_az": self.right_is_plus_az if self._direction_known else None,
            "camera_axes": axes.matrix.tolist() if axes.matrix is not None else None,
            "main_offset": ({"d_az_sky_deg": offset.d_az_sky_deg, "d_alt_deg": offset.d_alt_deg,
                             "observations": offset.observations}
                            if offset.observations else None),
            "mount": None,
            "main_box": self.main_box,
            "finder_map": self.labels.map.tolist() if self.labels.map is not None else None,
            "main_in_finder": ({"a": self.main_in_finder.a.tolist(),
                                "center": self.main_in_finder.center.tolist(),
                                "residual_px": self.main_in_finder.residual_px}
                               if self.main_in_finder else None),
        }
        if hasattr(self.finder, "model_dict") and self.finder.synced:
            data["mount"] = {"site": [self.site.lat_deg, self.site.lon_deg], **self.finder.model_dict()}
        return data

    def _save_calibration(self) -> None:
        if self.on_calibration_change:
            self.on_calibration_change(self.calibration())

    def _restore_calibration(self, data: dict) -> None:
        if data.get("right_is_plus_az") is not None:
            self.right_is_plus_az, self._direction_known = data["right_is_plus_az"], True
        if data.get("camera_axes") is not None:
            self.centerer.axes.matrix = np.array(data["camera_axes"])
        self.main_box = data.get("main_box")  # main field corners on the finder view
        if data.get("finder_map") is not None:
            self.labels.map = np.array(data["finder_map"])
        if (mif := data.get("main_in_finder")) is not None:
            self.main_in_finder = MainInFinder(np.array(mif["a"]), np.array(mif["center"]),
                                               mif["residual_px"])
        if data.get("main_offset"):
            o = data["main_offset"]
            self.centerer.offset = MainOffset(o["d_az_sky_deg"], o["d_alt_deg"], o["observations"])
        mount = data.get("mount")
        if mount and self.finder is not None and hasattr(self.finder, "model"):
            boots = getattr(self.finder, "encoder_boots", None)
            if boots is not None and boots() > 0:
                return  # the encoder board booted: its counts are 0, the saved model is stale
            lat, lon = mount["site"]
            if _km_between(lat, lon, self.site) <= SITE_MOVE_KM:  # same place, same encoder counts: no re-sync needed
                self.finder.model = calibration_store.model_from_dict(mount)
                self.finder.synced = True
                self._model_site = replace(self.site, lat_deg=lat, lon_deg=lon)  # its origin

    def _learn_direction(self, az: float, t: float) -> list[dict]:
        """G3: after the first left/right cue, the first clear azimuth move (within
        DIRECTION_PROBE_S) shows the user's sense of left and right; adopt it once."""
        if self._direction_known or self._direction_probe is None:
            return []
        word, az0, t0 = self._direction_probe
        learned = self._direction_learner.observe(word, wrap180(az - az0))
        if learned is None:
            if t - t0 >= DIRECTION_PROBE_S:
                self._direction_probe = None  # no clear move in time: try the next cue
            return []
        self._direction_known, self._direction_probe = True, None
        if learned == self.right_is_plus_az:
            self._save_calibration()
            return []
        self.right_is_plus_az = learned
        if self.guide is not None:
            self.guide.right_is_plus_az = learned
        self._save_calibration()
        return [say("Got it, I'll use your left and right from now on.")]

    def _start_direction_probe(self, spoken: str, az: float, t: float) -> None:
        if self._direction_known or self._direction_probe is not None:
            return
        word = next((w for w in ("left", "right") if w in spoken.split(", ")[0].split()), None)
        if word:
            self._direction_probe = (word, az, t)

    def align(self) -> list[dict]:
        """Align (Phase 3b): with a bright star in the main view, watch it drift in both cameras
        for ALIGN_S, then fit where the main camera's view sits on the finder's."""
        if self.finder is None or self.main_camera is None:
            return [say("Align needs both cameras connected.")]
        if self._camera_busy():
            return [say("Stop the capture first, then Align.")]
        solved, _ = self.finder.sync(fresh=True)  # the solve turns finder pixels into sky
        frame = self.camera_frame("finder")
        h, w = frame[0].shape if frame else (960, 1280)
        near = self.main_in_finder.center if self.main_in_finder else np.array([w / 2, h / 2])
        self.guide, self._centering = None, False
        self._align = AlignRun(near, (w, h), self.finder.last_solution if solved else None)
        box_only = "" if solved else (" The finder can't plate-solve right now (clouds?), so this "
                                      "moves the box only; Go to keeps its old aim.")
        return [say(f"Aligning: keep the bright star in the telescope view and don't touch the scope "
                    f"for {ALIGN_S:.0f} seconds.{box_only}")]

    def _align_step(self) -> list[dict]:
        run = self._align
        for name in ("finder", "main"):
            run.add(name, frame_with_time(self.camera(name)))
        if time.monotonic() < run.until:
            return []
        self._align = None
        fit = run.result()
        if isinstance(fit, str):
            return [say(fit)]
        w, h = run.finder_size
        self.main_box = box_on_finder_view(fit, run.main_size, (w, h), ROTATE["finder"])
        self.main_in_finder = fit
        if run.sol is None:  # no solve: the box from the drift alone, Go to's aim unchanged
            self._save_calibration()
            return [say(f"Box moved: the telescope's view is on the finder view, turned "
                        f"{fit.rotation_deg:.0f} degrees. Go to keeps its old aim until an Align "
                        "with the finder seeing clear sky.")]
        east, north = finder_offset_to_sky(fit.center[0] - w / 2, fit.center[1] - h / 2, run.sol, w)
        self.centerer.offset = sky_offset_to_altaz(run.sol, east, north, self.site, self.clock())
        self._save_calibration()
        return [say(f"Aligned. The telescope points {np.hypot(east, north) * 60:.0f} arcminutes "
                    f"from the finder's center, turned {fit.rotation_deg:.0f} degrees. Its box is "
                    "on the finder view.")]

    def _guide_to(self, alt: float, az: float) -> Guide:
        return Guide(*self._aim(alt, az), tolerance_arcmin=self._tolerance_arcmin(),
                     right_is_plus_az=self.right_is_plus_az)

    def _aim(self, alt: float, az: float) -> tuple[float, float]:
        """Where the finder model should point so the target lands in the main camera."""
        if self.centerer.offset.observations:
            return self.centerer.offset.correct(alt, az)
        return alt, az

    def _should_center(self) -> bool:
        return (self.main_camera is not None and self.target in EXTENDED_TARGETS
                and not self._camera_busy() and self.target != self._center_gave_up)

    def _center_step(self, t: float) -> list[dict]:
        if t - self._center_at < CENTER_STEP_S:
            return []
        self._center_at = t
        if self._camera_busy():  # a recording or stack took the camera
            self._centering = False
            return []
        try:
            frame = self.main_camera.capture()
        except (RuntimeError, OSError) as e:
            self._centering = False
            return [say(f"The telescope camera stopped responding. ({e})")]
        step = self.centerer.update(self.position(), brightest_blob(frame))
        if step.lost:  # back to finder guidance, as the words promise
            self._centering = False
            self._center_gave_up = self.target  # don't bounce straight back (looped 2x/s outside)
            alt, az = self.altaz_of(self.target)
            self.guide = self._guide_to(alt, az)
        elif step.done:
            self._centering = False
            self._save_calibration()  # camera axes and finder-to-main offset are learned now
        urgent = step.done or step.lost or step.say == CALIBRATED
        spoken = self._center_limiter.speak(step.say, t, urgent) if step.say else None
        return [say(spoken)] if spoken else []

    def start_collimation(self) -> list[dict]:
        """F3: coach the primary mirror's screws from a defocused star in the main camera."""
        if self.main_camera is None:
            return [say("There's no telescope camera connected.")]
        if self._camera_busy():
            return [say("Let's finish what the camera is doing first.")]
        self.guide, self._centering = None, False
        self._collimation = CollimationCoach()
        return [say("Let's check collimation. Center a bright star, then turn the focus knob "
                    "until it becomes a big donut with a dark middle. Say stop when we're done.")]

    def _collimation_step(self, t: float) -> list[dict]:
        if t - self._collimation_at < COLLIMATION_STEP_S or self._collimation is None:
            return []
        self._collimation_at = t
        try:
            frame = self.main_camera.capture()
        except (RuntimeError, OSError) as e:
            self._collimation = None
            return [say(f"The telescope camera stopped responding, so I stopped. ({e})")]
        cue, done = self._collimation.check(frame)
        if done:
            self._collimation = None
        return [say(cue)] if cue else []

    def _announce_picture(self) -> list[dict]:
        name, job = self._jobs.pop(0)
        try:
            result = job.result()
        except (ValueError, OSError) as e:
            log.exception("picture failed", extra={"data": {"name": name}})
            return [say(f"I couldn't make the picture of {name}: {e}"), {"type": "capture", "state": "idle"}]
        if not self._jobs:  # nothing queued still needs its raw video
            prune(self.recorder.out_dir)  # keep only the newest raw videos
        return [say(f"Your picture of {name} is ready. Tap Pictures to see it."),
                {"type": "picture", "file": result.path.name}]

    def _announce_stack(self, live) -> list[dict]:
        if not live.frames:
            return [say(live.error or f"I couldn't stack any pictures of {live.name}."), {"type": "capture", "state": "idle"}]
        why = f"{live.error} " if live.error else ""
        return [say(f"{why}Your picture of {live.name} is ready, from {live.frames} short "
                    "pictures. Tap Pictures to see it."),
                {"type": "picture", "file": live.picture.name}]

    # --- horizon walk (calibration wizard) ------------------------------------------------
    def start_horizon(self) -> list[dict]:
        if self._collimation is not None:
            return [say(COLLIMATING)]
        if self.finder is not None and not self.finder.synced:
            ok, msg = self.finder.sync()  # marks are only as good as the pointing
            if not ok:
                return [say(f"Before the horizon walk, I need to see the stars. {msg}")]
        self._horizon = []
        self._clear_guidance()
        return [say("Let's record the treeline. Point the telescope just above the trees and "
                    "say 'mark'. Then move along the treeline and mark again. "
                    "Eight to fifteen marks all the way around is ideal. Say 'done' to finish.")]

    def mark_horizon(self) -> list[dict]:
        if self._horizon is None:
            return [say("Say 'start the horizon walk' first.")]
        alt, az = self.position()
        self._horizon.append((az % 360, alt))
        return [say(f"Marked {len(self._horizon)}.")]

    def finish_horizon(self) -> list[dict]:
        points, self._horizon = self._horizon or [], None
        if len(points) < MIN_HORIZON_MARKS:
            return [say(f"I only have {len(points)} marks, so I kept the old horizon. "
                        "We need at least three.")]
        self.horizon = HorizonMask(tuple(sorted(points)))
        if self.on_horizon_change:
            self.on_horizon_change(self.horizon)
        if self.spot:  # the telescope's walk replaces the spot's (tablet) treeline
            self.spots = with_mask(self.spots, self.spot, self.horizon)
            self._spots_changed()
        msg = f"Saved the treeline from {len(points)} marks. I'll only suggest things above it."
        if (covered := _azimuth_coverage([az for az, _ in points])) < MIN_HORIZON_COVERAGE_DEG:
            msg += (f" Your marks only go about {covered:.0f} degrees around, so I guessed a "
                    "straight line across the rest. Mark the other side when you can.")
        return [say(msg)]

    @property
    def wizard_active(self) -> bool:
        return self.wizard is not None and self.wizard.active

    def _wizard_command(self, name: str) -> list[dict]:
        if name == "setup":
            if self._collimation is not None:
                return [say(COLLIMATING)]
            if self.finder is None:
                return [say("There's no finder camera, so I can't run setup.")]
            self.wizard = SetupWizard(self.request_location, self.finder.sync,
                                      self.finder.alignment, self.start_horizon,
                                      self.cancel_location)
            self._clear_guidance()
            return self.wizard.start()
        return self.wizard.ready() if name == "ready" else self.wizard.skip()

    def status_text(self) -> str:
        """Short facts for the agent to summarize (not spoken verbatim); a locked snapshot."""
        with self._lock:
            return self._status_facts()

    def _status_facts(self) -> str:
        facts = [f"at spot: {self.spot}" if self.spot else "spot: not chosen"]
        if self.spots:
            facts.append("known spots: " + ", ".join(s.name for s in self.spots))
        if self.finder is not None:
            facts.append("aligned with the stars" if self.finder.synced else "not aligned yet")
            if sol := getattr(self.finder, "last_solution", None):
                facts.append(f"last plate solve: {sol.matches} stars matched, "
                             f"false-match odds {sol.false_prob:.0e}, roll {sol.roll_deg:.0f} deg, "
                             f"{sol.scale_arcsec_px:.1f} arcsec per pixel")
        facts.append(f"target: {self.target}" if self.target else "no target")
        facts.append("Barlow in" if self.barlow else "no Barlow")
        if self._camera_busy():
            facts.append("a picture is being taken")
        pictures = list(self.gallery_dir.glob("*.png")) if self.gallery_dir.exists() else []
        facts.append(f"{len(pictures)} pictures in the gallery")
        facts.append("treeline recorded" if self.horizon.points != HorizonMask().points
                     else "treeline not recorded (default 20 degrees)")
        clouds = self.sky.cached_clouds  # never fetch the forecast under the lock
        facts.append(f"cloud cover about {clouds:.0f}%" if clouds is not None
                     else "no weather forecast (offline)")
        return "; ".join(facts)

    def request_location(self) -> list[dict]:
        """Ask the tablet for a GPS fix; it answers with a `location` message carrying the
        same id (see set_location). A newer request or cancel_location() voids older ones."""
        self._location_request += 1
        return [say("Let me ask the tablet where we are. Please allow location access."),
                {"type": "get_location", "id": self._location_request}]

    def cancel_location(self) -> None:
        """Ignore the answer to the outstanding request (setup skipped it)."""
        self._location_request += 1

    def set_location(self, lat: float, lon: float, elevation_m: float | None,
                     accuracy_m: float | None, request_id: int | None = None) -> list[dict]:
        """Use a GPS fix from the tablet. Moving resets pointing, so the next goto re-syncs.

        A fix answering a cancelled or superseded request (`request_id`) is ignored."""
        if request_id is not None and request_id != self._location_request:
            return []
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return [say("That location doesn't look right, so I kept the old one.")]
        self._relocate(replace(self.site, lat_deg=lat, lon_deg=lon, elevation_m=self.site.elevation_m
                               if elevation_m is None else elevation_m))
        near = f", accurate to about {accuracy_m:.0f} meters" if accuracy_m else ""
        out = [say(f"Got it, I know where we are{near}.")]
        if self.wizard_active:  # setup was waiting for this before the first sync
            out += self.wizard.location_done()
        return out

    def _relocate(self, new: Site) -> None:
        """Move to `new`. Far from where the mount model was built (great-circle distance, so a
        chain of small updates can't drift away), pointing, treeline and plans start over."""
        ref = self._model_site
        self.site = new
        if _km_between(ref.lat_deg, ref.lon_deg, new) > SITE_MOVE_KM:
            self._model_site = new
            if self.finder is not None:
                self.finder.reset(new)
            self._clear_guidance()
            self._horizon, self._suggestions = None, []
            self.horizon = HorizonMask()
            if self.spot:
                self.spot = None
                self._spots_changed()
            if self.on_horizon_change:
                self.on_horizon_change(self.horizon)
        if self.on_site_change:
            self.on_site_change(new)

    def location_failed(self, message: str, request_id: int | None = None) -> list[dict]:
        """The tablet couldn't give a GPS fix: keep the saved site (setup moves on with it)."""
        if request_id is not None and request_id != self._location_request:
            return []  # answer to a request that was cancelled or superseded
        out = [say(f"I couldn't get the tablet's location. {message} Using the saved location.")]
        if self.wizard_active:
            out += self.wizard.location_done()
        return out

    def target_safety(self, alt: float, az: float) -> SafetyResult:
        """Sun, daytime and below-horizon checks, plus the local treeline (horizon mask)."""
        result = check_target(alt, az, self.site, self.clock(), self.override)
        if self.override and self.daytime():  # the page's daytime toggle: all but the Sun
            return result if result.reason == "too close to the Sun" else SafetyResult(True)
        if result.ok and alt < float(self.horizon.min_alt(az)):
            return SafetyResult(False, "behind the trees right now")
        return result

    def daytime(self) -> bool:
        return bool(body_altaz("sun", self.site, self.clock())[0] > DAYTIME_SUN_ALT_DEG)

    def dark(self) -> bool:
        """Dark enough for the night view's glow removal and stretch (checked every minute)."""
        t = time.monotonic()
        if t - self._dark[1] > 60:
            self._dark = (bool(body_altaz("sun", self.site, self.clock())[0] < DARK_SUN_ALT_DEG), t)
        return self._dark[0]

    def _tolerance_arcmin(self) -> float:
        return TOLERANCE_BARLOW_ARCMIN if self.barlow else TOLERANCE_ARCMIN


def _azimuth_coverage(azs: list[float]) -> float:
    """Degrees of azimuth the marks span: 360 minus the biggest gap between neighbors."""
    a = sorted(az % 360 for az in azs)
    gaps = np.diff([*a, a[0] + 360])
    return 360 - float(gaps.max())


def _km_between(lat: float, lon: float, site: Site) -> float:
    """Great-circle distance from (lat, lon) to `site`."""
    return float(np.radians(separation_deg(lat, lon, site.lat_deg, site.lon_deg)) * EARTH_RADIUS_KM)
