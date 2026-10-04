"""One observing session: ties planner, safety, mount position and guidance together.

`handle(text)` answers offline intents; `tick()` runs guidance at ~10 Hz. Both return
messages for the tablet: {"type": "say", "text": ...} and {"type": "state", ...}.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from astro import calibration_store
from astro.capture.focus import FocusCoach, laplacian_variance
from astro.capture.live_stacker import LiveStacker
from astro.capture.recorder import CaptureRefused, Recorder, prune
from astro.capture.roi import brightest_blob, roi_around
from astro.devices.base import Camera
from astro.guidance.centering import CALIBRATED, Centerer
from astro.guidance.engine import CueLimiter, DirectionLearner, Guide, wrap180
from astro.intents import Intent, match_name, parse
from astro.planner.catalog import load_targets
from astro.planner.horizon import HorizonMask
from astro.planner.tonight import PLANET_NOTES, PLANETS, next_dark, plan
from astro.pointing.coords import Site, body_altaz, radec_to_altaz
from astro.pointing.finder_sync import FinderSync, check_focus
from astro.pointing.geometry import separation_deg
from astro.pointing.main_offset import MainOffset
from astro.pointing.platesolve import finder_gray
from astro.process.planet import MIN_FRAMES, StackResult, process_ser
from astro.safety import DAYTIME_SUN_ALT_DEG, SafetyResult, check_target
from astro.wizard import SetupWizard

Clock = Callable[[], datetime]
TARGET_REFRESH_S = 1.0  # targets drift ~15"/s, so re-resolve their alt/az once a second
FOCUS_STEP_S = 1.0  # one finder focus measurement per second while coaching
MIN_FOCUS_SAMPLES = 3  # focus readings before "done" counts (else the gate was never checked)
CENTER_STEP_S = 0.5  # main-camera centering cue rate
DIRECTION_PROBE_S = 1.5  # after a left/right cue, look this long for the azimuth to move
FIX_STALE_S = 2.0  # a plate-solve fix older than this is too old to steer by
ENCODER_STALE_S = 1.0  # encoder positions older than this mean the board or cable is gone
MIN_HFR_PX = 0.5  # floor so a perfectly sharp (tiny) star can't blow up the focus score
TOLERANCE_ARCMIN = {False: 4.0, True: 2.0}  # guidance tolerance without / with the 2x Barlow
RECORD_SECONDS = 60  # planetary video length
STACK_SECONDS = 90  # deep-sky live stack; the target drifts out of the field in ~2 min
FOCUS_CROP_PX = 256  # sharpness measured on a crop around the planet
MIN_HORIZON_MARKS = 3
SITE_MOVE_KM = 1.0  # moving farther than this from the model's site invalidates the mount model
CLOUD_CACHE_S = 15 * 60  # Open-Meteo is hourly; don't ask on every request
LATER_MIN = 30  # "tonight" more than this far ahead: say when it gets dark
CLOUDY_PCT = 50  # at or above this cloud cover, tonight's suggestions mention the clouds
EARTH_RADIUS_KM = 6371.0


def utcnow() -> datetime:
    return datetime.now(UTC)


class Session:
    def __init__(self, site: Site, position: Callable[[], tuple[float, float]] | None = None,
                 clock: Clock = utcnow, developer_override: bool = False,
                 finder: FinderSync | None = None, main_camera: Camera | None = None,
                 main_sensor: tuple[int, int] = (3856, 2180), data_dir: Path = Path("data"),
                 on_site_change: Callable[[Site], None] | None = None,
                 horizon: HorizonMask | None = None,
                 on_horizon_change: Callable[[HorizonMask], None] | None = None,
                 weather: Callable[[float, float, datetime], float | None] | None = None,
                 calibration: dict | None = None,
                 on_calibration_change: Callable[[dict], None] | None = None):
        """Pointing comes from `finder` (encoders + mount model + plate solving), or, for
        tests without a finder, from `position()` returning true (alt, az)."""
        self.site, self.clock, self.finder = site, clock, finder
        self.on_site_change = on_site_change  # e.g. persist the GPS fix, update simulators
        self.horizon = horizon or HorizonMask()  # treeline for the planner
        self.on_horizon_change = on_horizon_change
        self._horizon: list[tuple[float, float]] | None = None  # points during a horizon walk
        self.wizard: SetupWizard | None = None  # first-time setup at a location
        self._location_request = 0  # id of the GPS request whose answer we'd accept
        self._model_site = site  # site the current mount model was built for
        self.weather = weather  # (lat, lon, when) -> cloud % or None offline; None = no forecast
        self._clouds: float | None = None
        self._clouds_at = -1e9
        self._clouds_lock = threading.Lock()
        self._clouds_site: tuple[float, float] | None = None
        self.position = finder.position if finder else position
        if finder is not None:
            finder.safety = self.exposure_safety  # no finder exposure skips the Sun/daytime gate
            if hasattr(finder, "start"):  # a background solver may only run once gated
                finder.start()
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
        self.recorder = (Recorder(main_camera, main_sensor, data_dir / "captures",
                                  self.exposure_safety) if main_camera else None)
        self._announced_done = True
        self.gallery_dir = data_dir / "gallery"
        self.stack_seconds = STACK_SECONDS
        self.stacker = (LiveStacker(main_camera, self.gallery_dir, self.exposure_safety,
                                    preview_dir=data_dir / "live")
                        if main_camera else None)
        self._stack_done_announced = True
        self._preview_seen = 0
        self._processor = ThreadPoolExecutor(max_workers=1)  # one stacking job at a time
        self._holding = False  # asked the user to hold still for a fresh fix
        self._jobs: list[tuple[str, Future[StackResult]]] = []  # pictures being made, in order
        self._focus_at = -1e9
        self.centerer = Centerer((main_sensor[0], main_sensor[1]))  # learns finder->main offset
        self._centering = False
        self._center_at = -1e9
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
        name = match_name(spoken, self.names())
        if name is None:
            return [say(f"I don't know {spoken}.")]
        if name.lower() in PLANETS:
            kind, note = "a planet", PLANET_NOTES[name.lower()]
        elif name == "Moon":
            kind, note = "our Moon", "Craters and mountains show best along the shadow line."
        else:
            target = self.catalog[name]
            kind, note = f"a {target.category} ({target.id})", target.note
        alt, az = self.altaz_of(name)
        where = (f"Right now it's about {alt:.0f} degrees up, toward the {_compass(az)}."
                 if alt > 0 else "It's below the horizon right now.")
        if 0 < alt < float(self.horizon.min_alt(az)):
            where += " That's behind the trees from here."
        return [say(f"{name} is {kind}. {note} {where}")]

    def names(self) -> list[str]:
        return [p.capitalize() for p in PLANETS] + ["Moon", *self.catalog]

    def altaz_of(self, name: str) -> tuple[float, float]:
        if name.lower() in PLANETS or name == "Moon":
            return body_altaz(name.lower(), self.site, self.clock())
        t = self.catalog[name]
        return radec_to_altaz(t.ra, t.dec, self.site, self.clock())

    # --- commands ------------------------------------------------------------------------
    def handle(self, text: str) -> list[dict]:
        intent = parse(text)
        if intent is not None and intent.name == "tonight":
            return self.tonight()  # planning is slow and read-only: keep it off the lock
        with self._lock:
            return self._handle(text)

    def goto_spoken(self, target: str) -> list[dict]:
        """Go to a target named in free text (agent tool): matched by name, never re-parsed."""
        with self._lock:
            name = match_name(target, self.names())
            return self.goto(name) if name else [say(f"I don't know {target}.")]

    def _handle(self, text: str) -> list[dict]:
        intent = parse(text)
        if intent is not None and intent.name in ("ready", "skip") and not self.wizard_active:
            # Outside setup, "okay" is conversation, and "next step" just means "next".
            intent = Intent("next") if "next" in text.lower() else None
        if intent is not None and intent.name in ("setup", "ready", "skip"):
            return self._wizard_command(intent.name)
        if intent is None:
            return [say("Sorry, I didn't catch that. Try 'what's good tonight' or 'go to Saturn'.")]
        if intent.name == "goto":
            name = match_name(intent.target or "", self.names())
            return self.goto(name) if name else [say(f"I don't know {intent.target}.")]
        if intent.name == "stop":
            if self.wizard is not None and self.wizard.active:
                self.wizard = None
                return [say("OK, setup stopped. Say 'set up the telescope' to start again.")]
            if self._horizon is not None:  # "done" / "stop" finishes the horizon walk
                return self.finish_horizon()
            if self._focus_coach is not None:
                if self._focus_coach.samples < MIN_FOCUS_SAMPLES:
                    return [say("Keep turning slowly a little longer, so I can find the "
                                "sharpest point.")]
                if self._focus_mode == "main":
                    self.main_focus_ok = True
                self._focus_coach = None
                return [say("OK, focus is set.")]
            if self._camera_busy():  # "stop" while taking a picture ends the picture
                return self._handle("stop recording")
            self.target, self.guide, self._centering = None, None, False
            return [say("Stopped.")]
        if intent.name in ("barlow_on", "barlow_off"):
            self.barlow = intent.name == "barlow_on"
            self.main_focus_ok = False
            if self._focus_coach is not None and self._focus_mode == "main":
                self._focus_coach = FocusCoach()  # old scores don't compare across optics
            if self.guide is not None:  # an active guide switches tolerance too
                self.guide.tol_deg = TOLERANCE_ARCMIN[self.barlow] / 60
            return [say("Got it. The Barlow changes focus, so we'll refocus before taking pictures.")]
        if intent.name == "focus":
            return self.start_main_focus()
        if intent.name == "capture":
            return self.capture()
        if intent.name == "stop_capture":
            if self.stacker is not None and self.stacker.busy:
                self.stacker.stop()
                return [say("Stopping. I'll keep what's stacked so far.")]
            if self.recorder is None or not self.recorder.busy:
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
        if intent.name == "horizon_start":
            return self.start_horizon()
        if intent.name == "horizon_mark":
            return self.mark_horizon()
        if intent.name == "location":
            return self.request_location()
        if intent.name == "describe":
            return self.describe(intent.target or "")
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
        if self._camera_busy():  # one user of the camera at a time
            return [say("I'm recording right now. Say 'stop recording' first.")]
        self.guide = None  # keep the target; we're on it
        self.main_focus_ok = False  # a new focus pass must finish before capture
        self._focus_coach, self._focus_mode = FocusCoach(), "main"
        return [say("Turn the telescope's focus knob slowly. I'll tell you when it gets sharper. "
                    "Say stop when I say it's the sharpest.")]

    def capture(self) -> list[dict]:
        if self.recorder is None:
            return [say("There's no main camera connected.")]
        if self._focus_coach is not None:  # focus is still using a camera
            return [say("Let's finish focusing first. Say done when it's sharpest.")]
        if not self.main_focus_ok:  # pre-flight gate (plan Phase 1 step 8)
            return [say("Let's make sure it's sharp first."), *self.start_main_focus()]
        if self._camera_busy():
            return [say("I'm already recording." if self.recorder.busy else "I'm already stacking.")]
        name = self.target or "capture"
        try:
            if self.target is None or self.target in self._extended_targets():
                self.recorder.start(name, self.record_seconds)
                self._picture_started()
                self._announced_done = False
                return [say(f"Recording for {self.record_seconds:g} seconds. "
                            "Try not to touch the telescope.")]
            self.stacker.start(name, self.stack_seconds)  # deep-sky: live stack short subs
        except CaptureRefused as e:
            return [say(str(e))]  # nothing started: guidance carries on as before
        self._picture_started()
        self._stack_done_announced, self._preview_seen = False, 0
        return [say(f"Stacking short pictures of {name}. Watch it build up on the screen. "
                    "Try not to touch the telescope.")]

    def _picture_started(self) -> None:
        """The picture has the camera: guidance goes quiet (no "right a little" while the user
        was asked not to touch the telescope). The target is kept."""
        self._centering, self.guide = False, None

    def _camera_busy(self) -> bool:
        return any(job is not None and job.busy for job in (self.recorder, self.stacker))

    def goto(self, name: str) -> list[dict]:
        if self._camera_busy():
            return [say("I'm taking a picture. Say stop first, then we can move.")]
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
        guide = Guide(*self._aim(alt, az), tolerance_arcmin=TOLERANCE_ARCMIN[self.barlow],
                          right_is_plus_az=self.right_is_plus_az)
        self._centering = False
        self.target, self.guide, self._focus_coach = name, guide, None
        return [*pre, say(f"Let's find {name}.")]

    def clouds(self, site: Site | None = None) -> float | None:
        """Cloud cover now (%) at `site`, from Open-Meteo; None offline. Cached per place; its
        own lock (not the session's) so a slow fetch never stalls guidance."""
        site = site or self.site
        here = (site.lat_deg, site.lon_deg)
        with self._clouds_lock:
            stale = time.monotonic() - self._clouds_at > CLOUD_CACHE_S or here != self._clouds_site
            if self.weather is not None and stale:  # a GPS move refetches
                self._clouds = self.weather(*here, self.clock())
                self._clouds_at, self._clouds_site = time.monotonic(), here
            return self._clouds

    def _plan(self) -> tuple[dict, datetime | None, float | None]:
        """Tonight's choices from the next dark time, in local time (asked at 4 PM, this plans
        the coming night). Runs outside the session lock: planning takes ~0.5 s of astropy."""
        with self._lock:  # one consistent snapshot; the slow work below runs unlocked
            site, horizon = self.site, self.horizon
        now = self.clock().astimezone(site.timezone)
        start = next_dark(site, now)
        clouds = self.clouds(site)
        if start is None:
            return {}, None, clouds
        return plan(site, start, mask=horizon, cloud_cover=clouds), start, clouds

    def tonight(self) -> list[dict]:
        choices, start, clouds = self._plan()
        flat = sorted((c for cs in choices.values() for c in cs), key=lambda c: -c.score)
        if not flat:
            return [say("Nothing good is up right now.")]
        with self._lock:
            self._suggestions = [c.name for c in flat[1:6]]
        best = flat[0]
        others = ", ".join(c.name for c in flat[1:3])
        when = ""
        if start is not None and start - self.clock() > timedelta(minutes=LATER_MIN):
            when = f"It's still light out. Once it's dark, around {_clock(start)}: "
        sky = ""
        if clouds is not None and clouds >= CLOUDY_PCT:
            sky = f"It looks about {clouds:.0f} percent cloudy, so it may come and go. "
        return [say(f"{when}{sky}{best.name} is the best. {best.note} "
                    f"Other good ones: {others}. Say 'go to' a name, or 'next'.")]

    def tonight_by_category(self) -> str:
        """Compact text for the agent: best target per category, local times."""
        choices, _, clouds = self._plan()
        lines = [f"{cat}: {cs[0].name} (best around {_clock(cs[0].best_time)}). {cs[0].note}"
                 for cat, cs in choices.items()]
        if clouds is not None:
            lines.append(f"cloud cover: about {clouds:.0f}%")
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
            if rec.frames < MIN_FRAMES:
                return [say(f"{rec.error or 'Done.'} I only got {rec.frames} frames, "
                            "not enough for a picture.")]
            job = self._processor.submit(process_ser, rec.path, self.gallery_dir)
            self._jobs.append((rec.name, job))
            done = rec.error or "Done."  # e.g. drifted out of view: still make the picture
            return [say(f"{done} I saved {rec.frames} frames. I'm making your picture now.")]
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
        if self._focus_coach is not None:
            if self._focus_mode == "main":
                return self._main_focus_step(t)
            return self._finder_focus_step(t)
        if (self._centering or self.guide is not None) and (lost := self._pointing_lost()):
            self.guide, self._centering = None, False  # keep the target: "go to" it again later
            return [say(lost)]
        if self._centering:
            return self._center_step(t)
        if self.guide is None or self.target is None:
            self._direction_probe = None  # don't classify unrelated motion later
            return []
        if t - self._resolved_at >= TARGET_REFRESH_S:
            self._resolved_at = t
            alt, az = self.altaz_of(self.target)
            if not self.target_safety(alt, az).ok:
                self.target, self.guide = None, None
                return [say("Stopping: the target is no longer safe to point at.")]
            self.guide.target = self._aim(alt, az)
        fix_age = getattr(self.finder, "fix_age", None)
        if fix_age is not None and fix_age() > FIX_STALE_S:  # solving only, scope moving
            if self._holding:
                return []
            self._holding = True
            return [say("Hold still for a second so I can see where we are.")]
        self._holding = False
        alt_now, az_now = self.position()
        learned = self._learn_direction(az_now, t)  # before the cue: a flip must apply to it
        state, cue = self.guide.update(alt_now, az_now, t)
        out = [{"type": "state", "target": self.target, "right_is_plus_az": self.right_is_plus_az,
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

    def _pointing_lost(self) -> str | None:
        """Why guidance can't trust the pointing any more, or None."""
        if self.finder is None:
            return None
        age = getattr(self.finder, "encoder_age", None)
        if age is not None and age() > ENCODER_STALE_S:
            return "I lost the telescope's position sensors, so I stopped guiding. Check the cable."
        if not self.finder.synced:  # e.g. the encoder board restarted and was reset
            return "I lost track of where the telescope points. Say 'sync' and let me look again."
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
        }
        model = getattr(self.finder, "model", None)
        if model is not None and self.finder.synced:
            data["mount"] = {"site": [self.site.lat_deg, self.site.lon_deg],
                             **calibration_store.model_to_dict(model)}
        return data

    def _save_calibration(self) -> None:
        if self.on_calibration_change:
            self.on_calibration_change(self.calibration())

    def _restore_calibration(self, data: dict) -> None:
        if data.get("right_is_plus_az") is not None:
            self.right_is_plus_az, self._direction_known = data["right_is_plus_az"], True
        if data.get("camera_axes") is not None:
            self.centerer.axes.matrix = np.array(data["camera_axes"])
        if data.get("main_offset"):
            o = data["main_offset"]
            self.centerer.offset = MainOffset(o["d_az_sky_deg"], o["d_alt_deg"], o["observations"])
        mount = data.get("mount")
        if mount and self.finder is not None and hasattr(self.finder, "model"):
            boots = getattr(self.finder, "encoder_boots", None)
            if boots is not None and boots() > 0:
                return  # the encoder board booted: its counts are 0, the saved model is stale
            lat, lon = mount["site"]
            km = np.radians(separation_deg(lat, lon, self.site.lat_deg, self.site.lon_deg)) \
                * EARTH_RADIUS_KM
            if km <= SITE_MOVE_KM:  # same place, same encoder counts: no re-sync needed
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

    def _aim(self, alt: float, az: float) -> tuple[float, float]:
        """Where the finder model should point so the target lands in the main camera."""
        if self.centerer.offset.observations:
            return self.centerer.offset.correct(alt, az)
        return alt, az

    def _should_center(self) -> bool:
        return (self.main_camera is not None and self.target in self._extended_targets()
                and not self._camera_busy())

    def _center_step(self, t: float) -> list[dict]:
        if t - self._center_at < CENTER_STEP_S:
            return []
        self._center_at = t
        if self._camera_busy():  # a recording or stack took the camera
            self._centering = False
            return []
        if reason := self.exposure_safety():
            self._centering = False
            return [say(f"I stopped centering: {reason}.")]
        try:
            frame = self.main_camera.capture()
        except (RuntimeError, OSError) as e:
            self._centering = False
            return [say(f"The main camera stopped responding. ({e})")]
        step = self.centerer.update(self.position(), brightest_blob(frame))
        if step.lost:  # back to finder guidance, as the words promise
            self._centering = False
            alt, az = self.altaz_of(self.target)
            self.guide = Guide(*self._aim(alt, az), tolerance_arcmin=TOLERANCE_ARCMIN[self.barlow],
                          right_is_plus_az=self.right_is_plus_az)
        elif step.done:
            self._centering = False
            self._save_calibration()  # camera axes and finder-to-main offset are learned now
        urgent = step.done or step.lost or step.say == CALIBRATED
        spoken = self._center_limiter.speak(step.say, t, urgent) if step.say else None
        return [say(spoken)] if spoken else []


    def _finder_focus_step(self, t: float) -> list[dict]:
        if t - self._focus_at < FOCUS_STEP_S or self.finder is None or self._focus_coach is None:
            return []
        self._focus_at = t
        if stop := self._stop_focus_if_unsafe():
            return stop
        try:
            report = self.finder.focus_report()
        except (RuntimeError, OSError) as e:
            return self._camera_failed("finder", e)
        if report.stars == 0:
            return [say("I can't see any stars yet.")]
        # Fewer visible stars also means softer focus, so fold the count into the score.
        cue = self._focus_coach.update(report.stars / max(report.hfr_px, MIN_HFR_PX))
        return [say(cue)] if cue else []


    def _main_focus_step(self, t: float) -> list[dict]:
        if t - self._focus_at < FOCUS_STEP_S or self.main_camera is None or self._focus_coach is None:
            return []
        self._focus_at = t
        if stop := self._stop_focus_if_unsafe():
            return stop
        try:
            frame = self.main_camera.capture()
        except (RuntimeError, OSError) as e:
            return self._camera_failed("main", e)
        if self.target is None or self.target in self._extended_targets():
            center = brightest_blob(frame)
            if center is None:
                return [say("I don't see anything bright in the main camera.")]
            h, w = frame.shape
            r = roi_around(center, FOCUS_CROP_PX, (w, h))
            score = laplacian_variance(frame[r.y:r.y + r.height, r.x:r.x + r.width])
        else:  # stars: smaller is sharper
            report = check_focus(finder_gray(frame))  # expects hot-pixel-cleaned, binned gray
            if report.stars == 0:
                return [say("I don't see any stars in the main camera.")]
            score = 1 / max(report.hfr_px, MIN_HFR_PX)
        cue = self._focus_coach.update(score)
        return [say(cue)] if cue else []

    def _announce_picture(self) -> list[dict]:
        name, job = self._jobs.pop(0)
        try:
            result = job.result()
        except (ValueError, OSError) as e:
            return [say(f"I couldn't make the picture of {name}: {e}")]
        if not self._jobs:  # nothing queued still needs its raw video
            prune(self.recorder.out_dir)  # keep only the newest raw videos
        return [say(f"Your picture of {name} is ready. Tap Pictures to see it."),
                {"type": "picture", "file": result.path.name}]

    def _announce_stack(self, live) -> list[dict]:
        if not live.frames:
            return [say(live.error or f"I couldn't stack any pictures of {live.name}.")]
        why = f"{live.error} " if live.error else ""
        return [say(f"{why}Your picture of {live.name} is ready, from {live.frames} short "
                    "pictures. Tap Pictures to see it."),
                {"type": "picture", "file": live.picture.name}]

    def _camera_failed(self, which: str, error: Exception) -> list[dict]:
        """A camera failed even after the driver's retry: stop focusing and say so."""
        self._focus_coach = None
        return [say(f"The {which} camera stopped responding, so I stopped focusing. ({error})")]

    # --- horizon walk (calibration wizard) ------------------------------------------------
    def start_horizon(self) -> list[dict]:
        if self.finder is not None and not self.finder.synced:
            ok, msg = self.finder.sync()  # marks are only as good as the pointing
            if not ok:
                return [say(f"Before the horizon walk, I need to see the stars. {msg}")]
        self._horizon = []
        self.target, self.guide, self._centering = None, None, False
        self._focus_coach = None  # one mode at a time: focus prompts would talk over the walk
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
        return [say(f"Saved the treeline from {len(points)} marks. "
                    "I'll only suggest things above it.")]

    @property
    def wizard_active(self) -> bool:
        return self.wizard is not None and self.wizard.active

    def _wizard_command(self, name: str) -> list[dict]:
        if name == "setup":
            if self.finder is None:
                return [say("There's no finder camera, so I can't run setup.")]
            self.wizard = SetupWizard(self.request_location, self.finder.sync,
                                      self.finder.alignment, self.start_horizon,
                                      self.cancel_location)
            self.target, self.guide, self._centering, self._focus_coach = None, None, False, None
            return self.wizard.start()
        return self.wizard.ready() if name == "ready" else self.wizard.skip()

    def status_text(self) -> str:
        """Short facts for the agent to summarize (not spoken verbatim); a locked snapshot."""
        with self._lock:
            return self._status_facts()

    def _status_facts(self) -> str:
        facts = []
        if self.finder is not None:
            facts.append("aligned with the stars" if self.finder.synced else "not aligned yet")
        facts.append(f"target: {self.target}" if self.target else "no target")
        facts.append("Barlow in" if self.barlow else "no Barlow")
        if self.main_camera is not None:
            facts.append("focus checked" if self.main_focus_ok else "focus not checked yet")
        if self._camera_busy():
            facts.append("a picture is being taken")
        pictures = list(self.gallery_dir.glob("*.png")) if self.gallery_dir.exists() else []
        facts.append(f"{len(pictures)} pictures in the gallery")
        facts.append("treeline recorded" if self.horizon.points != HorizonMask().points
                     else "treeline not recorded (default 20 degrees)")
        clouds = self._clouds  # cached only: never fetch the forecast under the lock
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
        new = replace(self.site, lat_deg=lat, lon_deg=lon,
                      elevation_m=self.site.elevation_m if elevation_m is None else elevation_m)
        # Great-circle distance from where the mount model was built (separation_deg works on
        # any lat/lon pair), so a chain of small updates can't drift away without a reset.
        ref = self._model_site
        moved_km = np.radians(separation_deg(ref.lat_deg, ref.lon_deg, lat, lon)) * EARTH_RADIUS_KM
        self.site = new
        if moved_km > SITE_MOVE_KM:  # a new place: its pointing, treeline and plans don't apply
            self._model_site = new
            if self.finder is not None:
                self.finder.reset(new)
            self.target, self.guide, self._centering, self._focus_coach = None, None, False, None
            self._horizon, self._suggestions = None, []
            self.horizon = HorizonMask()
            if self.on_horizon_change:
                self.on_horizon_change(self.horizon)
        if self.on_site_change:
            self.on_site_change(new)
        near = f", accurate to about {accuracy_m:.0f} meters" if accuracy_m else ""
        out = [say(f"Got it, I know where we are{near}.")]
        if self.wizard_active:  # setup was waiting for this before the first sync
            out += self.wizard.location_done()
        return out

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
        if result.ok and alt < float(self.horizon.min_alt(az)):
            return SafetyResult(False, "behind the trees right now")
        return result

    def exposure_safety(self) -> str | None:
        """Spoken reason why taking an exposure now is unsafe, else None (astro/safety.py)."""
        when = self.clock()
        if self.finder is not None and not self.finder.synced:
            # Pointing unknown: we can't rule out the Sun, so allow only when it is down.
            if body_altaz("sun", self.site, when)[0] > DAYTIME_SUN_ALT_DEG:
                return "it's daytime and I don't know where the telescope is pointing"
            return None
        result = check_target(*self.position(), self.site, when, self.override)
        return None if result.ok else f"the telescope is pointing {result.reason}"

    def _stop_focus_if_unsafe(self) -> list[dict]:
        if reason := self.exposure_safety():
            self._focus_coach = None
            return [say(f"I stopped focusing: {reason}.")]
        return []

    def _extended_targets(self) -> set[str]:
        return {p.capitalize() for p in PLANETS} | {"Moon"}


def _compass(az: float) -> str:
    """Azimuth in degrees -> 'northeast' etc. (8 points)."""
    points = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    return points[round(az % 360 / 45) % 8]


def _clock(t: datetime) -> str:
    """Local wall-clock time as spoken: '9:15 PM'."""
    return t.strftime("%I:%M %p").lstrip("0")


def say(text: str) -> dict:
    return {"type": "say", "text": text}
