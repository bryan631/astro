"""Setup wizard: a voice-led first-time setup at a new location (plan Phase 3 step 3).

location (tablet GPS) -> first sync -> two more syncs elsewhere (refines the mount's tilt)
-> optional horizon walk. Each step waits for "ready"; "skip" moves on; "stop" ends it.
"""

from collections.abc import Callable

SYNC_STEPS = 3
GOOD_ALIGNMENT_ARCMIN = 10.0

Reply = list[dict]


def say(text: str) -> dict:
    return {"type": "say", "text": text}


class SetupWizard:
    def __init__(self, request_location: Callable[[], Reply], sync: Callable[[], tuple[bool, str]],
                 alignment: Callable[[], tuple[int, float | None]],
                 start_horizon: Callable[[], Reply]):
        self.request_location, self.sync = request_location, sync
        self.alignment, self.start_horizon = alignment, start_horizon
        self.syncs_done = 0
        self.step = "location"
        self.active = True

    def start(self) -> Reply:
        """Ask for the location right away (the tablet answers by itself), then the first sync."""
        self.step = "sync"
        return [say("Let's set up the telescope here. First, the location."),
                *self.request_location(), say(self._sync_prompt())]

    def ready(self) -> Reply:
        if self.step == "sync":
            ok, msg = self.sync()
            if not ok:
                return [say(f"{msg} Say ready to try again, or skip.")]
            self.syncs_done += 1
            return [say(self._alignment_report()), *self._next_after_sync()]
        if self.step == "horizon":
            self.active = False
            return self.start_horizon()
        return []

    def skip(self) -> Reply:
        if self.step == "sync":
            self.step = "horizon"
            return [say(self._horizon_prompt())]
        self.active = False
        return [say("Setup is complete. Say 'what's good tonight' to begin.")]

    def _next_after_sync(self) -> Reply:
        if self.syncs_done < SYNC_STEPS:
            return [say(self._sync_prompt())]
        self.step = "horizon"
        return [say(self._horizon_prompt())]

    def _sync_prompt(self) -> str:
        if self.syncs_done == 0:
            return ("Point the telescope high in the sky, away from trees and lights, "
                    "and say ready.")
        return (f"Now push to a different part of the sky, at least a quarter turn around, "
                f"and say ready. ({self.syncs_done} of {SYNC_STEPS} done.)")

    def _alignment_report(self) -> str:
        n, rms = self.alignment()
        if rms is None or n < 3:
            return "Got it."
        quality = "good" if rms <= GOOD_ALIGNMENT_ARCMIN else "still a bit rough"
        return f"Got it. The alignment is {quality}, about {rms:.0f} arcminutes."

    def _horizon_prompt(self) -> str:
        return ("Last step: record the treeline so I only suggest things above it. "
                "Say ready to start, or skip.")
