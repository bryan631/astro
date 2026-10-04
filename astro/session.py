"""One observing session: ties planner, safety, mount position and guidance together.

`handle(text)` answers offline intents; `tick()` runs guidance at ~10 Hz. Both return
messages for the tablet: {"type": "say", "text": ...} and {"type": "state", ...}.
"""

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime

from astro.guidance.engine import Guide
from astro.intents import match_name, parse
from astro.planner.catalog import load_targets
from astro.planner.tonight import PLANETS, plan
from astro.pointing.coords import Site, body_altaz, radec_to_altaz
from astro.pointing.geometry import separation_deg
from astro.safety import check_target

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Session:
    def __init__(self, site: Site, position: Callable[[], tuple[float, float]],
                 clock: Clock = utcnow, developer_override: bool = False):
        """`position()` returns the scope's current true (alt, az) from the mount model."""
        self.site, self.position, self.clock = site, position, clock
        self.override = developer_override
        self.catalog = {t.name: t for t in load_targets()}
        for t in list(self.catalog.values()):
            self.catalog.setdefault(t.id, t)
        self.target: str | None = None
        self.guide: Guide | None = None
        self._suggestions: list[str] = []
        self._resolved_at = -1e9

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
        intent = parse(text)
        if intent is None:
            return [say("Sorry, I didn't catch that. Try 'what's good tonight' or 'go to Saturn'.")]
        if intent.name == "goto":
            name = match_name(intent.target or "", self.names())
            return self.goto(name) if name else [say(f"I don't know {intent.target}.")]
        if intent.name == "stop":
            self.target, self.guide = None, None
            return [say("Stopped.")]
        if intent.name == "tonight":
            return self.tonight()
        if intent.name == "next":
            if not self._suggestions:
                return [say("Ask me what's good tonight first.")]
            return self.goto(self._suggestions.pop(0))
        if intent.name == "where":
            return self.where()
        return [say(f"{intent.name.replace('_', ' ').capitalize()} isn't ready yet.")]

    def goto(self, name: str) -> list[dict]:
        alt, az = self.altaz_of(name)
        safe = check_target(alt, az, self.site, self.clock(), self.override)
        if not safe.ok:
            return [say(f"I can't go to {name}: it's {safe.reason}.")]
        self.target, self.guide = name, Guide(alt, az)
        return [say(f"Let's find {name}.")]

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
        alt, az = self.position()
        nearest = min(self.names(), key=lambda n: separation_deg(alt, az, *self.altaz_of(n)))
        d = separation_deg(alt, az, *self.altaz_of(nearest))
        if d < 1:
            return [say(f"You're on {nearest}.")]
        return [say(f"You're about {d:.0f} degrees from {nearest}.")]

    # --- guidance loop -------------------------------------------------------------------
    def tick(self, t: float) -> list[dict]:
        if self.guide is None or self.target is None:
            return []
        if t - self._resolved_at >= 1.0:  # targets drift ~15"/s; refresh once a second
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


def say(text: str) -> dict:
    return {"type": "say", "text": text}
