"""Offline intent grammar: works with no internet and no LLM.

Covers the core commands; anything else goes to the LLM agent when online.
"""

import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str  # goto, stop, next, tonight, capture, stop_capture, focus, finder_focus, sync,
    # where, barlow_on, barlow_off, location, horizon_start, horizon_mark, setup, ready, skip,
    # describe, collimate, view_finder, view_main, view_debug, view_hide,
    # video_finder, video_main, video_stop, centered
    target: str | None = None


# Order matters: specific phrases before general ones.
_PATTERNS = [
    ("video_stop", r"\b(stop|end|turn off|hide) (the )?(live )?video\b|\bvideo off\b"),
    ("video_finder", r"\b(live )?video (of |from )?(the )?finder\b|\bfinder (live )?video\b|\blive finder\b"),
    ("video_main", r"\b(live )?video (of |from )?(the )?(main|big|telescope)\b|\b(main|big) (camera )?(live )?video\b|\blive (main|video)\b"),
    ("stop", r"\b(stop|finish|end|done with)( the)? horizon( walk)?\b"),  # ends a walk
    ("horizon_start", r"\b(start|begin|do|record|calibrate)( the)? horizon( walk)?\b"),
    ("setup", r"\b(set ?up|setup)( the)?( telescope| scope)?\b|\bcalibrate( the)? (telescope|scope)\b"),
    ("ready", r"^(ready|i'?m ready|go ahead|ok(ay)?|done pointing)\b"),
    ("skip", r"^(skip|skip it|next step|not now)\b"),
    ("stop_capture", r"\b(stop|end|finish) (the )?(capture|recording|pictures?)\b"),
    ("barlow_on", r"\b(barlow (is )?(on|in)|(put|added?) (in )?the barlow)\b"),
    ("barlow_off", r"\b(barlow (is )?(off|out)|(took|take|removed?) (out )?the barlow)\b"),
    ("stop", r"^(stop|halt|hold|freeze|that's it|done)\b"),
    ("next", r"\b(next|another one|something else)\b"),
    ("collimate", r"\b(collimat\w*)\b"),
    ("finder_focus", r"\b(focus (the )?finder|finder focus)\b"),
    ("focus", r"\bfocus\b"),
    # Views, before "goto": "show me the finder" must not look for a target named finder.
    ("view_finder", r"\b(show|open|view|see|display)( me)?( the)?( live)? finder\b|\bfinder (view|camera|image)\b"),
    ("view_main", r"\b(show|open|view|see|display)( me)?( the)?( live)? (main|big|telescope) (camera|view|image)\b|\bmain (camera|view)\b"),
    ("view_debug", r"\b(show|open|view|display)( me)?( the)? (debug|diagnostics?|details?)\b|^debug\b"),
    ("view_hide", r"^(hide|close|clear)\b|\b(hide|close)( the)? (camera|cameras|debug|details?|views?|picture)\b"),
    # "Saturn is centered": the main camera is on it, so learn the finder-to-main offset.
    ("centered", r"^(?:the\s+)?(.+?)\s+(?:is|'s)\s+(?:centered|centred|in the middle)\b"),
    ("sync", r"\b(sync|align|plate ?solve|find (out )?where (we are|i am|it is) pointing)\b"),
    ("horizon_mark", r"^(mark|mark it|mark this|here)\b"),
    ("location", r"\b(set|update|use|get|find) (my |our |the )?(location|position|gps)\b"),
    ("where", r"\b(where am i|what am i (looking at|pointing at))\b"),
    # Before "tonight": "show me Saturn tonight" is a goto (but see _goto_target).
    ("goto", r"\b(?:go ?to|find|show me|point (?:at|to)|take me to|look at)\s+(?:the\s+)?(.+)"),
    ("tonight", r"\b(what('s| is) (good|up|out|visible)|what can i see|tonight)\b"),
    ("describe", r"\b(?:tell me about|what is|what's|describe)\s+(?:the\s+)?(.+)"),
    ("capture", r"\b(capture|take (a )?(picture|photo|image)s?|record)\b"),
]
_NOT_A_TARGET = ("what", "something", "anything", "good")  # "show me what's good tonight"
_FILLER = re.compile(r"(^|\s+)(tonight|now|please|right now)$")


def _goto_target(raw: str) -> str | None:
    """Clean a goto target, or None if it isn't really a target name."""
    target = raw.strip()
    while (stripped := _FILLER.sub("", target)) != target:
        target = stripped
    if not target or target.startswith(_NOT_A_TARGET):
        return None
    return target


def parse(text: str) -> Intent | None:
    t = text.lower().strip().rstrip(".!?")
    for name, pattern in _PATTERNS:
        m = re.search(pattern, t)
        if not m:
            continue
        if name in ("goto", "describe", "centered"):
            target = _goto_target(m.group(1))
            if target is None:
                continue  # e.g. "show me what's good tonight" -> tonight
            return Intent(name, target)
        return Intent(name)
    return None


def match_name(spoken: str, names: list[str], cutoff: float = 0.6) -> str | None:
    """Fuzzy-match a spoken target ('the ring nebula', 'm 57') to a known name."""
    s = re.sub(r"\bmessier\b", "m", spoken.lower())
    s = re.sub(r"\bm[\s-]*(\d+)", r"m\1", s).removeprefix("the ").strip()
    lower = {n.lower(): n for n in names}
    if s in lower:
        return lower[s]
    best = difflib.get_close_matches(s, list(lower), n=1, cutoff=cutoff)
    return lower[best[0]] if best else None
