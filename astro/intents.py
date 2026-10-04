"""Offline intent grammar: works with no internet and no LLM.

Covers the core commands; anything else goes to the LLM agent when online.
"""

import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str  # goto, stop, next, tonight, capture, stop_capture, focus, where
    target: str | None = None


_PATTERNS = [
    ("stop_capture", r"\b(stop|end|finish) (the )?(capture|recording|pictures?)\b"),
    ("stop", r"^(stop|halt|hold|freeze|that's it)\b"),
    ("next", r"\b(next|another one|something else)\b"),
    ("tonight", r"\b(what('s| is) (good|up|out|visible)|what can i see|tonight)\b"),
    ("capture", r"\b(capture|take (a )?(picture|photo|image)s?|record)\b"),
    ("focus", r"\bfocus\b"),
    ("where", r"\b(where am i|what am i (looking at|pointing at))\b"),
    ("goto", r"\b(?:go ?to|find|show me|point (?:at|to)|take me to|look at)\s+(?:the\s+)?(.+)"),
]


def parse(text: str) -> Intent | None:
    t = text.lower().strip().rstrip(".!?")
    for name, pattern in _PATTERNS:
        m = re.search(pattern, t)
        if m:
            return Intent(name, m.group(1).strip() if name == "goto" else None)
    return None


def match_name(spoken: str, names: list[str], cutoff: float = 0.6) -> str | None:
    """Fuzzy-match a spoken target ('the ring nebula', 'm 57') to a known name."""
    s = re.sub(r"\bm\s+(\d+)", r"m\1", spoken.lower()).removeprefix("the ").strip()
    lower = {n.lower(): n for n in names}
    if s in lower:
        return lower[s]
    best = difflib.get_close_matches(s, list(lower), n=1, cutoff=cutoff)
    return lower[best[0]] if best else None
