"""Wake word for hands-free mode: "Astro, go to Jupiter" or "Astro" then the command."""
import re

# Whisper often hears "Astro" as "Astra" or "Austro".
_WAKE = re.compile(r"^\W*(?:hey\W+)?(?:astro|astra|austro)\b[\s,.!?:;-]*", re.IGNORECASE)


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_INSTRUCTS = re.compile(r"\b(say|try)\b", re.IGNORECASE)
_QUOTED = re.compile(r"(?<!\w)'(.+?)'(?!\w)")  # 'go to Saturn', but not the apostrophe in "what's"


def add_wake(text: str) -> str:
    """Hands-free: spoken instructions name the wake word. "Say 'next'." -> "Say 'Astro, next'."."""
    def sentence(s: str) -> str:
        if not _INSTRUCTS.search(s):
            return s
        if _QUOTED.search(s):
            return _QUOTED.sub(r"'Astro, \1'", s)
        return _INSTRUCTS.sub(lambda m: f"{m.group(0)} Astro,", s, count=1)

    return " ".join(sentence(s) for s in _SENTENCE.split(text))


def strip_wake(text: str) -> str | None:
    """The command after the wake word (maybe empty), or None if there is no wake word."""
    m = _WAKE.match(text)
    return text[m.end():].strip() if m else None
