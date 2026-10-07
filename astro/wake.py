"""Wake word for hands-free mode: "Astro, go to Jupiter" or "Astro" then the command."""
import re

# Whisper often hears "Astro" as "Astra" or "Austro".
_WAKE = re.compile(r"^\W*(?:hey\W+)?(?:astro|astra|austro)\b[\s,.!?:;-]*", re.IGNORECASE)


def strip_wake(text: str) -> str | None:
    """The command after the wake word (maybe empty), or None if there is no wake word."""
    m = _WAKE.match(text)
    return text[m.end():].strip() if m else None
