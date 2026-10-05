"""Messages for the tablet."""


def say(text: str) -> dict:
    """Spoken (and shown) text."""
    return {"type": "say", "text": text}
