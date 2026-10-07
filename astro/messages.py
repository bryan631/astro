"""Messages for the tablet."""


def say(text: str) -> dict:
    """Spoken (and shown) text."""
    return {"type": "say", "text": text}


def notice(text: str) -> dict:
    """Shown, not spoken: the answer to a button press."""
    return {"type": "notice", "text": text}
