import pytest

from astro.intents import Intent, match_name, parse


@pytest.mark.parametrize("text,expected", [
    ("Stop!", Intent("stop")),
    ("What's good tonight?", Intent("tonight")),
    ("go to Saturn", Intent("goto", "saturn")),
    ("show me the ring nebula", Intent("goto", "ring nebula")),
    ("take a picture", Intent("capture")),
    ("stop the capture", Intent("stop_capture")),
    ("help me focus", Intent("focus")),
    ("focus the finder", Intent("finder_focus")),
    ("sync", Intent("sync")),
    ("find out where we are pointing", Intent("sync")),
    ("what am I looking at", Intent("where")),
    ("next", Intent("next")),
    ("tell me a joke", None),
])
def test_parse(text, expected):
    assert parse(text) == expected


NAMES = ["Saturn", "Ring Nebula", "M57", "Albireo", "Moon"]


@pytest.mark.parametrize("spoken,expected", [
    ("saturn", "Saturn"), ("the moon", "Moon"), ("m 57", "M57"),
    ("ring nebular", "Ring Nebula"), ("albireo", "Albireo"), ("pizza", None),
])
def test_match_name(spoken, expected):
    assert match_name(spoken, NAMES) == expected
