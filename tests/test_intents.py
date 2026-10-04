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


@pytest.mark.parametrize("text,expected", [
    ("stop the horizon walk", Intent("stop")),
    ("finish the horizon walk", Intent("stop")),
    ("I'm done with the horizon walk", Intent("stop")),
    ("start the horizon walk", Intent("horizon_start")),
    ("calibrate the horizon", Intent("horizon_start")),
    ("calibrate the telescope", Intent("setup")),
    ("finder focus", Intent("finder_focus")),
    ("show me Saturn tonight", Intent("goto", "saturn")),
    ("take me to the Moon tonight", Intent("goto", "moon")),
    ("show me what's good tonight", Intent("tonight")),
    ("what's good to look at tonight", Intent("tonight")),
])
def test_review_misparses(text, expected):
    """Review H4: these used to restart the walk, start setup, or miss the target."""
    assert parse(text) == expected


@pytest.mark.parametrize("spoken", ["messier 57", "Messier 57", "M-57", "m 57"])
def test_messier_spellings(spoken):
    assert match_name(spoken, ["M57", "Saturn"]) == "M57"
