import pytest

from astro.wake import strip_wake


@pytest.mark.parametrize("text, want", [
    ("Astro, go to Jupiter.", "go to Jupiter."),
    ("hey astro what's good tonight", "what's good tonight"),
    ("Astra stop", "stop"),
    ("Astro.", ""),
    ("go to Astro", None),
    ("astronomy is fun", None),
    ("", None),
])
def test_strip_wake(text, want):
    assert strip_wake(text) == want
