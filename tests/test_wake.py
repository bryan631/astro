import pytest

from astro.wake import add_wake, collapse_repeats, strip_wake


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


@pytest.mark.parametrize("text, want", [
    ("Say 'go to' and a name, or say 'next'.", "Say 'Astro, go to' and a name, or say 'Astro, next'."),
    ("Sorry. Say 'what's good tonight' or 'go to Saturn'.",
     "Sorry. Say 'Astro, what's good tonight' or 'Astro, go to Saturn'."),
    ("Say ready to try again, or skip.", "Say Astro, ready to try again, or skip."),
    ("Say stop when I say it's the sharpest.", "Say Astro, stop when I say it's the sharpest."),
    ("Three targets. I'm ready.", "Three targets. I'm ready."),
])
def test_add_wake(text, want):
    assert add_wake(text) == want


@pytest.mark.parametrize("text, want", [
    ("Astro, what's good tonight? Astro, what's good tonight?", "Astro, what's good tonight?"),
    ("what's good tonight? Astro, what's good tonight?", "what's good tonight?"),
    ("Astro. Astro.", "Astro."),
    ("Go to Saturn. Go to Saturn", "Go to Saturn."),
    ("Next. Stop.", "Next. Stop."),
])
def test_collapse_repeats(text, want):
    assert collapse_repeats(text).rstrip(".?") == want.rstrip(".?")
