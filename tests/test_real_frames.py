"""Every recorded real finder frame must still produce its expected outcome.

Add cases with scripts/hwcheck/capture_cases.py; this test picks them up from the manifest.
"""

import os

import pytest

if not os.environ.get("ASTRO_REQUIRE_SOLVER"):
    pytest.importorskip("tetra3", reason="run scripts/install-solver.sh")

from astro.pointing.platesolve import FinderSolver
from astro.pointing.sky_test import OUTCOMES, classify
from tests import frames

CASES = frames.manifest()


@pytest.fixture(scope="module")
def solver():
    return FinderSolver()


def test_manifest_is_valid():
    for case in CASES:
        assert case["expect"] in OUTCOMES, case
        assert (frames.FINDER / case["file"]).exists(), case


@pytest.mark.parametrize("case", CASES, ids=[c["file"] for c in CASES])
def test_real_frame_outcome(case, solver):
    assert classify(frames.load(case["file"][:-4]), solver) == case["expect"], case["notes"]


class StubSolver:
    def __init__(self, solves):
        self.solves = solves

    def solve(self, raw):
        return object() if self.solves else None


@pytest.mark.parametrize("focus_outcome,solves,expected", [
    ("ok", True, "solves"),
    ("out_of_focus", True, "solves"),  # a solve wins over a failed focus check
    ("ok", False, "no_match"),
    ("no_stars", False, "no_stars"),
    ("not_sky", False, "not_sky"),
    ("few_stars", False, "few_stars"),
    ("out_of_focus", False, "out_of_focus"),
])
def test_classify_outcome_contract(monkeypatch, focus_outcome, solves, expected):
    import numpy as np

    from astro.pointing import sky_test
    from astro.pointing.finder_sync import FocusReport

    report = FocusReport(5, 1.0, focus_outcome == "ok", "", focus_outcome)
    monkeypatch.setattr(sky_test, "check_focus", lambda gray: report)
    assert classify(np.zeros((8, 8), np.uint8), StubSolver(solves)) == expected
