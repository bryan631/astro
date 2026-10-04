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
