import pytest


@pytest.fixture(autouse=True)
def _ignore_disk_space(monkeypatch, request):
    """The recorder refuses to start below 2 GB free; tests shouldn't depend on this machine's
    disk. The disk-space test sets its own conditions."""
    if "full_disk" not in request.node.name:
        monkeypatch.setattr("astro.capture.recorder.MIN_FREE_BYTES", 0)
