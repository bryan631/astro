import pytest


@pytest.fixture(autouse=True)
def _ignore_disk_space(monkeypatch, request):
    """The recorder refuses to start below 2 GB free; tests shouldn't depend on this machine's
    disk. The disk-space test sets its own conditions."""
    if "full_disk" not in request.node.name:
        monkeypatch.setattr("astro.capture.recorder.MIN_FREE_BYTES", 0)


@pytest.fixture(autouse=True)
def _no_gaia_download(monkeypatch):
    """Align fetches the field's Gaia stars from VizieR; tests stay offline."""
    monkeypatch.setattr("astro.session.field_catalog", lambda *a, **k: None)
