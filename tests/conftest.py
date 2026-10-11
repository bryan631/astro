import pytest


@pytest.fixture(autouse=True)
def _ignore_disk_space(monkeypatch, request):
    """The recorder refuses to start below 2 GB free; tests shouldn't depend on this machine's
    disk. The disk-space test sets its own conditions."""
    if "full_disk" not in request.node.name:
        monkeypatch.setattr("astro.capture.recorder.MIN_FREE_BYTES", 0)


@pytest.fixture(autouse=True)
def _no_gaia_download(monkeypatch, request):
    """Align fetches the field's Gaia stars from ESA or VizieR; tests stay offline (a cached
    tile is still read). The download test fakes the network itself."""
    if "fetched" not in request.node.name:
        monkeypatch.setattr("astro.pointing.main_solve._download", lambda *a, **k: None)
