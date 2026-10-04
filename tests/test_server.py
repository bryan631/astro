import os

os.environ["ASTRO_SIM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""  # tests never call the real API
os.environ["ASTRO_NO_LOG_FILE"] = "1"  # don't write data/logs from tests

import pytest
from fastapi.testclient import TestClient

from astro import server


def receive_until(ws, kind):
    while True:
        msg = ws.receive()
        if "text" in msg and msg["text"] and f'"type":"{kind}"' in msg["text"].replace(" ", ""):
            return msg["text"]
        if kind == "bytes" and msg.get("bytes"):
            return msg["bytes"]


def test_index_served():
    r = TestClient(server.app).get("/")
    assert r.status_code == 200 and "Hold to talk" in r.text


def test_hello_reports_no_server_speech_by_default():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "hello", "server_stt": False, "server_tts": False}


@pytest.fixture(autouse=True)
def saved_site(monkeypatch, tmp_path):
    """Pretend setup already happened, and keep any GPS writes out of the repo."""
    monkeypatch.setattr(server.site_store, "has_saved", lambda root: True)
    monkeypatch.setattr(server.site_store, "save", lambda root, site: None)
    monkeypatch.setattr(server, "load_site", lambda: server.Site(26.7, -80.1))


def test_text_command():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "go to pizza"})
        assert "I don't know pizza." in receive_until(ws, "say")


class FakeStt:
    def available(self):
        return True

    def transcribe(self, audio):
        return "go to pizza" if audio == b"AUDIO" else ""


class FakeTts:
    def available(self):
        return True

    def synthesize(self, text):
        return b"WAV:" + text.encode()


def test_server_speech_roundtrip(monkeypatch):
    monkeypatch.setattr(server, "stt", FakeStt())
    monkeypatch.setattr(server, "tts", FakeTts())
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_bytes(b"AUDIO")
        assert "go to pizza" in receive_until(ws, "heard")
        assert receive_until(ws, "bytes") == b"WAV:I don't know pizza."


def test_first_run_asks_tablet_for_gps(monkeypatch):
    monkeypatch.setattr(server.site_store, "has_saved", lambda root: False)
    saved = []
    monkeypatch.setattr(server.site_store, "save", lambda root, site: saved.append(site))
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert "allow location access" in receive_until(ws, "say")
        assert receive_until(ws, "get_location")
        ws.send_json({"type": "location", "lat": 27.0, "lon": -80.2, "alt": 4, "accuracy": 12})
        assert "accurate to about 12 meters" in receive_until(ws, "say")
    assert saved[0].lat_deg == 27.0 and saved[0].lon_deg == -80.2


def test_location_error_is_spoken():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "location_error", "message": "Permission denied."})
        assert "couldn't get the tablet's location" in receive_until(ws, "say")


def test_site_change_moves_simulated_sky(monkeypatch):
    saved = []
    monkeypatch.setattr(server.site_store, "save", lambda root, site: saved.append(site))
    session, _ = server.build_session()
    finder_cam, main_cam = session.finder.camera, session.main_camera
    main_cam.capture()  # fills the body-position cache for the old site
    session.set_location(40.0, -105.0, None, None)
    assert finder_cam.site.lat_deg == 40.0 and main_cam.site.lat_deg == 40.0
    assert main_cam._positions is None and saved[-1].lon_deg == -105.0


def test_gallery_lists_pictures(monkeypatch, tmp_path):
    (tmp_path / "a.png").write_bytes(b"png")
    monkeypatch.setattr(server, "GALLERY", tmp_path)
    assert TestClient(server.app).get("/gallery").json() == ["a.png"]


def test_real_hub_is_shared_and_broadcasts(monkeypatch):
    sim_session, _ = server.build_session()  # stands in for the real hardware session
    monkeypatch.setattr(server, "SIM", False)
    monkeypatch.setattr(server, "build_real_session", lambda: sim_session)
    monkeypatch.setattr(server, "_real_hub", None)
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.send_json({"type": "text", "text": "go to pizza"})
        assert "I don't know pizza." in receive_until(a, "say")
        assert "I don't know pizza." in receive_until(b, "say")  # both tablets hear it
    assert server._real_hub.session is sim_session


def test_failed_main_camera_rolls_back_pointing(monkeypatch, tmp_path):
    cfg = tmp_path / "devices.toml"
    cfg.write_text('[finder]\ndriver = "svbony"\nmodel = "SV905C"\n'
                   '[main]\ndriver = "svbony"\nmodel = "SV705C"\n[mount]\ndriver = "solve"\n')
    closed = []
    monkeypatch.setenv("ASTRO_DEVICES", str(cfg))
    monkeypatch.setattr(server.devices, "build_pointing",
                        lambda *a: (object(), lambda: closed.append("pointing")))

    def broken_camera(cfg):
        raise RuntimeError("SVB error 1")

    monkeypatch.setattr(server.devices, "open_camera", broken_camera)
    server.build_real_session.cache_clear()
    with pytest.raises(RuntimeError):
        server.build_real_session()
    assert closed == ["pointing"]
    server.build_real_session.cache_clear()


def test_unknown_main_driver_rejected_before_hardware():
    from astro.devices import config as devices

    with pytest.raises(ValueError, match="main camera driver"):
        devices.validate({"finder": {"driver": "svbony"}, "main": {"driver": "webcam"},
                          "mount": {"driver": "solve"}})


def test_failing_tick_does_not_end_guidance(monkeypatch):
    session, _ = server.build_session()
    calls = {"n": 0}

    def flaky_tick(t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("camera glitch")
        return [{"type": "say", "text": "still guiding"}]

    monkeypatch.setattr(session, "tick", flaky_tick)
    monkeypatch.setattr(server, "build_session", lambda: (session, None))
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert "still guiding" in receive_until(ws, "say")
    assert calls["n"] >= 2
