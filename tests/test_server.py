import os

os.environ["ASTRO_SIM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""  # tests never call the real API

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
