import os

os.environ["ASTRO_SIM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""  # tests never call the real API

from fastapi.testclient import TestClient

from astro.server import app


def test_index_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "Hold to talk" in r.text


def test_websocket_answers_text():
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "go to pizza"})
        msg = ws.receive_json()
        while msg["type"] != "say":
            msg = ws.receive_json()
        assert msg["text"] == "I don't know pizza."
