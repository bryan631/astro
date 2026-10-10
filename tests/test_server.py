import os

os.environ["ASTRO_SIM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""  # tests never call the real API
os.environ["ASTRO_NO_LOG_FILE"] = "1"  # don't write data/logs from tests
os.environ["ASTRO_OFFLINE"] = "1"  # no forecast fetches from tests

import asyncio
import os
import time as _time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from astro import server
from astro.devices import config as devices
from astro.server import Hub


def receive_until(ws, kind):
    while True:
        msg = ws.receive()
        if "text" in msg and msg["text"] and f'"type":"{kind}"' in msg["text"].replace(" ", ""):
            return msg["text"]
        if kind == "bytes" and msg.get("bytes"):
            return msg["bytes"]


def test_index_served():
    r = TestClient(server.app).get("/")
    assert r.status_code == 200 and "Capture" in r.text and "Go to" in r.text
    assert "Tap to talk" not in r.text  # volume off: no voice on the page (Phase 3b)


def test_hello():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "hello"}


@pytest.fixture(autouse=True)
def saved_site(monkeypatch, tmp_path):
    """Pretend setup already happened, and keep any GPS writes out of the repo."""
    monkeypatch.setattr(server.site_store, "has_saved", lambda root: True)
    monkeypatch.setattr(server.site_store, "save", lambda root, site: None)
    monkeypatch.setattr(server, "load_site", lambda: server.Site(26.7, -80.1))
    monkeypatch.setattr(server.spot_store, "load", lambda root: ([], None))
    monkeypatch.setattr(server.spot_store, "save", lambda root, spots, current: None)


def test_text_command():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "go to pizza"})
        assert "I don't know pizza." in receive_until(ws, "say")


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


def test_unplugged_main_camera_runs_without_it(monkeypatch, tmp_path):
    cfg = tmp_path / "devices.toml"
    cfg.write_text('[finder]\ndriver = "svbony"\nmodel = "SV905C"\n'
                   '[main]\ndriver = "svbony"\nmodel = "SV705C"\n[mount]\ndriver = "solve"\n')
    closed = []
    monkeypatch.setenv("ASTRO_DEVICES", str(cfg))
    monkeypatch.setattr(server.devices, "build_pointing",
                        lambda *a: (None, lambda: closed.append("pointing")))

    def broken_camera(cfg):
        raise RuntimeError("SVB error 1")

    monkeypatch.setattr(server.devices, "open_camera", broken_camera)
    server.build_real_session.cache_clear()
    session = server.build_real_session()
    assert session.main_camera is None and not closed  # finder, voice and guidance still run
    server.close_hardware()
    server.build_real_session.cache_clear()


def test_unknown_main_driver_rejected_before_hardware():
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
        assert "something went wrong" in receive_until(ws, "say")  # spoken once
        assert "still guiding" in receive_until(ws, "say")  # and guidance carries on
    assert calls["n"] >= 2


def test_shutdown_closes_the_hardware(monkeypatch):
    closed = []
    monkeypatch.setattr(server, "_hardware_closers", [lambda: closed.append("finder"),
                                                       lambda: closed.append("main")])
    with TestClient(server.app):
        pass  # startup, then shutdown
    assert closed == ["main", "finder"]  # newest first


def test_failed_session_build_rolls_back_everything(monkeypatch, tmp_path):
    cfg = tmp_path / "devices.toml"
    cfg.write_text('[finder]\ndriver = "svbony"\nmodel = "SV905C"\n'
                   '[main]\ndriver = "none"\n[mount]\ndriver = "solve"\n')
    closed = []
    monkeypatch.setenv("ASTRO_DEVICES", str(cfg))
    monkeypatch.setattr(server.devices, "build_pointing",
                        lambda *a: (object(), lambda: closed.append("pointing")))

    def corrupt(root):
        raise ValueError("bad data/horizon.toml")

    monkeypatch.setattr(server.horizon_store, "load", corrupt)
    server.build_real_session.cache_clear()
    with pytest.raises(ValueError):
        server.build_real_session()
    assert closed == ["pointing"] and server._hardware_closers == []
    server.build_real_session.cache_clear()


def test_slow_answer_says_let_me_think(monkeypatch):
    monkeypatch.setattr(server, "THINKING_AFTER_S", 0.1)
    session, _ = server.build_session()
    monkeypatch.setattr(server, "build_session", lambda: (session, None))

    def slow(self, text):
        _time.sleep(0.4)
        return [{"type": "say", "text": "here you go"}]

    monkeypatch.setattr(server.Agent, "handle", slow)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "tell me something"})
        assert "Let me think." in receive_until(ws, "say")
        assert "here you go" in receive_until(ws, "say")


def test_bad_message_is_spoken_and_keeps_the_connection():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        assert "something went wrong" in receive_until(ws, "say")
        ws.send_json({"type": "text", "text": "go to pizza"})
        assert "I don't know pizza." in receive_until(ws, "say")  # still connected


def test_pages_revalidate():
    assert TestClient(server.app).get("/").headers["cache-control"] == "no-cache"


def test_service_worker_is_stamped_with_the_build():
    r = TestClient(server.app).get("/sw.js")
    assert r.text.startswith('const BUILD = "') and "skipWaiting" in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_client_log_reaches_the_server_log(caplog):
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "client_log", "text": "The browser blocked speech"})
        ws.send_json({"type": "text", "text": "stop"})  # a later reply proves it was handled
        receive_until(ws, "say")
    assert any(r.getMessage() == "client" for r in caplog.records)


def test_camera_views_are_always_live_and_never_resend_a_frame():
    client = TestClient(server.app)
    with client.websocket_connect("/ws"):  # starts the simulated hub
        r = client.get("/api/camera/finder.jpg")  # waits for the stream's first frame
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
        seq = int(r.headers["x-seq"])
        assert seq > 0 and "x-frame-age" in r.headers
        again = client.get(f"/api/camera/finder.jpg?after={seq}")  # the next frame, or 204
        assert again.status_code in (200, 204) and int(again.headers["x-seq"]) >= seq
        assert client.get("/api/camera/main.jpg").headers.get("x-focus") is not None  # night sim sky
        assert client.get("/api/camera/other.jpg").status_code == 404
        info = client.get("/api/debug").json()
        assert {"time", "site", "pointing", "finder_camera", "guidance", "system"} <= info.keys()
        assert info["finder_camera"]["frame_px"] is not None


def test_buttons_are_actions():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "action", "do": "goto", "target": "pizza"})
        assert "I don't know pizza." in receive_until(ws, "say")
        ws.send_json({"type": "action", "do": "focus"})
        assert "focus number" in receive_until(ws, "say")
        ws.send_json({"type": "action", "do": "frobnicate"})
        assert "isn't ready yet" in receive_until(ws, "notice")


def test_target_list_for_go_to():
    client = TestClient(server.app)
    with client.websocket_connect("/ws"):
        targets = client.get("/api/targets").json()
    assert targets and {"name", "category", "alt", "az", "up", "note"} <= targets[0].keys()

def test_hardware_startup_failure_is_reported(monkeypatch):
    def broken():
        raise RuntimeError("SV905C not found")

    monkeypatch.setattr(server, "get_hub", broken)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert "isn't ready: SV905C not found" in receive_until(ws, "unavailable")


def test_state_updates_coalesce_while_events_keep_order():
    async def run():
        hub = Hub.__new__(Hub)
        hub._outbox, hub._state = asyncio.Queue(), None
        for i in range(50):
            await hub.broadcast({"type": "state", "n": i})
        await hub.broadcast({"type": "say", "text": "hi"})
        await hub.broadcast({"type": "state", "n": 99})
        return hub._outbox.qsize(), hub._state
    size, state = asyncio.run(run())
    assert size == 2 and state == {"type": "state", "n": 99}


def test_sim_tablets_share_one_session(monkeypatch):
    monkeypatch.setattr(server, "_sim_hub", None)
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.receive_json(), b.receive_json()
        assert len(server._sim_hub.clients) == 2


def test_sim_world_resets_after_the_last_tablet_leaves(monkeypatch):
    monkeypatch.setattr(server, "_sim_hub", None)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.receive_json()
        first = server._sim_hub
    assert server._sim_hub is None and first is not None


def test_token_gates_pages_and_socket(monkeypatch):
    monkeypatch.setenv("ASTRO_TOKEN", "s3cret")
    client = TestClient(server.app)
    assert client.get("/").status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
        pass
    assert client.get("/?token=s3cret").status_code == 200  # sets the cookie
    assert client.get("/gallery").status_code == 200
    with client.websocket_connect("/ws") as ws:  # the cookie alone is enough
        assert ws.receive_json()["type"] in ("hello", "say")


def test_tablet_treeline_is_saved_as_the_current_spot(monkeypatch):
    saved = {}
    monkeypatch.setattr(server.spot_store, "save",
                        lambda root, spots, current: saved.update(spots=spots, current=current))
    monkeypatch.setattr(server.horizon_store, "save", lambda root, mask: saved.update(mask=mask))
    monkeypatch.setattr(server, "_sim_hub", None)
    client = TestClient(server.app)
    assert client.post("/api/spots", json={"name": "Deck", "points": [[0, 30]]}).status_code == 400
    bad = {"name": "Deck", "lat": 100, "lon": 0, "points": [[0, 30], [90, 30], [180, 30]]}
    assert client.post("/api/spots", json=bad).status_code == 400 and not saved
    bad |= {"lat": 26.7, "points": [[0, 30], [90], [180, 30]]}
    assert client.post("/api/spots", json=bad).status_code == 400 and not saved
    r = client.post("/api/spots", json={"name": "Deck", "lat": 26.7, "lon": -80.1,
                                        "points": [[90, 40], [0, 30], [200, 25]]})
    assert r.status_code == 200 and saved["current"] == "Deck"
    assert saved["mask"].points == ((0.0, 30.0), (90.0, 40.0), (200.0, 25.0))
    assert client.post("/api/spots", json=["Deck"]).status_code == 400
    assert client.post("/api/spots", json={"name": "Deck", "points": 5}).status_code == 400
    monkeypatch.setattr(server.spot_store, "declination_deg", lambda lat, lon, when: -8.0)
    r = client.post("/api/spots", json={"name": "Deck", "lat": 26.7, "lon": -80.1, "north": "magnetic",
                                        "points": [[90, 40], [4, 30], [200, 25]]})
    assert r.status_code == 200  # magnetic north is 8 degrees west of true here
    assert saved["mask"].points == ((82.0, 40.0), (192.0, 25.0), (356.0, 30.0))


def test_tonight_page_renders_without_starting_the_telescope(monkeypatch):
    monkeypatch.setattr(server.tonight_report, "build", lambda spots, now, clouds: None)
    r = TestClient(server.app).get("/tonight")
    assert r.status_code == 200 and "It does not get dark tonight." in r.text


