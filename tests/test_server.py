import json
import os
import threading
import time

os.environ["ASTRO_SIM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""  # tests never call the real API
os.environ["ASTRO_NO_LOG_FILE"] = "1"  # don't write data/logs from tests
os.environ["ASTRO_OFFLINE"] = "1"  # no forecast fetches from tests

import pytest
from fastapi.testclient import TestClient

from astro import server
from astro.session import Session


@pytest.fixture
def night(monkeypatch):
    """Video tests must not depend on the wall clock: by day the Sun gate refuses the camera."""
    monkeypatch.setattr(Session, "exposure_safety", lambda self: None)


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
    monkeypatch.setattr(server.spot_store, "load", lambda root: ([], None))
    monkeypatch.setattr(server.spot_store, "save", lambda root, spots, current: None)


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

    def warm(self, phrases):  # the server pre-renders cues at startup
        pass


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
    import time as _time

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


def test_handsfree_needs_the_wake_word():
    def spoken(ws, text):
        ws.send_json({"type": "text", "text": text, "spoken": True})

    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "handsfree", "on": True})
        spoken(ws, "go to pizza")
        assert "go to pizza" in receive_until(ws, "ignored")
        spoken(ws, "Astro")  # wake word alone: the next utterance is the command
        receive_until(ws, "armed")
        spoken(ws, "go to pizza")
        assert "go to pizza" in receive_until(ws, "heard")
        spoken(ws, "Astro, go to pizza")  # wake word and command together
        assert "go to pizza" in receive_until(ws, "heard")
        ws.send_json({"type": "handsfree", "on": False})
        spoken(ws, "go to pizza")  # no wake word needed once off
        assert "go to pizza" in receive_until(ws, "heard")


def test_handsfree_hints_name_the_wake_word():
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "blah blah"})
        assert "Say 'what's good tonight'" in receive_until(ws, "say")
        ws.send_json({"type": "handsfree", "on": True})
        ws.send_json({"type": "text", "text": "blah blah"})
        assert "Say 'Astro, what's good tonight' or 'Astro, go to Saturn'" in receive_until(ws, "say")


def test_pages_revalidate():
    assert TestClient(server.app).get("/").headers["cache-control"] == "no-cache"


def test_service_worker_is_stamped_with_the_build():
    r = TestClient(server.app).get("/sw.js")
    assert r.text.startswith('const BUILD = "') and "skipWaiting" in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_handsfree_ignores_the_same_command_while_it_answers():
    def spoken(ws, text):
        ws.send_json({"type": "text", "text": text, "spoken": True})

    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "handsfree", "on": True})
        spoken(ws, "Astro, go to pizza. Astro, go to pizza.")  # whisper said it twice: one command
        assert "go to pizza" in receive_until(ws, "heard")
        server.get_hub().speaking_until = time.monotonic() + 60  # still reading the answer
        spoken(ws, "Astro, go to pizza")
        assert "go to pizza" in receive_until(ws, "ignored")
        spoken(ws, "Astro, stop")  # a different command always gets through
        assert "stop" in receive_until(ws, "heard")


def test_handsfree_transcription_failure_is_silent(monkeypatch):
    def broken(audio):
        raise RuntimeError("whisper died")

    monkeypatch.setattr(server.stt, "transcribe", broken)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "handsfree", "on": True})
        ws.send_bytes(b"audio")
        receive_until(ws, "ignored")  # no spoken apology for background noise
        ws.send_json({"type": "handsfree", "on": False})
        ws.send_bytes(b"audio")
        assert "something went wrong" in receive_until(ws, "say")  # but a button press is told


def test_handsfree_empty_transcript_is_silent(monkeypatch):
    monkeypatch.setattr(server.stt, "transcribe", lambda audio: "")
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "handsfree", "on": True})
        ws.send_bytes(b"audio")
        receive_until(ws, "ignored")  # noise whisper heard as nothing: no "didn't hear anything"


def test_client_log_reaches_the_server_log(caplog):
    with TestClient(server.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "client_log", "text": "The browser blocked speech"})
        ws.send_json({"type": "text", "text": "stop"})  # a later reply proves it was handled
        receive_until(ws, "say")
    assert any(r.getMessage() == "client" for r in caplog.records)


def test_camera_and_debug_views():
    client = TestClient(server.app)
    with client.websocket_connect("/ws"):  # starts the simulated hub
        hub = server.get_hub()
        assert client.get("/api/camera/finder.jpg").status_code == 404  # nothing captured yet
        hub.session.finder.camera.capture()
        r = client.get("/api/camera/finder.jpg")
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and "x-frame-age" in r.headers
        assert client.get("/api/camera/other.jpg").status_code == 404
        info = client.get("/api/debug").json()
        assert {"time", "site", "pointing", "finder_camera", "guidance", "system"} <= info.keys()
        assert info["finder_camera"]["frame_px"] is not None
        ws_msgs = hub.session.handle("show me the finder")
        assert ws_msgs[0] == {"type": "view", "what": "finder"}


@pytest.mark.usefixtures("night")
def test_live_video_starts_stops_and_restores_the_camera():
    with TestClient(server.app).websocket_connect("/ws"):
        session = server.get_hub().session
        cam = session.finder.camera
        before = (cam.exposure_s, cam.gain)
        out = session.handle("live video of the finder")
        assert out[0] == {"type": "view", "what": "finder", "video": True}
        assert cam.exposure_s != before[0]
        t0 = cam.last_at
        end = time.monotonic() + 5
        while cam.last_at == t0 and time.monotonic() < end:
            time.sleep(0.05)
        assert cam.last_at != t0  # frames keep coming
        out = session.handle("stop the video")
        assert out[0] == {"type": "view", "what": "finder"}
        assert (cam.exposure_s, cam.gain) == before
        assert "no video" in session.handle("stop the video")[0]["text"]


@pytest.mark.usefixtures("night")
def test_video_runs_through_unrelated_commands_until_stopped():
    with TestClient(server.app).websocket_connect("/ws"):
        session = server.get_hub().session
        session.handle("live video of the finder")
        session.handle("barlow in")  # doesn't need a camera
        assert session.video_active("finder")
        assert session.handle("stop")[0] == {"type": "view", "what": "finder"}  # nothing else running
        session.handle("live video of the finder")
        session.handle("go to Saturn")  # guidance needs the camera back
        assert session.video_now() is None


@pytest.mark.usefixtures("night")
def test_main_video_leaves_the_finder_solving_and_a_failed_start_unpauses():
    with TestClient(server.app).websocket_connect("/ws"):
        session = server.get_hub().session
        session.finder.paused = threading.Event()  # as the plate-solve tracker has
        session.handle("live video of the main camera")
        assert session.video_active("main") and not session.finder.paused.is_set()
        session.handle("stop the video")

        def broken(s):
            raise RuntimeError("camera unplugged")

        cam = session.finder.camera._camera
        real, cam.set_exposure = cam.set_exposure, broken
        try:
            with pytest.raises(RuntimeError):
                session.handle("live video of the finder")
        finally:
            cam.set_exposure = real
        assert not session.finder.paused.is_set() and session.video_now() is None


@pytest.mark.usefixtures("night")
def test_video_buttons_are_silent_and_a_reload_shows_the_video():
    def until_view(ws):
        while True:
            msg = ws.receive()
            if not msg.get("text"):
                continue  # speech audio
            if (m := json.loads(msg["text"]))["type"] == "view":
                return m
            assert m["type"] not in ("say", "notice"), m  # silent; and fail, don't wait forever

    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "video", "camera": "finder"})
        assert until_view(ws) == {"type": "view", "what": "finder", "video": True}
        with client.websocket_connect("/ws") as ws2:  # another tablet, or this one reloaded
            assert until_view(ws2) == {"type": "view", "what": "finder", "video": True}
        ws.send_json({"type": "video", "camera": None, "then": "debug"})
        assert until_view(ws) == {"type": "view", "what": "debug"}
        assert server.get_hub().session.video_now() is None
        ws.send_json({"type": "text", "text": "barlow in"})
        assert "Barlow" in receive_until(ws, "say")  # the first thing spoken is this answer


@pytest.mark.usefixtures("night")
def test_video_stream_serves_motion_jpeg():
    import asyncio

    async def two_frames():
        response = await server.api_camera_stream("finder")
        parts = []
        async for part in response.body_iterator:
            parts.append(part)
            if len(parts) == 2:
                break
        await response.body_iterator.aclose()
        return response, parts

    with TestClient(server.app).websocket_connect("/ws"):
        session = server.get_hub().session
        session.handle("live video of the finder")
        response, parts = asyncio.run(two_frames())
        assert response.media_type.startswith("multipart/x-mixed-replace")
        assert all(b"Content-Type: image/jpeg" in p and b"\xff\xd8" in p for p in parts)
        session.handle("stop the video")


def test_save_utterance_keeps_audio_and_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UTTERANCES", tmp_path)
    server.save_utterance(b"RIFFxxxx", "astro stop", True)
    (wav,) = tmp_path.glob("*.wav")
    assert wav.read_bytes() == b"RIFFxxxx"
    assert json.loads(wav.with_suffix(".json").read_text()) == {"heard": "astro stop", "handsfree": True}


def test_hardware_startup_failure_is_reported(monkeypatch):
    def broken():
        raise RuntimeError("SV905C not found")

    monkeypatch.setattr(server, "get_hub", broken)
    with TestClient(server.app).websocket_connect("/ws") as ws:
        assert "isn't ready: SV905C not found" in receive_until(ws, "unavailable")


def test_state_updates_coalesce_while_events_keep_order():
    import asyncio

    from astro.server import Hub

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
    from starlette.websockets import WebSocketDisconnect

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
    r = client.post("/api/spots", json={"name": "Deck", "lat": 26.7, "lon": -80.1,
                                        "points": [[90, 40], [0, 30], [200, 25]]})
    assert r.status_code == 200 and saved["current"] == "Deck"
    assert saved["mask"].points == ((0.0, 30.0), (90.0, 40.0), (200.0, 25.0))


def test_tonight_page_renders_without_starting_the_telescope(monkeypatch):
    monkeypatch.setattr(server.tonight_report, "build", lambda spots, now: None)
    r = TestClient(server.app).get("/tonight")
    assert r.status_code == 200 and "It does not get dark tonight." in r.text
