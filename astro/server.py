"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import contextlib
import functools
import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import socket
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from astro import calibration_store, logs, site_store
from astro import spots as spot_store
from astro.agent import Agent
from astro.devices import config as devices
from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.main_cam import SimMainCamera
from astro.devices.sim.scope import SimEncoders, SimScope, SimUser
from astro.guidance.centering import centering_phrases
from astro.guidance.engine import cue_phrases
from astro.optics import MAIN_SENSOR_PX
from astro.planner import horizon_store
from astro.planner import report as tonight_report
from astro.planner.horizon import HorizonMask
from astro.planner.weather import cloud_cover_pct
from astro.pointing.coords import Site
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.process.view import ROTATE, jpeg
from astro.session import Session, utcnow
from astro.voice.speech import Stt, Tts
from astro.wake import add_wake, collapse_repeats, strip_wake

ROOT = Path(__file__).resolve().parents[1]
SIM = os.environ.get("ASTRO_SIM") == "1"
OFFLINE = os.environ.get("ASTRO_OFFLINE") == "1"  # no forecast fetches (tests, field hotspot)
TICK_S = 0.1
ARMED_S = 8.0  # after the wake word alone, how long the next utterance counts as the command
THINKING_AFTER_S = 1.0  # say "Let me think." if an answer takes longer than this
SORRY = "Sorry, something went wrong. Please try that again."


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=value lines) for secrets like ANTHROPIC_API_KEY."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


log = logging.getLogger("astro.server")
LOGGED = {"say", "picture", "get_location"}  # not the 10 Hz "state" or "live" updates
GALLERY = ROOT / "data" / "gallery"
UTTERANCES = ROOT / "data" / "utterances"  # ASTRO_SAVE_AUDIO=1: what the tablet sent, to tune voice
UTTERANCE_KEEP_DAYS, UTTERANCE_MAX_BYTES = 60, 2 * 1024**3  # then the oldest go
LIVE = ROOT / "data" / "live"  # live-stack previews while they build (not gallery pictures)

# Module state: speech engines, and the shared hubs (one per process; see get_hub).
stt, tts = Stt(), Tts()
_real_hub: "Hub | None" = None
_sim_hub: "Hub | None" = None


def load_site() -> Site:
    return site_store.load(ROOT)


def build_session() -> tuple[Session, SimScope | None]:
    """Sim mode: an uncalibrated simulated mount with a finder that sees the real sky.
    Otherwise the real devices from config/devices.toml."""
    if not SIM:
        return build_real_session(), None
    site, clock = load_site(), utcnow
    scope = SimScope(45, 180)
    camera = SimFinderCamera(lambda: (scope.alt, scope.az), site, clock, solver().star_table)
    finder = FinderSync(camera, solver(), MountModel(), SimEncoders(scope), site, clock)
    override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
    main = SimMainCamera(lambda: (scope.alt, scope.az), site, clock)
    session = Session(site, clock=clock, developer_override=override, finder=finder,
                      main_camera=main, main_sensor=main.sensor_size, data_dir=ROOT / "data",
                      on_site_change=lambda s: on_site_change(s, camera, main),
                      horizon=horizon_store.load(ROOT),
                      on_horizon_change=lambda m: horizon_store.save(ROOT, m),
                      weather=None if OFFLINE else cloud_cover_pct,
                      calibration=calibration_store.load(ROOT),
                      on_calibration_change=lambda d: calibration_store.save(ROOT, d),
                      **_spot_args())
    return session, scope


def _spot_args() -> dict:
    spots, current = spot_store.load(ROOT)
    return {"spots": spots, "spot": current,
            "on_spots_change": lambda sp, cur: spot_store.save(ROOT, sp, cur)}


@functools.cache
def build_real_session() -> Session:
    """Real hardware is opened once per process; every tablet shares this session (one Hub).

    Everything opened is recorded in `_hardware_closers`: a failure part-way closes it all
    again, and shutdown closes it (heaters off, cameras and solver released)."""
    site, clock = load_site(), utcnow
    cfg = devices.load(Path(os.environ.get("ASTRO_DEVICES", ROOT / "config" / "devices.toml")))
    try:
        finder, close_pointing = devices.build_pointing(cfg, solver(), site, clock)
        _hardware_closers.append(close_pointing)
        main = None
        if cfg["main"]["driver"] != "none":
            try:  # an unplugged main camera mustn't stop the finder, voice and guidance
                main = devices.open_camera(cfg["main"])
            except Exception:
                log.exception("main camera unavailable; running without it")
        if main is not None:
            _hardware_closers.append(main.close)
        override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
        return Session(site, clock=clock, developer_override=override, finder=finder,
                       main_camera=main, main_sensor=main.sensor_size if main and main.connected else MAIN_SENSOR_PX,
                       data_dir=ROOT / "data", on_site_change=lambda s: site_store.save(ROOT, s),
                       horizon=horizon_store.load(ROOT),
                       on_horizon_change=lambda m: horizon_store.save(ROOT, m),
                       weather=None if OFFLINE else cloud_cover_pct,
                       calibration=calibration_store.load(ROOT),
                       on_calibration_change=lambda d: calibration_store.save(ROOT, d),
                       **_spot_args())
    except Exception:
        close_hardware()  # roll back, so the next attempt doesn't find devices still owned
        raise


_hardware_closers: list[Callable[[], None]] = []


def close_hardware() -> None:
    """Close every real device opened, newest first; keep going if one fails."""
    while _hardware_closers:
        closer = _hardware_closers.pop()
        try:
            closer()
        except Exception:
            log.exception("closing hardware failed")


def on_site_change(site: Site, finder_cam: SimFinderCamera, main_cam: SimMainCamera) -> None:
    """Persist a GPS fix and move the simulated sky with it."""
    site_store.save(ROOT, site)
    finder_cam.site = site
    main_cam.set_site(site)


@functools.cache
def solver() -> FinderSolver:
    return FinderSolver()  # loads the star database once per process


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup work lives here, not at import (importing must not touch .env, data/ or the
    network); shutdown releases the hardware (S5: heaters off on host shutdown)."""
    load_env()
    GALLERY.mkdir(parents=True, exist_ok=True)
    LIVE.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("ASTRO_NO_LOG_FILE"):  # tests
        log.info("server start", extra={"data": {"log": str(logs.setup(ROOT)), "sim": SIM}})
    warm_speech()
    try:
        yield
    finally:  # also on errors/cancellation: never leave heaters, cameras or the solver running
        for hub in (_sim_hub, _real_hub):
            if hub is not None:
                hub.session.stop_video()  # its thread captures until stopped: before closing
        close_hardware()


app = FastAPI(lifespan=lifespan)


def warm_speech() -> None:
    """Load Piper and pre-render the guidance cues in the background: "stop" must be instant."""
    if tts.available():
        phrases = cue_phrases() + centering_phrases()
        threading.Thread(target=tts.warm, args=(phrases,), daemon=True).start()


class Hub:
    """One session's tablets: a single guidance loop whose output goes to every client.

    Everything outgoing goes through one queue and one sender task: messages keep their order,
    a message and its audio are never interleaved with another, and the guidance loop never
    waits for speech synthesis."""

    def __init__(self, session: Session, scope: SimScope | None):
        self.session, self.scope = session, scope
        self.agent = Agent(session)
        self.user = SimUser() if scope else None
        self.clients: set[WebSocket] = set()
        self.speaking_until = 0.0  # monotonic time the last queued reply finishes playing
        self.handsfree: set[WebSocket] = set()  # tablets in hands-free mode: hints name the wake word
        self._loop: asyncio.Task | None = None
        self._sender: asyncio.Task | None = None
        self._outbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self._state: dict | None = None  # newest unsent state; None in the queue stands for it
        self._t0 = time.monotonic()

    def join(self, socket: WebSocket) -> None:
        self.clients.add(socket)
        if self._loop is None or self._loop.done():
            self._loop = asyncio.create_task(self._guidance_loop())
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_loop())

    def leave(self, socket: WebSocket) -> None:
        self.clients.discard(socket)
        self.handsfree.discard(socket)
        if not self.clients:
            for task in (self._loop, self._sender):
                if task is not None:
                    task.cancel()

    async def broadcast(self, msg: dict) -> None:
        """Queue a message for every tablet (returns at once). States coalesce: a slow
        tablet gets the newest one, not a growing backlog; events keep their order."""
        if msg["type"] == "state":
            queued, self._state = self._state is not None, msg
            if queued:
                return
            msg = None
        await self._outbox.put(msg)

    async def _send_loop(self) -> None:
        while True:
            msg = await self._outbox.get()
            if msg is None:
                msg, self._state = self._state, None
            if msg["type"] in LOGGED:
                log.info(msg["type"], extra={"data": {k: v for k, v in msg.items() if k != "type"}})
            audio = None
            if msg["type"] == "say" and self.handsfree:  # one shared session: text and audio agree
                msg = {**msg, "text": add_wake(msg["text"])}
            if msg["type"] == "say" and tts.available():
                try:
                    audio = await asyncio.to_thread(tts.synthesize, msg["text"])
                    self.speaking_until = max(self.speaking_until, time.monotonic()) + wav_seconds(audio)
                except Exception:  # the words still go out as text
                    log.exception("speech synthesis failed")
            to = list(self.clients)
            asker = msg.pop("_to", None)  # GPS requests go only to the tablet that asked:
            if asker in self.clients:     # every tablet answering with its own fix would race
                to = [asker]
            for client in to:
                try:
                    await client.send_json(msg)
                    if audio:
                        await client.send_bytes(audio)
                except (WebSocketDisconnect, RuntimeError):
                    self.clients.discard(client)

    async def handle_text(self, socket: WebSocket, text: str) -> None:
        log.info("heard", extra={"data": {"text": text}})
        await socket.send_json({"type": "heard", "text": text})
        reply = asyncio.create_task(asyncio.to_thread(self.agent.handle, text))
        done, _ = await asyncio.wait({reply}, timeout=THINKING_AFTER_S)
        if not done:  # a slow LLM round trip: let the user know we heard them
            await self.broadcast({"type": "say", "text": "Let me think."})
        for out in await reply:
            await self.broadcast(_for(out, socket))

    async def _guidance_loop(self) -> None:
        failing = False  # speak a failure once per streak, not ten times a second
        while True:
            t = time.monotonic() - self._t0
            try:
                msgs = await asyncio.to_thread(self.session.tick, t)  # camera calls block
                failing = False
            except Exception:  # never let one bad tick end guidance for the night
                log.exception("tick failed")
                msgs = [] if failing else [{"type": "say", "text": SORRY}]
                failing = True
            for msg in msgs:
                if self.user and msg["type"] == "say":
                    self.user.hear(msg["text"], t)
                await self.broadcast(msg)
            if self.user and self.scope:
                self.scope.step(*self.user.act(t), TICK_S)
            await asyncio.sleep(TICK_S)


def _for(msg: dict, socket: WebSocket) -> dict:
    """A GPS request is addressed to the tablet whose turn produced it."""
    return {**msg, "_to": socket} if msg["type"] == "get_location" else msg


def get_hub() -> Hub:
    """One shared hub, so every tablet sees the same session (V8). Sim: the simulated world
    lasts while any tablet is connected, then starts fresh."""
    global _sim_hub, _real_hub
    if SIM:
        if _sim_hub is None:  # dropped when its last tablet leaves (see ws)
            _sim_hub = Hub(*build_session())
        return _sim_hub
    if _real_hub is None:
        _real_hub = Hub(build_real_session(), None)
    return _real_hub


def running_hub() -> Hub:
    """The hub a tablet already started. The camera and debug views never start the hardware
    themselves: they run on worker threads, and two starts would open each camera twice."""
    hub = _sim_hub if SIM else _real_hub
    if hub is None:
        raise HTTPException(503, "The telescope isn't started yet: open the page first.")
    return hub


def _allowed(conn: Request | WebSocket) -> bool:
    """Optional shared token (ASTRO_TOKEN) for public WiFi: open the app once as
    https://.../?token=..., and a cookie remembers it. Unset: anyone on the network."""
    token = os.environ.get("ASTRO_TOKEN")
    if not token:
        return True
    given = conn.query_params.get("token") or conn.cookies.get("astro_token") or ""
    return hmac.compare_digest(given.encode(), token.encode())


@app.middleware("http")
async def require_token(request: Request, call_next):
    if not _allowed(request):
        return PlainTextResponse("Open the link with the access token.", status_code=401)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"  # revalidate (ETag) so a tablet never keeps an old page
    if token := request.query_params.get("token"):
        response.set_cookie("astro_token", token, httponly=True,
                            secure=request.url.scheme == "https",
                            samesite="strict", max_age=365 * 86400)
    return response


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    global _sim_hub
    if not _allowed(socket):
        await socket.close(code=1008)  # policy violation
        return
    await socket.accept()
    try:
        hub = get_hub()
    except Exception as e:  # e.g. a camera not plugged in: say so instead of a dead page
        log.exception("startup failed")
        await socket.send_json({"type": "unavailable", "text": f"The telescope isn't ready: {e}"})
        await socket.close()
        return
    await socket.send_json({"type": "hello", "server_stt": stt.available(),
                            "server_tts": tts.available()})
    hub.join(socket)
    if name := hub.session.video_now():  # video keeps running across reloads: show it again
        await socket.send_json({"type": "view", "what": name, "video": True})
    conn = {"handsfree": False, "armed_until": 0.0, "last": ""}  # per-tablet voice mode
    if not site_store.has_saved(ROOT):  # setup: first run at this installation
        for out in hub.session.request_location():
            await hub.broadcast(_for(out, socket))
    try:
        while True:
            msg = await socket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            try:  # one bad message must not drop the tablet
                await handle_message(hub, socket, conn, msg)
            except WebSocketDisconnect:
                break
            except Exception:
                if WebSocketState.DISCONNECTED in (socket.client_state, socket.application_state):
                    break  # the tablet left (reload, restart) mid-answer: nothing to apologise to
                log.exception("message failed")
                await hub.broadcast({"type": "say", "text": SORRY})
    except WebSocketDisconnect:
        pass
    finally:
        hub.leave(socket)
        if hub is _sim_hub and not hub.clients:
            hub.session.stop_video()  # no orphan capture thread
            _sim_hub = None  # the next tablet starts a fresh simulated world


def wav_seconds(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio)) as w:
        return w.getnframes() / w.getframerate()


def save_utterance(audio: bytes, text: str, handsfree: bool) -> None:
    """Keep the recording and what whisper heard: real audio to score voice changes against."""
    UTTERANCES.mkdir(parents=True, exist_ok=True)
    stem = UTTERANCES / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    ext = "wav" if audio[:4] == b"RIFF" else "webm"
    Path(f"{stem}.{ext}").write_bytes(audio)
    Path(f"{stem}.json").write_text(json.dumps({"heard": text, "handsfree": handsfree}))
    prune_media(UTTERANCES, UTTERANCE_KEEP_DAYS, UTTERANCE_MAX_BYTES)


def prune_media(folder: Path, keep_days: float, max_bytes: int) -> None:
    """Delete files older than `keep_days`, then the oldest until the folder fits `max_bytes`."""
    files = sorted((f for f in folder.iterdir() if f.is_file()), key=lambda f: f.stat().st_mtime)
    cutoff, total = time.time() - keep_days * 86400, sum(f.stat().st_size for f in files)
    for f in files:
        if f.stat().st_mtime >= cutoff and total <= max_bytes:
            break
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


async def handle_spoken(hub: Hub, socket: WebSocket, conn: dict, text: str) -> None:
    """Speech from the tablet. In hands-free mode only "Astro ..." (or the utterance right after
    a bare "Astro") is a command; anything else is shown but ignored."""
    text = collapse_repeats(text)
    if conn["handsfree"]:
        now = time.monotonic()
        command = strip_wake(text)
        if command is None and now < conn["armed_until"]:
            command = text
        if command is None:
            log.info("ignored", extra={"data": {"text": text}})  # tune the wake word from these
            await socket.send_json({"type": "ignored", "text": text})
            return
        if not command:
            conn["armed_until"] = now + ARMED_S
            await socket.send_json({"type": "armed", "seconds": ARMED_S})
            return
        conn["armed_until"] = 0.0
        same = re.sub(r"\W+", " ", command).strip().lower()
        if same == conn["last"] and now < hub.speaking_until:  # the same command while it answers
            await socket.send_json({"type": "ignored", "text": text})
            return
        conn["last"] = same
        text = command
    await hub.handle_text(socket, text)


async def handle_message(hub: Hub, socket: WebSocket, conn: dict, msg: dict) -> None:
    session = hub.session
    if msg.get("bytes"):  # recorded speech from the tablet
        try:
            text = await asyncio.to_thread(stt.transcribe, msg["bytes"])
        except Exception:
            if not conn["handsfree"]:
                raise  # the tablet's own press: tell them it failed
            log.exception("transcription failed")  # hands-free hears background noise: stay quiet
            await socket.send_json({"type": "ignored", "text": ""})
            return
        if os.environ.get("ASTRO_SAVE_AUDIO") == "1":
            save_utterance(msg["bytes"], text, conn["handsfree"])
        if text:
            await handle_spoken(hub, socket, conn, text)
        elif conn["handsfree"]:  # background noise: stay quiet
            await socket.send_json({"type": "ignored", "text": ""})
        else:
            await hub.broadcast({"type": "say", "text": "Sorry, I didn't hear anything."})
        return
    if not msg.get("text"):
        return
    data = json.loads(msg["text"])
    request_id = int(data["id"]) if data.get("id") is not None else None
    if data.get("type") == "client_log":  # a failure inside the tablet's browser
        log.warning("client", extra={"data": {"text": str(data.get("text", ""))[:300]}})
    elif data.get("type") == "handsfree":
        conn["handsfree"], conn["armed_until"] = bool(data.get("on")), 0.0
        (hub.handsfree.add if conn["handsfree"] else hub.handsfree.discard)(socket)
    elif data.get("type") == "checks":  # the daytime "skip checks" toggle
        session.override = bool(data.get("off"))
    elif data.get("type") == "side_finder":  # scripts/dev/drift_offset.py: both cameras at once
        await asyncio.to_thread(session.side_finder, bool(data.get("on")))
    elif data.get("type") == "camera":  # the camera view's exposure and zoom buttons
        if data.get("camera") in ("finder", "main"):
            await asyncio.to_thread(session.adjust_camera, data["camera"], data.get("exposure"),
                                    data.get("zoom"))
    elif data.get("type") == "video":  # the Live video button: answered silently
        for out in await asyncio.to_thread(session.video, data.get("camera"), data.get("then")):
            await hub.broadcast(out)
    elif data.get("type") == "text":
        if data.get("spoken"):  # the browser's own recognizer
            await handle_spoken(hub, socket, conn, data["text"])
        else:
            await hub.handle_text(socket, data["text"])
    elif data.get("type") == "location":
        log.info("location", extra={"data": {"accuracy_m": data.get("accuracy")}})
        alt = float(data["alt"]) if data.get("alt") is not None else None
        accuracy = float(data["accuracy"]) if data.get("accuracy") is not None else None
        for out in session.set_location(float(data["lat"]), float(data["lon"]), alt, accuracy,
                                        request_id):
            await hub.broadcast(out)
    elif data.get("type") == "location_error":
        for out in session.location_failed(str(data.get("message", "")), request_id):
            await hub.broadcast(out)


@app.get("/tonight")
async def tonight() -> HTMLResponse:
    """What every spot can see tonight with the cloud forecast; doesn't start the hardware."""
    spots, _ = spot_store.load(ROOT)
    spots = spots or [spot_store.Spot("Here", load_site(), horizon_store.load(ROOT))]
    clouds = (lambda lat, lon: None) if OFFLINE else tonight_report.hourly_cloud_cover
    night = await asyncio.to_thread(tonight_report.build, spots, utcnow(), clouds)
    return HTMLResponse(tonight_report.render_html(night, spots))


@app.get("/api/spots")
def api_spots() -> list[str]:
    return [s.name for s in spot_store.load(ROOT)[0]]


@app.post("/api/spots")
async def api_save_spot(request: Request) -> dict:
    """A treeline from the tablet walk: {name, lat, lon, elevation?, points: [[az, alt], ...]}."""
    data = await request.json()
    if not isinstance(data, dict) or not isinstance(data.get("points"), list):
        raise HTTPException(400, "A spot needs a name and at least three treeline marks.")
    name, points = str(data.get("name", "")).strip(), data["points"]
    if not name or len(points) < 3:
        raise HTTPException(400, "A spot needs a name and at least three treeline marks.")
    try:
        lat, lon, elev = (float(data["lat"]), float(data["lon"]), float(data.get("elevation") or 0))
        if not (all(map(math.isfinite, (lat, lon, elev))) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("that location doesn't look right")
        # A phone compass points to magnetic north: turn its azimuths to true north.
        turn = spot_store.declination_deg(lat, lon, utcnow()) if data.get("north") == "magnetic" else 0.0
        marks = tuple(sorted(((float(az) + turn) % 360, float(alt)) for az, alt in points))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Bad spot data: {e}") from e
    if not all(math.isfinite(az) and -10 <= alt <= 90 for az, alt in marks):
        raise HTTPException(400, "That treeline doesn't look right.")
    spot = spot_store.Spot(name, Site(lat, lon, elev), HorizonMask(marks))
    hub = _sim_hub if SIM else _real_hub
    if hub is not None:  # the running session switches to it (site, treeline, planning)
        msgs = await asyncio.to_thread(hub.session.save_spot, spot)  # the tick may hold its lock
        for m in msgs:
            await hub.broadcast(m)
        return {"saved": name, "said": msgs[0]["text"]}
    spots, _ = spot_store.load(ROOT)  # not started: store it as the current spot for next time
    spot_store.save(ROOT, spot_store.upsert(spots, spot), name)
    site_store.save(ROOT, spot.site)
    horizon_store.save(ROOT, spot.mask)
    return {"saved": name, "said": f"Saved {name}."}


@app.get("/gallery")
def gallery() -> list[str]:
    """Processed pictures, newest first."""
    files = sorted(GALLERY.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


_net = {"at": -1e9, "ok": False}


def internet_ok() -> bool:
    """Can the Mele reach Claude's servers? Checked at most every 30 s."""
    if time.monotonic() - _net["at"] > 30:
        try:
            socket.create_connection(("api.anthropic.com", 443), timeout=2).close()
            ok = True
        except OSError:
            ok = False
        _net.update(at=time.monotonic(), ok=ok)
    return _net["ok"]


@app.get("/api/status")
def api_status() -> dict:
    """The page's status line: cameras, encoders, internet and Claude."""
    hub = running_hub()
    return {**hub.session.connections(), "internet": internet_ok(), "claude": hub.agent.status(),
            "daytime": hub.session.daytime(), "checks_off": hub.session.override,
            "build": web_build()}  # the page reloads itself when this changes


@app.get("/api/debug")
def api_debug() -> dict:
    return running_hub().session.debug_info()


@app.get("/api/camera/{name}.jpg")
def api_camera(name: str) -> Response:
    """The named camera's last frame (finder or main), as the page's live view polls it."""
    if name not in ("finder", "main"):
        raise HTTPException(404)
    session = running_hub().session
    frame = session.camera_frame(name)
    if frame is None:  # say why, so a blank view isn't a mystery
        raise HTTPException(404, "no frame yet: it hasn't taken a picture since the server started")
    raw, bayer, age = frame
    return Response(jpeg(raw, bayer, session.zoom[name], ROTATE[name]), media_type="image/jpeg", headers={"X-Frame-Age": f"{age:.1f}"})


@app.get("/api/camera/{name}.mjpg")
async def api_camera_stream(name: str) -> StreamingResponse:
    """Live video: every new frame of the camera as motion JPEG, while the video runs."""
    if name not in ("finder", "main"):
        raise HTTPException(404)
    session = running_hub().session
    cam = session.camera(name)
    if cam is None:
        raise HTTPException(404, "no such camera")

    async def frames():
        seen = 0.0
        while session.video_active(name):  # the stream ends with the video
            if cam.last is not None and cam.last_at != seen:
                seen = cam.last_at
                data = await asyncio.to_thread(jpeg, cam.last, cam.bayer, session.zoom[name])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data)
                       + data + b"\r\n")
            await asyncio.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/sw.js")
def service_worker() -> Response:
    """sw.js stamped with a hash of the web files: every deploy changes it, so tablets update."""
    return Response(f'const BUILD = "{web_build()}";\n{(ROOT / "web" / "sw.js").read_text()}',
                    media_type="text/javascript")


def web_build() -> str:
    """A hash of the web files: changes with every deploy of the page."""
    web = ROOT / "web"
    return hashlib.sha256(b"".join(f.read_bytes() for f in sorted(web.glob("*")) if f.is_file())).hexdigest()[:12]


app.mount("/pictures", StaticFiles(directory=GALLERY, check_dir=False), name="pictures")
app.mount("/live", StaticFiles(directory=LIVE, check_dir=False), name="live")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
