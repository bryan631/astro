"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import contextlib
import functools
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from astro import calibration_store, logs, site_store
from astro.agent import Agent
from astro.devices import config as devices
from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.main_cam import SimMainCamera
from astro.devices.sim.scope import SimEncoders, SimScope, SimUser
from astro.guidance.centering import centering_phrases
from astro.guidance.engine import cue_phrases
from astro.optics import MAIN_SENSOR_PX
from astro.planner import horizon_store
from astro.planner.weather import cloud_cover_pct
from astro.pointing.coords import Site
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.session import Session, utcnow
from astro.voice.speech import Stt, Tts
from astro.wake import strip_wake

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
                      on_calibration_change=lambda d: calibration_store.save(ROOT, d))
    return session, scope


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
        main = devices.open_camera(cfg["main"]) if cfg["main"]["driver"] != "none" else None
        if main is not None:
            _hardware_closers.append(main.close)
        override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
        return Session(site, clock=clock, developer_override=override, finder=finder,
                       main_camera=main, main_sensor=main.sensor_size if main else MAIN_SENSOR_PX,
                       data_dir=ROOT / "data", on_site_change=lambda s: site_store.save(ROOT, s),
                       horizon=horizon_store.load(ROOT),
                       on_horizon_change=lambda m: horizon_store.save(ROOT, m),
                       weather=None if OFFLINE else cloud_cover_pct,
                       calibration=calibration_store.load(ROOT),
                       on_calibration_change=lambda d: calibration_store.save(ROOT, d))
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
            if msg["type"] == "say" and tts.available():
                try:
                    audio = await asyncio.to_thread(tts.synthesize, msg["text"])
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
    conn = {"handsfree": False, "armed_until": 0.0}  # per-tablet voice mode
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
            except Exception:
                log.exception("message failed")
                await hub.broadcast({"type": "say", "text": SORRY})
    except WebSocketDisconnect:
        pass
    finally:
        hub.leave(socket)
        if hub is _sim_hub and not hub.clients:
            _sim_hub = None  # the next tablet starts a fresh simulated world


def save_utterance(audio: bytes, text: str, handsfree: bool) -> None:
    """Keep the recording and what whisper heard: real audio to score voice changes against."""
    UTTERANCES.mkdir(parents=True, exist_ok=True)
    stem = UTTERANCES / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    ext = "wav" if audio[:4] == b"RIFF" else "webm"
    Path(f"{stem}.{ext}").write_bytes(audio)
    Path(f"{stem}.json").write_text(json.dumps({"heard": text, "handsfree": handsfree}))


async def handle_spoken(hub: Hub, socket: WebSocket, conn: dict, text: str) -> None:
    """Speech from the tablet. In hands-free mode only "Astro ..." (or the utterance right after
    a bare "Astro") is a command; anything else is shown but ignored."""
    if conn["handsfree"]:
        now = time.monotonic()
        command = strip_wake(text)
        if command is None and now < conn["armed_until"]:
            command = text
        if command is None:
            await socket.send_json({"type": "ignored", "text": text})
            return
        if not command:
            conn["armed_until"] = now + ARMED_S
            await socket.send_json({"type": "armed", "seconds": ARMED_S})
            return
        conn["armed_until"] = 0.0
        text = command
    await hub.handle_text(socket, text)


async def handle_message(hub: Hub, socket: WebSocket, conn: dict, msg: dict) -> None:
    session = hub.session
    if msg.get("bytes"):  # recorded speech from the tablet
        text = await asyncio.to_thread(stt.transcribe, msg["bytes"])
        if os.environ.get("ASTRO_SAVE_AUDIO") == "1":
            save_utterance(msg["bytes"], text, conn["handsfree"])
        if text:
            await handle_spoken(hub, socket, conn, text)
        else:
            await hub.broadcast({"type": "say", "text": "Sorry, I didn't hear anything."})
        return
    if not msg.get("text"):
        return
    data = json.loads(msg["text"])
    request_id = int(data["id"]) if data.get("id") is not None else None
    if data.get("type") == "handsfree":
        conn["handsfree"], conn["armed_until"] = bool(data.get("on")), 0.0
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


@app.get("/gallery")
def gallery() -> list[str]:
    """Processed pictures, newest first."""
    files = sorted(GALLERY.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


app.mount("/pictures", StaticFiles(directory=GALLERY, check_dir=False), name="pictures")
app.mount("/live", StaticFiles(directory=LIVE, check_dir=False), name="live")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
