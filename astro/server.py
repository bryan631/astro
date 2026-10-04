"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from astro import logs, site_store
from astro.agent import Agent
from astro.devices import config as devices
from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.main_cam import SimMainCamera
from astro.devices.sim.scope import SimEncoders, SimScope, SimUser
from astro.guidance.centering import centering_phrases
from astro.guidance.engine import cue_phrases
from astro.planner import horizon_store
from astro.planner.weather import cloud_cover_pct
from astro.pointing.coords import Site
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.session import Session, utcnow
from astro.voice.speech import Stt, Tts

ROOT = Path(__file__).resolve().parents[1]
SIM = os.environ.get("ASTRO_SIM") == "1"
OFFLINE = os.environ.get("ASTRO_OFFLINE") == "1"  # no forecast fetches (tests, field hotspot)
TICK_S = 0.1
THINKING_AFTER_S = 1.0  # say "Let me think." if an answer takes longer than this


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=value lines) for secrets like ANTHROPIC_API_KEY."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


log = logging.getLogger("astro.server")
LOGGED = {"say", "picture", "get_location"}  # not the 10 Hz "state" or "live" updates


def load_site() -> Site:
    return site_store.load(ROOT)


def build_session() -> tuple[Session, SimScope | None]:
    """Sim mode: an uncalibrated simulated mount with a finder that sees the real sky.
    Otherwise the real devices from config/devices.toml."""
    if not SIM:
        return build_real_session(), None
    site, clock = load_site(), utcnow
    scope = SimScope(45, 180)
    camera = SimFinderCamera(lambda: (scope.alt, scope.az), site, clock, solver()._t3.star_table)
    finder = FinderSync(camera, solver(), MountModel(), SimEncoders(scope), site, clock)
    override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
    main = SimMainCamera(lambda: (scope.alt, scope.az), site, clock)
    session = Session(site, clock=clock, developer_override=override, finder=finder,
                      main_camera=main, main_sensor=main.sensor_size, data_dir=ROOT / "data",
                      on_site_change=lambda s: on_site_change(s, camera, main),
                      horizon=horizon_store.load(ROOT),
                      on_horizon_change=lambda m: horizon_store.save(ROOT, m),
                      weather=None if OFFLINE else cloud_cover_pct)
    return session, scope


@functools.cache
def build_real_session() -> Session:
    """Real hardware is opened once per process; every tablet shares this session (one Hub).

    Everything opened is recorded in `_hardware_closers`: a failure part-way closes it all
    again, and shutdown closes it (heaters off, cameras and solver released)."""
    site, clock = load_site(), utcnow
    cfg = devices.load(Path(os.environ.get("ASTRO_DEVICES", ROOT / "config" / "devices.toml")))
    devices.validate(cfg)  # all drivers known before any hardware opens
    try:
        finder, close_pointing = devices.build_pointing(cfg, solver(), site, clock)
        _hardware_closers.append(close_pointing)
        main = devices.open_camera(cfg["main"]) if cfg["main"]["driver"] != "none" else None
        if main is not None:
            _hardware_closers.append(main.close)
        override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
        return Session(site, clock=clock, developer_override=override, finder=finder,
                       main_camera=main, main_sensor=main.sensor_size if main else (3856, 2180),
                       data_dir=ROOT / "data", on_site_change=lambda s: site_store.save(ROOT, s),
                       horizon=horizon_store.load(ROOT),
                       on_horizon_change=lambda m: horizon_store.save(ROOT, m),
                       weather=None if OFFLINE else cloud_cover_pct)
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
    if not os.environ.get("ASTRO_NO_LOG_FILE"):  # tests
        log.info("server start", extra={"data": {"log": str(logs.setup(ROOT)), "sim": SIM}})
    warm_speech()
    try:
        yield
    finally:  # also on errors/cancellation: never leave heaters, cameras or the solver running
        close_hardware()


app = FastAPI(lifespan=lifespan)
stt, tts = Stt(), Tts()


def warm_speech() -> None:
    """Load Piper and pre-render the guidance cues in the background: "stop" must be instant."""
    if tts.available():
        phrases = cue_phrases() + centering_phrases()
        threading.Thread(target=tts.warm, args=(phrases,), daemon=True).start()


class Hub:
    """One session's tablets: a single guidance loop whose output goes to every client."""

    def __init__(self, session: Session, scope: SimScope | None):
        self.session, self.scope = session, scope
        self.agent = Agent(session)
        self.user = SimUser() if scope else None
        self.clients: set[WebSocket] = set()
        self._loop: asyncio.Task | None = None
        self._t0 = time.monotonic()

    def join(self, socket: WebSocket) -> None:
        self.clients.add(socket)
        if self._loop is None or self._loop.done():
            self._loop = asyncio.create_task(self._guidance_loop())

    def leave(self, socket: WebSocket) -> None:
        self.clients.discard(socket)
        if not self.clients and self._loop is not None:
            self._loop.cancel()

    async def broadcast(self, msg: dict) -> None:
        if msg["type"] in LOGGED:
            log.info(msg["type"], extra={"data": {k: v for k, v in msg.items() if k != "type"}})
        audio = None
        if msg["type"] == "say" and tts.available():
            audio = await asyncio.to_thread(tts.synthesize, msg["text"])
        for client in list(self.clients):
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
            await self.broadcast(out)

    async def _guidance_loop(self) -> None:
        while True:
            t = time.monotonic() - self._t0
            try:
                msgs = await asyncio.to_thread(self.session.tick, t)  # camera calls block
            except Exception:  # never let one bad tick end guidance for the night
                log.exception("tick failed")
                msgs = []
            for msg in msgs:
                if self.user and msg["type"] == "say":
                    self.user.hear(msg["text"], t)
                await self.broadcast(msg)
            if self.user and self.scope:
                self.scope.step(*self.user.act(t), TICK_S)
            await asyncio.sleep(TICK_S)


def get_hub() -> Hub:
    """Sim: a fresh simulated world per connection. Real: one shared hub for the hardware."""
    if SIM:
        return Hub(*build_session())
    global _real_hub
    if _real_hub is None:
        _real_hub = Hub(build_real_session(), None)
    return _real_hub


_real_hub: Hub | None = None


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    hub = get_hub()
    session = hub.session
    await socket.send_json({"type": "hello", "server_stt": stt.available(),
                            "server_tts": tts.available()})
    hub.join(socket)
    if not site_store.has_saved(ROOT):  # setup: first run at this installation
        for out in session.request_location():
            await hub.broadcast(out)
    try:
        while True:
            msg = await socket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):  # recorded speech from the tablet
                text = await asyncio.to_thread(stt.transcribe, msg["bytes"])
                if text:
                    await hub.handle_text(socket, text)
                else:
                    await hub.broadcast({"type": "say", "text": "Sorry, I didn't hear anything."})
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "text":
                    await hub.handle_text(socket, data["text"])
                elif data.get("type") == "location":
                    log.info("location", extra={"data": {"accuracy_m": data.get("accuracy")}})
                    request_id = data.get("id")
                    for out in session.set_location(float(data["lat"]), float(data["lon"]),
                                                    data.get("alt"), data.get("accuracy"),
                                                    int(request_id) if request_id is not None
                                                    else None):
                        await hub.broadcast(out)
                elif data.get("type") == "location_error":
                    request_id = data.get("id")
                    for out in session.location_failed(str(data.get("message", "")),
                                                       int(request_id) if request_id is not None
                                                       else None):
                        await hub.broadcast(out)
    except WebSocketDisconnect:
        pass
    finally:
        hub.leave(socket)


GALLERY = ROOT / "data" / "gallery"


@app.get("/gallery")
def gallery() -> list[str]:
    """Processed pictures, newest first."""
    files = sorted(GALLERY.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


app.mount("/pictures", StaticFiles(directory=GALLERY, check_dir=False), name="pictures")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
