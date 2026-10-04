"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import functools
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from astro import site_store
from astro.agent import Agent
from astro.devices.sim.finder import SimFinderCamera
from astro.devices.sim.main_cam import SimMainCamera
from astro.devices.sim.scope import SimEncoders, SimScope, SimUser
from astro.pointing.coords import Site
from astro.pointing.finder_sync import FinderSync
from astro.pointing.mount_model import MountModel
from astro.pointing.platesolve import FinderSolver
from astro.session import Session, utcnow
from astro.voice.speech import Stt, Tts

ROOT = Path(__file__).resolve().parents[1]
SIM = os.environ.get("ASTRO_SIM") == "1"
TICK_S = 0.1


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=value lines) for secrets like ANTHROPIC_API_KEY."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


load_env()


def load_site() -> Site:
    return site_store.load(ROOT)


def build_session() -> tuple[Session, SimScope | None]:
    """Sim mode: an uncalibrated simulated mount with a finder that sees the real sky."""
    if not SIM:
        raise RuntimeError("Only sim mode exists so far; set ASTRO_SIM=1")
    site, clock = load_site(), utcnow
    scope = SimScope(45, 180)
    camera = SimFinderCamera(lambda: (scope.alt, scope.az), site, clock, solver()._t3.star_table)
    finder = FinderSync(camera, solver(), MountModel(), SimEncoders(scope), site, clock)
    override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
    main = SimMainCamera(lambda: (scope.alt, scope.az), site, clock)
    session = Session(site, clock=clock, developer_override=override, finder=finder,
                      main_camera=main, main_sensor=main.sensor_size, data_dir=ROOT / "data",
                      on_site_change=lambda s: on_site_change(s, camera, main))
    return session, scope


def on_site_change(site: Site, finder_cam: SimFinderCamera, main_cam: SimMainCamera) -> None:
    """Persist a GPS fix and move the simulated sky with it."""
    site_store.save(ROOT, site)
    finder_cam.site = site
    main_cam.set_site(site)


@functools.cache
def solver() -> FinderSolver:
    return FinderSolver()  # loads the star database once per process


app = FastAPI()
stt, tts = Stt(), Tts()


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    session, scope = build_session()
    agent = Agent(session)
    user = SimUser() if scope else None
    t0 = time.monotonic()

    async def send(msg: dict) -> None:
        await socket.send_json(msg)
        if msg["type"] == "say" and tts.available():
            await socket.send_bytes(await asyncio.to_thread(tts.synthesize, msg["text"]))

    async def handle_text(text: str) -> None:
        await socket.send_json({"type": "heard", "text": text})
        for out in await asyncio.to_thread(agent.handle, text):
            await send(out)

    async def guidance_loop() -> None:
        while True:
            t = time.monotonic() - t0
            for msg in await asyncio.to_thread(session.tick, t):  # camera calls block
                if user and msg["type"] == "say":
                    user.hear(msg["text"], t)
                await send(msg)
            if user and scope:
                scope.step(*user.act(t), TICK_S)
            await asyncio.sleep(TICK_S)

    await socket.send_json({"type": "hello", "server_stt": stt.available(),
                            "server_tts": tts.available()})
    if not site_store.has_saved(ROOT):  # setup: first run at this installation
        for out in session.request_location():
            await send(out)
    loop = asyncio.create_task(guidance_loop())
    try:
        while True:
            msg = await socket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):  # recorded speech from the tablet
                text = await asyncio.to_thread(stt.transcribe, msg["bytes"])
                await handle_text(text) if text else await send(
                    {"type": "say", "text": "Sorry, I didn't hear anything."})
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "text":
                    await handle_text(data["text"])
                elif data.get("type") == "location":
                    for out in session.set_location(float(data["lat"]), float(data["lon"]),
                                                    data.get("alt"), data.get("accuracy")):
                        await send(out)
                elif data.get("type") == "location_error":
                    await send({"type": "say", "text": "I couldn't get the tablet's location. "
                                f"{data.get('message', '')} Using the saved location for now."})
    except WebSocketDisconnect:
        pass
    finally:
        loop.cancel()


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
