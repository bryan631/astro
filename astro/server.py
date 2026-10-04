"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import json
import os
import time
import tomllib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from astro.agent import Agent
from astro.devices.sim.scope import SimScope, SimUser
from astro.pointing.coords import Site
from astro.session import Session
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
    with (ROOT / "config" / "site.toml").open("rb") as f:
        cfg = tomllib.load(f)
    return Site(cfg["lat_deg"], cfg["lon_deg"], cfg.get("elevation_m", 0.0))


def build_session() -> tuple[Session, SimScope | None]:
    if not SIM:
        raise RuntimeError("Only sim mode exists so far; set ASTRO_SIM=1")
    scope = SimScope(45, 180)
    override = os.environ.get("ASTRO_DEV_OVERRIDE") == "1"
    return Session(load_site(), lambda: (scope.alt, scope.az), developer_override=override), scope


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
            for msg in session.tick(t):
                if user and msg["type"] == "say":
                    user.hear(msg["text"], t)
                await send(msg)
            if user and scope:
                scope.step(*user.act(t), TICK_S)
            await asyncio.sleep(TICK_S)

    await socket.send_json({"type": "hello", "server_stt": stt.available(),
                            "server_tts": tts.available()})
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
    except WebSocketDisconnect:
        pass
    finally:
        loop.cancel()


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
