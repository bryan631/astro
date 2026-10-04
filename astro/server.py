"""FastAPI server: serves the tablet PWA and a WebSocket for voice/text and guidance.

Run:  ASTRO_SIM=1 uvicorn astro.server:app --host 0.0.0.0 --port 8000
In sim mode a simulated user follows the spoken cues so the whole loop can be watched.
"""

import asyncio
import os
import time
import tomllib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from astro.devices.sim.scope import SimScope, SimUser
from astro.pointing.coords import Site
from astro.session import Session

ROOT = Path(__file__).resolve().parents[1]
SIM = os.environ.get("ASTRO_SIM") == "1"
TICK_S = 0.1


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


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    session, scope = build_session()
    user = SimUser() if scope else None
    t0 = time.monotonic()

    async def guidance_loop() -> None:
        while True:
            t = time.monotonic() - t0
            for msg in session.tick(t):
                if user and msg["type"] == "say":
                    user.hear(msg["text"], t)
                await socket.send_json(msg)
            if user and scope:
                scope.step(*user.act(t), TICK_S)
            await asyncio.sleep(TICK_S)

    loop = asyncio.create_task(guidance_loop())
    try:
        while True:
            msg = await socket.receive_json()
            if msg.get("type") == "text":
                for out in await asyncio.to_thread(session.handle, msg["text"]):
                    await socket.send_json(out)
    except WebSocketDisconnect:
        pass
    finally:
        loop.cancel()


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
