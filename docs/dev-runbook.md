# Developer runbook

## Run locally (simulators, no hardware)

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    ASTRO_SIM=1 .venv/bin/pytest -q
    ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0 --port 8000

Open http://localhost:8000. Type or say "what's good tonight", then "next". In sim mode a
simulated user follows the spoken cues so you can watch the arrow and hear the guidance.

- `ASTRO_DEV_OVERRIDE=1` lifts the daytime lockout (the 20° Sun exclusion always applies).
- `ANTHROPIC_API_KEY` in `.env` enables the Claude agent for free-form questions; without it
  (or offline) the built-in command grammar still handles goto/stop/next/tonight/where.
- The microphone needs HTTPS on anything but localhost (see "Tablet" below).

## Layout

| Path | What |
|---|---|
| `astro/pointing/` | coordinates, encoders, mount model, geometry |
| `astro/guidance/` | push-to cue engine |
| `astro/planner/` | tonight's targets, horizon mask, weather |
| `astro/capture/` | SER writer, planet ROI, focus metrics |
| `astro/devices/` | device interfaces; `sim/` fakes; real drivers go beside them |
| `astro/session.py` | ties it together; `agent.py` adds Claude; `server.py` serves the PWA |
| `config/` | `site.toml`, `targets.toml` |
| `web/` | tablet PWA |

## Safety rules (do not weaken)
- `astro/safety.py::check_target` runs before guiding and every second during guidance.
- Sun exclusion is ≥20° and cannot be narrowed or overridden.

## Tablet (prototype)
Speech in/out currently uses the tablet browser's built-in recognizer and voices. On Android
Chrome the recognizer may need internet; whisper.cpp + Piper on the server replace this next.
