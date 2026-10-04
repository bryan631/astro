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

## Tablet (Android) and HTTPS
The mic needs HTTPS. On the MiniPC, use Tailscale certs (enable MagicDNS + HTTPS in the admin
console), then `scripts/tailscale-cert.sh`; it prints the tablet URL. Install the tablet in the
tailnet, open the URL in Chrome, and "Add to Home screen".

## Speech
`scripts/install-voice.sh` (MiniPC) builds whisper.cpp and installs Piper, writing `voice.env`.
The server then announces `server_stt/server_tts` and the tablet records audio for the server
(works offline). Without it the tablet uses Chrome's recognizer, which may need internet.

## MiniPC services
    sudo cp deploy/*.service deploy/*.timer /etc/systemd/system/
    sudo systemctl enable --now astro astro-cert.timer
`scripts/run.sh` serves HTTPS on port 8443 when `certs/` has a cert, plain HTTP otherwise.
