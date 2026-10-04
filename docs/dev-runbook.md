# Developer runbook

## Run locally (simulators, no hardware)

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && scripts/install-solver.sh
    ASTRO_SIM=1 .venv/bin/pytest -q
    ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0 --port 8000

Open http://localhost:8000. In sim mode the mount starts uncalibrated: the first goto
plate-solves the simulated finder (it renders the real sky) to learn where the scope points, and
a simulated user follows the spoken cues so you can watch the arrow and hear the guidance.

Voice/text commands (offline grammar; Claude handles anything vaguer):

| Say | Does |
|---|---|
| what's good tonight / next | ranked targets for the next 4 hours / go to the next one |
| go to *name* / stop | push-to guidance (syncs first if needed) / stop |
| sync / where am I | plate-solve the finder / name the nearest target |
| focus the finder / focus / done | voice focus coach for finder or main camera / accept |
| take a picture / stop recording | planets & Moon: SER video then a stacked PNG; others: live stack |
| Barlow in / Barlow out | resets the focus gate, guidance tolerance 4' to 2' |
| use my location | tablet GPS, saved to `data/site.toml` (git-ignored) |
| start the horizon walk / mark / done | record the treeline; saved to `data/horizon.toml` (+ Stellarium file) |
| set up the telescope / ready / skip | first-time setup: GPS, 3 syncs (reports alignment), horizon walk |

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
| `astro/capture/` | SER writer, planet ROI, focus metrics, recorder, live stacker |
| `astro/process/` | planet lucky-imaging stack, deep-sky live stack |
| `astro/devices/` | device interfaces, SVBony driver; `sim/` fakes (finder sky, main camera) |
| `astro/site_store.py` | observing site: tablet GPS overrides `config/site.toml` |
| `scripts/hwcheck/` | hardware tools: `viewer.py` (browser), `viewer_plot.py`, `solve_sky.py`, checklist |
| `data/` | per-install, git-ignored: `site.toml`, `captures/*.ser`, `gallery/*.png` |
| `astro/session.py` | ties it together; `agent.py` adds Claude; `server.py` serves the PWA |
| `config/` | `site.toml`, `targets.toml` |
| `web/` | tablet PWA |

## Hardware tools

    scripts/hwcheck/viewer.py finder --exp 0.8 --gain 200     # browser view, http://localhost:8080
    scripts/hwcheck/viewer_plot.py finder --exp 0.8 --gain 200
    scripts/hwcheck/solve_sky.py                              # 0.8 s, gain 200; Ctrl+C for summary

`--exp` is seconds. Only one program can use a camera at a time. Results go in
`docs/hardware-results.md`.

## Safety rules (do not weaken)
- `astro/safety.py::check_target` runs before guiding and every second during guidance, and
  `Session.exposure_safety` before every focus exposure and every second of a recording or
  live stack. Before the first sync (pointing unknown) exposures are allowed only with the Sun down.
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
