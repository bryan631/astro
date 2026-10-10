# Developer runbook

## Run locally (simulators, no hardware)

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && scripts/install-solver.sh
    ASTRO_SIM=1 .venv/bin/pytest -q
    ASTRO_SIM=1 .venv/bin/uvicorn astro.server:app --host 0.0.0.0 --port 8000

Open http://localhost:8000 (landscape). Both camera views are always live: each camera streams
continuously (`astro/devices/stream.py`: a process per SVBony camera on the Mele, a paced thread
for the simulators) and every reader shares it. In sim mode the mount starts uncalibrated: the
first Go to plate-solves the simulated finder (it renders the real sky), and a simulated user
follows the guidance so you can watch the arrow.

The page works with the volume off (Phase 3b, docs/plan.md). Buttons send `{type: "action"}`:

| Button | Does |
|---|---|
| Go to | list from `/api/targets` (tonight, best first); guidance with an arrow on the finder view |
| Align | bright star in the main view: 10 s of both cameras -> main box on the finder view and the finder-to-main offset (`astro/pointing/align.py`) |
| Focus | shows the focus number large with a trend; 100 / median HFR over all usable stars (`measure_focus`) |
| Capture | start / stop: planets & Moon a SER video, others a live stack; progress and time to edge |
| Recenter | pause the capture, guide back to its target, Resume continues the same video or stack |
| Pictures, More | gallery and viewer; debug, day mode, Barlow, Sync now, Tonight, treeline |
| STOP | stops guidance or a capture |

Voice is not wired in; `astro/voice/`, `wake.py` and `intents.py` stay as libraries. Typed text
(`{type: "text"}`) still goes through the agent and the command grammar.

- `ASTRO_DEV_OVERRIDE=1` lifts the daytime lockout (the 20° Sun exclusion always applies).
- `ANTHROPIC_API_KEY` in `.env` enables the Claude agent for free-form questions; without it
  (or offline) the built-in command grammar handles exact commands (goto, stop, next, where,
  tonight). With a key, it still answers those offline-first; only vague requests go to Claude.
- Tablet GPS (first run at a new place) needs HTTPS on anything but localhost (see "Tablet" below).

## Layout

| Path | What |
|---|---|
| `astro/pointing/` | coordinates, encoders, mount model, plate solving, Align, star names on the finder |
| `astro/guidance/` | push-to cue engine |
| `astro/planner/` | tonight's targets, horizon mask, weather; `sky.py`: the session's target answers |
| `astro/capture/` | SER writer, planet ROI, focus metrics, view exposure, recorder, live stacker, collimation |
| `astro/process/` | planet lucky-imaging stack, deep-sky live stack |
| `astro/devices/` | device interfaces, SVBony driver; `sim/` fakes (finder sky, main camera) |
| `astro/site_store.py` | observing site: tablet GPS overrides `config/site.toml` |
| `scripts/hwcheck/` | hardware tools: `viewer.py` (browser), `viewer_plot.py`, `solve_sky.py`, checklist |
| `data/` | per-install, git-ignored: `site.toml`, `captures/*.ser`, `gallery/*.png` |
| `astro/session.py` | ties it together; `agent.py` adds Claude; `server.py` serves the PWA |
| `data/logs/` | one JSON-lines log per server run (newest 20): events, errors with tracebacks, crashed threads |
| `astro/wizard.py` | first-time setup steps (GPS, syncs, horizon walk) |
| `astro/voice/` | offline speech: whisper.cpp (in) and Piper (out); not wired in (Phase 3b) |
| `astro/devices/stream.py` | always-on cameras: a capture process (SVBony) or thread (sim) each |
| `astro/calibration_store.py` | calibration that survives a restart (`data/calibration.json`) |
| `firmware/` | Arduino Nano Every: encoders, BME280 and DS18B20, dew heaters |
| `deploy/` | systemd units (server, monthly cert renewal), installed by `setup.sh --minipc` |
| `config/` | `site.toml`, `targets.toml`, `devices.toml` (which drivers: sim, svbony, mcu, handset) |
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

## Speech (not wired in since Phase 3b)
The voice code stays for later: `astro/voice/` (whisper.cpp in, Piper out, set up by
`scripts/install-voice.sh`), `astro/wake.py`, `astro/intents.py`, `scripts/install-vad.sh` and
`scripts/dev/voice_cases.py`. Reconnecting it means restoring the server's audio handling and the
page's talk button (see git history before Phase 3b).

## MiniPC services
    sudo cp deploy/*.service deploy/*.timer /etc/systemd/system/
    sudo systemctl enable --now astro astro-cert.timer astro-logs.timer
`scripts/run.sh` serves HTTPS on port 8443 when `certs/` has a cert, plain HTTP otherwise.
8443 is the real port (the service runs `run.sh`); 8000 is only the dev command above.

Platform defaults are all overridable (R6): the SVBony SDK path (`SVB_LIB`), the MCU serial
port (`port` in `devices.toml`, else the first `/dev/ttyACM*`, then `/dev/ttyUSB*`), and the repo path and user in
`deploy/*.service` (rewritten by `setup.sh --minipc`).

## Logs for remote support
The server writes JSON logs to `data/logs/` (one file per run, newest 20 kept). Every day at noon,
or at the next boot if the Mele was off, `astro-logs.timer` pushes them, gzipped, with a
`check.sh` report to the private repo `bryan631/astro-logs` (`<hostname>/`). It uses a deploy key
that can write to that repo only: run `scripts/upload-logs.sh --setup` once on the Mele and add
the key it prints (the command is in the script's header). Read them without the Mele:

    gh repo clone bryan631/astro-logs ~/astro-logs   # later: git -C ~/astro-logs pull
    zcat ~/astro-logs/mele/astro-*.jsonl.gz | jq -c 'select(.level != "INFO")'

The repo must stay private: logs hold the site's location and what was said.

## Field use: HTTPS with no internet, and access token

The tablet's location (GPS) needs HTTPS, and the cert is for a Tailscale `*.ts.net` name.
Tailscale caches its peer map and MagicDNS answers locally, so an already-connected tablet
and MeLE should still reach each other on a hotspot with no internet. This is **not yet
verified**. To check it: connect both devices to the phone hotspot with mobile data off, then
open the tablet URL. If the page doesn't load, a self-signed LAN cert is the fallback (V2;
not implemented).

On public WiFi, set `ASTRO_TOKEN=<random>` in `.env` and open the app once as
`https://<name>:8443/?token=<random>`. A cookie remembers the token after that.

## Spots and the online "Tonight" page
Each spot (name, place, treeline) lives in the Mele's git-ignored `data/spots.toml`, recorded with
the tablet (`/treeline.html`, compass and tilt, about 5 degrees; the server turns the compass's magnetic north
to true north with the World Magnetic Model, `pygeomag`) or a telescope treeline walk at
the current spot (more exact; it replaces the tablet's). `/tonight` on the Mele renders the
report. Online: `scripts/publish-spots.sh` copies the spots into the `SPOTS_TOML` repo secret and
runs the `Tonight` workflow, which rebuilds the page on GitHub Pages at 1, 4, 7 and 9 PM Eastern.
Coordinates stay in the secret; the page shows only targets, times and cloud.
