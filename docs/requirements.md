# astro: requirements

The current requirements for the project. Each has an ID for reference.

## 1. Mission and users

- **M1** Convert a 2012 Orion SkyQuest XT8i (8" f/5.9, 1200 mm, manual alt-az) into a voice-controlled push-to and imaging rig.
- **M2** End user is 77, with weakening eyesight; never needs a keyboard, terminal or config file.
- **M3** Developer builds and tests remotely, installs on site later, supports remotely.
- **M4** Location: West Palm Beach suburbs (Bortle 7-8, very humid). Primary targets: Moon, planets, bright compact deep-sky objects; no tracking in phase 1.

## 2. Hard requirements

- **H1** All software is open source or in this repo. The only exception is a closed vendor driver/SDK with no open-source alternative (today the SVBony camera SDK; the ZWO SDK if the fallback camera is used). The Claude API is a cloud service used only for free-form conversation (see H5).
- **H2** Voice-first, large-text tablet UI; red night mode; works one-handed.
- **H3** Pointing guidance is deterministic local code, never the LLM. The LLM only interprets intent, calls tools, and narrates.
- **H4** Never guide toward the Sun: exclusion zone >= 20 deg; daytime lockout for guidance and capture unless an explicit developer override is set.
- **H5** Basic commands keep working with no internet.
- **H6** Remote support via Tailscale (SSH + HTTPS).

## 3. Safety (beyond H4)

- **S1** Safety re-checked every second during guidance; guidance stops if the target becomes unsafe.
- **S2** Every camera exposure is safety-checked: focus steps (finder and main), recording start and every second, live-stack start and every second, centering frames.
- **S3** Before the first sync (pointing unknown), exposures are allowed only with the Sun down.
- **S4** Refuse targets below the local horizon **mask** (trees), not just below 0 deg.
- **S5** Heaters fail safe: off on lost host heartbeat (10 s), on a failed sensor reading, and on host shutdown.

## 4. Hardware

- **HW1** Main camera SVBONY SV705C (IMX585); prefer 8-bit RAW + ROI; avoid 16-bit initially.
- **HW2** Finder camera SVBONY SV905C + 25 mm f/1.4 C-mount (5 mm CS-C ring), run at f/2-2.8, ~11x8 deg.
- **HW3** Computer MeLE Quieter 4C (N150), Ubuntu 24.04 LTS, USB-C PD power.
- **HW4** Arduino Nano Every reads the IntelliScope encoders, drives dew heaters, reads BME280 (+ optional DS18B20).
- **HW5** Encoders: 9216 counts/rev per axis, read as 5 V TTL quadrature by the Nano Every. Fallback: the IntelliScope handset over RS-232 ("Q" query).
- **HW6** Two 5 V USB dew heaters via logic-level MOSFETs on PWM pins, powered from the power bank.
- **HW7** Hardware abstraction: different computers differ only at the lowest driver level.
- **HW8** Each device has a simulator implementing the same interface; everything runs with no hardware.
- **HW9** Real-hardware mode without encoders: pointing from continuous finder plate solves.
- **HW10** The camera driver hides SDK quirks: it recovers from timeouts, applies settings changes reliably, and never returns the blank frames the SDK produces after starting.
- **HW11** Camera temperature reported if available.

## 5. Optics and pointing math

- **O1** Optics math in `astro/optics.py`, unit-tested: plate scale = 206.265 x pixel_um / focal_mm; main 0.50"/px (0.25"/px with 2x Barlow); finder ~31"/px; sidereal drift ~30 px/s at prime focus.
- **O2** RA/Dec <-> Alt/Az for site and time, with refraction; works offline.
- **O3** Encoder counts -> mount angles, counts/rev and direction configurable.
- **O4** Mount model from plate-solve syncs: az zero, alt zero, 2-axis base tilt from >= 2 syncs; refined by least squares.
- **O5** Finder -> main camera offset (d_alt, d_az), auto-updated whenever both see the target.

## 6. Plate solving (finder)

- **PS1** Solve a single finder frame with tetra3/cedar-solve for an ~11x8 deg field; return RA/Dec/roll + confidence.
- **PS2** Show a log-confidence level.
- **PS3** Show the camera rotation and the angle subtended per pixel, along with RA/Dec.
- **PS4** Real-sky test from the developer's yard: solve rate, time, failure modes (clouds, trees, streetlights).
- **PS5** Real test images of success and failure cases (lens cap, trees, positions/orientations, defocus both ways, etc.) kept as regression tests.
- **PS6** Star detection and solving are robust on real frames: hot pixels, quantized 8-bit noise, and saturated bright stars or planets don't cause false stars, false focus verdicts, or missed solves.

## 7. Pre-flight checks

- **F1** Finder focus gate before solving; voice-guided refocus; say why a solve failed (no stars, not the sky, few stars, out of focus, no match).
- **F2** Main-camera focus gate at session start and after a Barlow change, before any capture, using the focus coach.
- **F3** Collimation check: defocused-star ring evenness and secondary-shadow centering, spoken guidance for the primary's screws; at first light, periodically, and after transport.

## 8. Guidance

- **G1** ~10 Hz loop: current vs target alt/az -> cue stream.
- **G2** Coarse (degrees) -> fine (arcmin) -> "stop" within tolerance (4' prime focus, 2' with Barlow); hysteresis; rate limiting; never talks over itself.
- **G3** "Left/right" as seen by the user; direction sign learned on the first push.
- **G4** Arrow + distance for the UI, and an optional beep rate for proximity.
- **G5** Main-camera centering ("center it") after the finder says on target; learns and applies the finder-to-main offset.

## 9. Planner

- **PL1** "What's good tonight": visible window, altitude, Moon separation, weather (Open-Meteo), horizon mask, Bortle-8 suitability (curated list with notes).
- **PL2** Ranked choices per category: planet, Moon and Moon features, nebula, cluster, galaxy, double star.
- **PL3** Cloud cover mentioned when >= 50%; cached per site; offline-safe.
- **PL4** Horizon walk: "mark" along the treeline; saved mask + Stellarium export.

## 10. Voice, tablet and agent

- **V1** Tablet PWA: push-to-talk, arrows, live view panel, gallery, red night mode.
- **V2** HTTPS on the LAN (self-signed + install instructions, or Tailscale certs), required for the mic.
- **V3** STT whisper.cpp, TTS Piper, audio over WebSocket.
- **V4** Speech is fast enough for guidance: cues like "stop" are spoken without noticeable delay, and commands are understood within about a second.
- **V5** Claude agent tools: list_tonight, describe(target), goto, stop, where_am_i, what_am_i_looking_at, start_capture/stop_capture, focus_assist, set_location, calibrate_horizon, session_status. Short, warm, plain replies.
- **V6** Offline command grammar (no internet or LLM needed): go to, stop, next, what's good tonight, where am I, sync, focus (main/finder), take a picture / stop recording, Barlow in/out, use my location, horizon walk / mark / done, set up the telescope / ready / skip.
- **V7** Tablet GPS sets the site during setup; saved locally (never in the repo or sent to the LLM); a move > 1 km resets the mount model.
- **V8** One guidance loop per session, broadcast to every connected tablet.
- **V9** Setup wizard: "set up the telescope" -> GPS -> 3 syncs with an alignment report -> horizon walk.

## 11. Capture and processing

- **C1** SER video writer with an ROI that follows the brightest blob.
- **C2** Focus metric (HFR for stars, Laplacian variance for planets) -> voice "sharper / passed it".
- **C3** Planet pipeline: stack -> sharpen -> RGB align -> auto-crop -> PNG.
- **C4** Deep-sky pipeline: register + live-stack short subs with rotation/drift -> GraXpert background/denoise -> Siril color calibration + stretch.
- **C5** Validate with downloaded sample SER/FITS data.
- **C6** Gallery of processed pictures on the tablet; spoken "your picture is ready".

## 12. Phase 2 (MiniPC + Arduino)

- **P2-1** Ubuntu on the MeLE; `scripts/setup.sh`; systemd services; auto-start; BIOS auto-power-on; Tailscale; WiFi + fallback hotspot; log rotation.
- **P2-2** Firmware: interrupt quadrature; line protocol `POS`, `ENV`, `HEAT`, `ZERO`, `VER`; checksum; watchdog; dew control from BME280 (Magnus) keeping optics >= 2-3 C above dew point; heater failsafe.
- **P2-3** Handset fallback driver with the same interface as the MCU encoder driver.
- **P2-4** Performance on the N150: STT latency, solve time, live-stack frame rate, processing time.
- **P2-5** Power test: 1 h of stacking + both cameras + heaters on the power bank; no reboot on plug/unplug.
- **P2-6** Field test in the developer's yard: finder on a tripod with the MeLE, full voice loop, plate solving.

## 13. Phase 3 (on the telescope)

- **P3-1** Encoder bring-up: pinout/voltage, quadrature, counts/rev, direction (handset fallback if needed).
- **P3-2** Mechanical: finder bracket, heater on the secondary, strain relief, balance, dolly.
- **P3-3** Calibration wizard: location, solves, finder-main offset, horizon walk (+ Stellarium), focus reference positions (prime/Barlow).
- **P3-4** First-light checklist: back-focus reach, collimation, solve reliability, guidance feel, dew, battery runtime.
- **P3-5** Teach and tune: prompts, cue pacing, UI size for the actual user.
- **P3-6** Remote support: Tailscale access confirmed from the developer's home.

## 14. Reliability and observability

- **R1** Log everything as structured JSON; keep the last N sessions.
- **R2** A failing guidance tick never ends guidance; camera failures are spoken, never crash the session.
- **R3** Hardware opened once and rolled back if setup fails part-way; drivers validated before touching hardware.
- **R4** Only one user of a camera at a time (recording, stacking, focus, centering, viewers).
- **R5** Tests never touch the network (weather, Claude API); captures, pictures, site, horizon and logs from tests go to temporary directories.
- **R6** Works on the developer's laptop and the MiniPC; no OS-specific paths.
- **R7** Nightly log upload for remote support.

## 15. Conventions and repository

- **CV1** Monorepo: `astro/` (Python package), `web/`, `firmware/`, `scripts/`, `config/`, `deploy/`, `docs/`, `tests/`.
- **CV2** Config in `config/*.toml`; secrets (ANTHROPIC_API_KEY) in `.env`, never committed. Per-install data (site, horizon, captures, logs) in git-ignored `data/`.
- **CV3** `ASTRO_SIM=1` runs everything against simulators; all tests pass in sim mode.
- **CV4** Hardware tools: browser viewer (`viewer.py`) and matplotlib viewer (`viewer_plot.py`) sharing common code; exposures in seconds (values > 10 rejected); display rotated 180 deg (lens inversion) with `--raw`; scripts run the project venv from their shebang.
- **CV5** Hardware checkout results recorded.
- **CV6** Docs: one-page large-print user guide and a developer runbook.
- **CV7** Store: SQLite for locations, horizon masks, sessions, images, calibration.

## 16. Development process

- **D1** Clean, concise, reviewable code; no overly complex designs.
- **D2** Open-source libraries are welcome; no binaries beyond the H1 exception.
- **D3** Commit often; modern CI/CD.
- **D4** Good unit tests, not excessive.
- **D5** Prototype first, then productize.
- **D6** Propose code-hygiene steps after long tangents.
- **D7** Every change goes through a PR with GitHub Copilot review; feedback is addressed before merge; `main` is protected.
- **D8** Annotated git tag when the developer confirms something works on real hardware ("ship it!").
- **D9** Token-efficient work.
- **D10** American English in code, comments and docs.
- **D11** Parallel sessions use separate git worktrees; one session merges a branch.
- **D12** Phase 1 is done when, in sim mode, a full session works end to end via the tablet (voice -> plan -> guide -> simulated capture -> processed image), and the finder solves real sky.
