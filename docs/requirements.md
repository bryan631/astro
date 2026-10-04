# astro: requirements

The consolidated requirements for the project, used to judge it. Sources:

- **P**: the original plan (`docs/plan.md`, from `ATSTRO_initial_plan.md`).
- **U**: requests made by the developer during development (sessions of 2026-10-03/04).
- **A**: additions made during development (review findings, real-hardware findings, safety).

Status as of 2026-10-04 (`main` after PR #26):

| Status | Meaning |
|---|---|
| **Done** | Implemented and covered by tests (simulation and/or recorded real data). |
| **Done (HW)** | Also verified on real hardware. |
| **Needs HW** | Implemented, but only verifiable on hardware not yet available (MiniPC, Arduino, telescope). |
| **Partial** | Some of the requirement is met; the gap is stated. |
| **Not done** | Not implemented yet. |

---

## 1. Mission and users

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| M1 | Convert a 2012 Orion SkyQuest XT8i (8" f/5.9, 1200 mm, manual alt-az) into a voice-controlled push-to and imaging rig. | P | Partial | All software paths exist in simulation; telescope integration is Phase 3. |
| M2 | End user is 77, with weakening eyesight; never needs a keyboard, terminal or config file. | P | Partial | Voice + large-text PWA; setup wizard and GPS remove config. Developer still configures `devices.toml` once per install. |
| M3 | Developer builds and tests remotely, installs on site later, supports remotely. | P | Partial | Tailscale/systemd prep, JSON logs; no nightly log upload yet (see R7). |
| M4 | Location: West Palm Beach suburbs (Bortle 7-8, very humid). Primary targets: Moon, planets, bright compact deep-sky objects; no tracking in phase 1. | P | Done | Curated Bortle-8 target list, humid-night dew floor, untracked live stacking. |

## 2. Hard requirements

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| H1 | All software open source or in this repo; exception: vendor camera SDK binaries (SVBony). | P, U | Done | Python deps are open source (numpy, scipy, astropy, cedar-solve, astroalign, pyserial, FastAPI, Piper, whisper.cpp). The Claude API is a cloud service used only for free-form conversation (H5). |
| H2 | Voice-first, large-text tablet UI; red night mode; works one-handed. | P | Done | `web/index.html`: hold-to-talk circle, 28-40 px text, red default with day toggle. |
| H3 | Pointing guidance is deterministic local code, never the LLM. The LLM only interprets intent, calls tools, and narrates. | P | Done | `astro/guidance/engine.py`; agent tools only start/stop modes (`astro/agent.py`). |
| H4 | Never guide toward the Sun: exclusion zone >= 20 deg; daytime lockout for guidance and capture unless an explicit developer override is set. | P, U | Done | `astro/safety.py`; exclusion can't be narrowed or overridden (U: "it can always apply"); override (`ASTRO_DEV_OVERRIDE=1`) lifts only the daytime lockout. |
| H5 | Basic commands keep working with no internet. | P | Done | Offline grammar (`astro/intents.py`) for goto/stop/next/tonight/where/sync/focus/capture/Barlow/location/horizon/setup; Claude only for vague requests; local STT/TTS. |
| H6 | Remote support via Tailscale (SSH + HTTPS). | P | Needs HW | `scripts/tailscale-cert.sh`, `deploy/astro-cert.*`; untested until the MiniPC. |

## 3. Safety (beyond H4)

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| S1 | Safety re-checked every second during guidance; guidance stops if the target becomes unsafe. | A | Done | `Session._tick`. |
| S2 | Every camera exposure is safety-checked: focus steps (finder and main), recording start and every second, live-stack start and every second, centering frames. | A | Done | `Session.exposure_safety`, recorder/stacker safety callbacks; tests. |
| S3 | Before the first sync (pointing unknown), exposures are allowed only with the Sun down. | A | Done | `exposure_safety`. |
| S4 | Refuse targets below the local horizon **mask** (trees), not just below 0 deg. | P | Partial | Planner hides targets below the mask; `goto` refuses only below 0 deg altitude. |
| S5 | Heaters fail safe: off on lost host heartbeat (10 s), on a failed sensor reading, and on host shutdown. | P, A | Done | Firmware part unverified on hardware. Firmware heartbeat; `dew.heater_percent` returns 0 for non-finite readings; `Mcu.close` joins workers then turns heaters off. |

## 4. Hardware

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| HW1 | Main camera SVBONY SV705C (IMX585); prefer 8-bit RAW + ROI; avoid 16-bit initially. | P | Done (HW) | `astro/devices/svbony.py`, RAW8 only; 640x480 ROI at 130 fps on the laptop. |
| HW2 | Finder camera SVBONY SV905C + 25 mm f/1.4 C-mount (5 mm CS-C ring), run at f/2-2.8, ~11x8 deg. | P | Done (HW) | Measured field 10.39 deg wide by plate solving. |
| HW3 | Computer MeLE Quieter 4C (N150), Ubuntu LTS, USB-C PD power. | P, U | Needs HW | Ubuntu **24.04 LTS** chosen (U). |
| HW4 | Arduino Nano Every reads the IntelliScope encoders, drives dew heaters, reads BME280 (+ optional DS18B20). | P | Needs HW | `firmware/astro_mcu/` compiles in CI (41% flash). |
| HW5 | Encoders: 9216 steps/rev per axis; believed 5 V TTL quadrature (unverified). Fallback: IntelliScope handset over RS-232 ("Q" query). | P | Needs HW | Interrupt quadrature in firmware; `astro/devices/handset.py` (reply format unverified). |
| HW6 | Two 5 V USB dew heaters via logic-level MOSFETs on PWM pins, powered from the power bank. | P | Needs HW | Firmware D9/D10 PWM. |
| HW7 | Hardware abstraction: different computers differ only at the lowest driver level. | U | Done | `astro/devices/base.py` protocols; sim and real drivers behind them; `config/devices.toml` selects real drivers. |
| HW8 | Each device has a simulator implementing the same interface; everything runs with no hardware. | P | Done | `astro/devices/sim/`: finder renders the real sky, main camera renders planets/Moon/stars, uncalibrated encoders, simulated user. |
| HW9 | Real-hardware mode without encoders: pointing from continuous finder plate solves. | A | Done | Smoke-tested on the laptop with the real finder. Mount driver `solve` (`astro/pointing/solve_tracker.py`); for the Phase 2 tripod test and as an encoder fallback. |
| HW10 | Camera driver handles SDK quirks: settings changes and timeouts reopen; system libusb loaded first; the 2 blank frames after each start are discarded. | A | Done (HW) | Found during hardware checkout and the exposure sweep. |
| HW11 | Camera temperature reported if available. | P | Not done | |

## 5. Optics and pointing math

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| O1 | Optics math in `astro/optics.py`, unit-tested: plate scale = 206.265 x pixel_um / focal_mm; main 0.50"/px (0.25"/px with 2x Barlow); finder ~31"/px; sidereal drift ~30 px/s at prime focus. | P | Done | `tests/test_optics.py`. |
| O2 | RA/Dec <-> Alt/Az for site and time, with refraction; works offline. | P | Done | `astro/pointing/coords.py` (IERS downloads disabled). |
| O3 | Encoder counts -> mount angles, counts/rev and direction configurable. | P | Done | `EncoderAxis`; `devices.toml` sign/counts. |
| O4 | Mount model from plate-solve syncs: az zero, alt zero, 2-axis base tilt from >= 2 syncs; refined by least squares. | P | Done | `astro/pointing/mount_model.py`; fix for the zenith-flipped single-sync solution. |
| O5 | Finder -> main camera offset (d_alt, d_az), auto-updated whenever both see the target. | P | Done | `astro/pointing/main_offset.py`, `astro/guidance/centering.py`: learned from the main camera image, not a second solver. |

## 6. Plate solving (finder)

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| PS1 | Solve a single finder frame with tetra3/cedar-solve for an ~11x8 deg field; return RA/Dec/roll + confidence. | P | Done (HW) | First real solve 2026-10-03 (Saturn field): 10 matches, 12 ms. |
| PS2 | Show a log-confidence level. | U | Done | `conf` = -log10(false-match probability). |
| PS3 | Show the camera rotation and the angle subtended per pixel, along with RA/Dec. | U | Done | `rotation` and `scale` ("/px) in `solve_sky.py` output. |
| PS4 | Real-sky test from the developer's yard: solve rate, time, failure modes (clouds, trees, streetlights). | P | Partial | Exposure sweep done (solves at >= 0.8 s, gain >= 100); a full session across the sky is still to run. |
| PS5 | Real test images of success and failure cases (lens cap, trees, positions/orientations, defocus both ways, etc.) kept as regression tests. | U, P | Partial | Capture tool, manifest and test exist (`scripts/hwcheck/capture_cases.py`, `tests/test_real_frames.py`); 3 real frames recorded; the 18-case shot list (`docs/test-frames.md`) is not yet recorded. |
| PS6 | Robust on real frames: hot pixels removed, noise floor when the MAD is 0, single-pixel blobs ignored, saturated stars skipped for HFR; tetra3 binary opening disabled. | A | Done (HW) | Found on, and tested with, real SV905C frames. |

## 7. Pre-flight checks

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| F1 | Finder focus gate before solving; voice-guided refocus; say why a solve failed (no stars, not the sky, few stars, out of focus, no match). | U, P | Done | `check_focus`, "focus the finder". Thresholds tuned on simulation + 3 real frames. |
| F2 | Main-camera focus gate at session start and after a Barlow change, before any capture, using the focus coach. | U, P | Done | `Session.capture`. |
| F3 | Collimation check: defocused-star ring evenness and secondary-shadow centering, spoken guidance for the primary's screws; at first light, periodically, and after transport. | U, P | Not done | Planned for Phase 3. |

## 8. Guidance

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| G1 | ~10 Hz loop: current vs target alt/az -> cue stream. | P | Done | `astro/guidance/engine.py`. |
| G2 | Coarse (degrees) -> fine (arcmin) -> "stop" within tolerance (4' prime focus, 2' with Barlow); hysteresis; rate limiting; never talks over itself. | P | Done | Barlow switches tolerance, including an active guide. |
| G3 | "Left/right" as seen by the user; direction sign learned on the first push. | P | Partial | Configurable sign and a `DirectionLearner` exist, but learning is not wired into the session. |
| G4 | Arrow + distance for the UI, and an optional beep rate for proximity. | P | Partial | Arrow and distance shown; beep rate is computed but the tablet does not play it. |
| G5 | Main-camera centering ("center it") after the finder says on target; learns and applies the finder-to-main offset. | A | Done | Needs >= 2-encoder-count calibration pushes (0.039 deg/count). |

## 9. Planner

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| PL1 | "What's good tonight": visible window, altitude, Moon separation, weather (Open-Meteo), horizon mask, Bortle-8 suitability (curated list with notes). | P | Partial | All listed except Moon **phase** (separation only). |
| PL2 | Ranked choices per category: planet, Moon features, nebula, cluster, double star. | P | Partial | Planet, Moon (whole), nebula, cluster, galaxy, double star; no individual Moon features. |
| PL3 | Cloud cover mentioned when >= 50%; cached per site; offline-safe. | A | Done | |
| PL4 | Horizon walk: "mark" along the treeline; saved mask + Stellarium export. | P | Done | Also step 4 of the setup wizard. |

## 10. Voice, tablet and agent

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| V1 | Tablet PWA: push-to-talk, arrows, live view panel, gallery, red night mode. | P | Partial | Live view only while live-stacking; no live planetary view. |
| V2 | HTTPS on the LAN (self-signed + install instructions, or Tailscale certs), required for the mic. | P | Partial | Tailscale certs scripted (untested until the MiniPC); no self-signed option. |
| V3 | STT whisper.cpp, TTS Piper, audio over WebSocket. | P | Done | Real round trip verified on the laptop. |
| V4 | Speech fast enough for guidance: cues pre-rendered and cached; STT sized to the utterance. | A | Done | TTS 30-80 ms (0 ms cached), STT ~0.7 s on the laptop; N150 timing pending (P2-4). |
| V5 | Claude agent tools: list_tonight, describe(target), goto, stop, where_am_i, what_am_i_looking_at, start_capture/stop_capture, focus_assist, set_location, calibrate_horizon, session_status. Short, warm, plain replies. | P | Partial | `describe(target)` missing. Named differently: `what_am_i_looking_at` is covered by `where_am_i`; `take_picture`/`stop_picture` (start/stop_capture); `focus` (focus_assist); `horizon_walk` (calibrate_horizon). Added: `next`, `sync`, `barlow`. Uses Claude Haiku 4.5. |
| V6 | Offline intent fallback: goto, stop, next, capture, focus. | P | Done | Plus sync, Barlow, location, horizon, setup. |
| V7 | Tablet GPS sets the site during setup; saved locally (never in the repo or sent to the LLM); a move > 1 km resets the mount model. | U | Done | Not yet tried on the Android tablet. `astro/site_store.py`. |
| V8 | One guidance loop per session, broadcast to every connected tablet. | A | Done | `server.Hub`. |
| V9 | Setup wizard: "set up the telescope" -> GPS -> 3 syncs with an alignment report -> horizon walk. | P, A | Done | `astro/wizard.py`. |

## 11. Capture and processing

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| C1 | SER video writer with an ROI that follows the brightest blob. | P | Done | Unique file names; failures spoken, never "Done". |
| C2 | Focus metric (HFR for stars, Laplacian variance for planets) -> voice "sharper / passed it". | P | Done | |
| C3 | Planet pipeline: stack -> sharpen -> RGB align -> auto-crop -> PNG. | P | Partial | Own numpy lucky-imaging prototype (memory-bounded, vectorized); PlanetarySystemStacker and wavelets/deconvolution not integrated. |
| C4 | Deep-sky pipeline: register + live-stack short subs with rotation/drift -> GraXpert background/denoise -> Siril color calibration + stretch. | P | Partial | Live stacking with astroalign done; GraXpert/Siril not integrated. |
| C5 | Validate with downloaded sample SER/FITS data. | P | Not done | Validated on synthetic seeing and drifting star fields only. |
| C6 | Gallery of processed pictures on the tablet; spoken "your picture is ready". | A | Done | |

## 12. Phase 2 (MiniPC + Arduino)

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| P2-1 | Ubuntu on the MeLE; `scripts/setup.sh`; systemd services; auto-start; BIOS auto-power-on; Tailscale; WiFi + fallback hotspot; log rotation. | P | Partial | Needs the MiniPC to verify. setup.sh, systemd units, Tailscale script, app log retention (20 runs); no hotspot. |
| P2-2 | Firmware: interrupt quadrature; line protocol `POS`, `ENV`, `HEAT`, `ZERO`, `VER`; checksum; watchdog; dew control from BME280 (Magnus) keeping optics >= 2-3 C above dew point; heater failsafe. | P | Needs HW | Dew control runs on the host (stepped power). Encoder test at 5x rate pending. |
| P2-3 | Handset fallback driver with the same interface as the MCU encoder driver. | P | Needs HW | |
| P2-4 | Performance on the N150: STT latency, solve time, live-stack frame rate, processing time. | P | Needs HW | Laptop numbers recorded in PRs. |
| P2-5 | Power test: 1 h of stacking + both cameras + heaters on the power bank; no reboot on plug/unplug. | P | Needs HW | |
| P2-6 | Field test in the developer's yard: finder on a tripod with the MeLE, full voice loop, plate solving. | P | Needs HW | `solve` mount mode makes this possible without encoders. |

## 13. Phase 3 (on the telescope)

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| P3-1 | Encoder bring-up: pinout/voltage, quadrature, counts/rev, direction (handset fallback if needed). | P | Needs HW | |
| P3-2 | Mechanical: finder bracket, heater on the secondary, strain relief, balance, dolly. | P | Needs HW | |
| P3-3 | Calibration wizard: location, solves, finder-main offset, horizon walk (+ Stellarium), focus reference positions (prime/Barlow). | P | Partial | Wizard covers location, 3 solves and the horizon walk; the finder-main offset is learned automatically on the first centering (G5); focus reference positions not done. |
| P3-4 | First-light checklist: back-focus reach, collimation, solve reliability, guidance feel, dew, battery runtime. | P | Needs HW | Collimation check: see F3. |
| P3-5 | Teach and tune: prompts, cue pacing, UI size for the actual user. | P | Needs HW | |
| P3-6 | Remote support: Tailscale access confirmed; nightly log upload. | P | Partial | JSON logs exist; upload not implemented. |

## 14. Reliability and observability

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| R1 | Log everything as structured JSON; keep the last N sessions. | P | Done | `astro/logs.py`: one file per server run, newest 20 kept. |
| R2 | A failing guidance tick never ends guidance; camera failures are spoken, never crash the session. | A | Done | |
| R3 | Hardware opened once and rolled back if setup fails part-way; drivers validated before touching hardware. | A | Done | |
| R4 | Only one user of a camera at a time (recording, stacking, focus, centering, viewers). | A | Done | |
| R5 | Tests never touch the network (weather, Claude API); captures, pictures, site, horizon and logs from tests go to temporary directories. | A | Done | Importing the server still creates an empty `data/gallery/`. |
| R6 | Works on the developer's laptop and the MiniPC; no OS-specific paths. | P | Partial | Runs on Debian 12 (ChromeOS); Ubuntu 24.04 untested. |
| R7 | Nightly log upload for remote support. | P | Not done | See P3-6. |

## 15. Conventions and repository

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| CV1 | Monorepo: `server/`, `web/`, `firmware/`, `scripts/`, `docs/`, `tests/`. | P | Partial | Python lives in the `astro/` package (not `server/`); the others match. |
| CV2 | Config in `config/*.toml`; secrets (ANTHROPIC_API_KEY) in `.env`, never committed. Per-install data (site, horizon, captures, logs) in git-ignored `data/`. | P, A | Done | |
| CV3 | `ASTRO_SIM=1` runs everything against simulators; all tests pass in sim mode. | P | Done | 247 tests. |
| CV4 | Hardware tools: browser viewer (`viewer.py`) and matplotlib viewer (`viewer_plot.py`) sharing common code; exposures in seconds (values > 10 rejected); display rotated 180 deg (lens inversion) with `--raw`; scripts run the project venv from their shebang. | U | Done (HW) | `scripts/hwcheck/`. |
| CV5 | Hardware checkout results recorded. | P | Done | `docs/hardware-results.md`. |
| CV6 | Docs: one-page large-print user guide and a developer runbook. | P | Done | `docs/user-guide.md`, `docs/dev-runbook.md`. |
| CV7 | Store: SQLite for locations, horizon masks, sessions, images, calibration. | P | Partial | Persistence uses files instead: TOML (`data/site.toml`, `data/horizon.toml`), JSON logs, SER/PNG in `data/`. No SQLite; sessions and calibration (mount model, finder-main offset) are not persisted across restarts. |

## 16. Development process (`CLAUDE.md` and session requests)

| ID | Requirement | Src | Status | Evidence / gap |
|---|---|---|---|---|
| D1 | Clean, concise, reviewable code; no overly complex designs. | U | Done | Ongoing practice. |
| D2 | Open-source libraries welcome; avoid binaries except closed drivers with no alternative. | U | Done | See H1. |
| D3 | Commit often; modern CI/CD. | U | Done | GitHub Actions: lint, tests (sim, solver required), firmware compile. |
| D4 | Good unit tests, not excessive. | U | Done | Ongoing practice. |
| D5 | Prototype first, then productize. | U | Done | Ongoing practice. e.g. planet stacking prototype before PSS. |
| D6 | Propose code-hygiene steps after long tangents. | U | Done | Ongoing practice. |
| D7 | Every change through a PR with GitHub Copilot review; feedback addressed before merge. | U | Done | `main` protected (PR + passing `test` check, admins included); `scripts/dev/`. |
| D8 | Annotated git tag when the developer confirms something works on real hardware ("ship it!"). | U | Not done | No confirmation given yet; no tags. |
| D9 | Token-efficient work. | U | Done | Ongoing practice. |
| D10 | American English in code, comments and docs. | U | Done | |
| D11 | Parallel sessions use separate git worktrees; one session merges a branch. | A | Done | |
| D12 | Phase 1 is done when, in sim mode, a full session works end to end via the tablet (voice -> plan -> guide -> simulated capture -> processed image), and the finder solves real sky. | P | Partial | Sim loop runs end to end through the server's WebSocket (tests and scripted sessions), and the finder solved real sky on 2026-10-03; not yet exercised from the user's Android tablet. |

---

## Not done or partial: summary

Not done: HW11 camera temperature, F3 collimation check, C5 sample-data validation, R7/P3-6 log upload, D8 hardware tags (awaiting confirmation).

Partial: D12 tablet end-to-end run, CV7 SQLite store / persisted calibration, S4 horizon mask in goto, G3 left/right learning, G4 beeps, PL1 Moon phase, PL2 Moon features, V1 planetary live view, V2 self-signed HTTPS, V5 `describe(target)`, C3 PSS/wavelets, C4 GraXpert/Siril, PS4 full sky session, PS5 shot list, P2-1 hotspot, P3-3 focus reference positions, R6 Ubuntu, CV1 package layout, M1-M3.
