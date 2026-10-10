# astro — Voice-guided push-to + astrophotography for an 8" Dobsonian

## Mission
Convert a 2012 Orion SkyQuest XT8i IntelliScope (8" f/5.9, 1200mm FL, manual alt-az Dob) into
a voice-controlled push-to and imaging rig for a 77-year-old user with weakening eyesight.
He talks to a tablet ("I'm in the backyard, what's a good planet tonight?"), picks a target,
and is talked + arrowed onto it ("push left… slower… stop… up a little… stop"). Live view
appears on the tablet; stacked, processed color images appear in a gallery afterward.
Location: West Palm Beach, FL suburbs (Bortle 7–8, very humid). Primary targets: Moon,
planets, bright compact DSOs (no tracking in phase 1 — DSO results will be limited).

The developer (repo owner) builds and tests now, installs on-site later, and supports remotely.
The end user must never need a keyboard, terminal, or config file.

## Hard requirements
- All software open source or written in this repo. Exception: vendor camera SDK binaries (SVBony).
- Voice-first, large-text tablet UI; red night mode; works one-handed.
- Pointing guidance is deterministic local code, NOT the LLM. The LLM only interprets intent,
  calls tools, and narrates.
- SAFETY: never guide toward the Sun. Enforce a solar exclusion zone (≥20°) and a daytime
  lockout for guidance/capture unless an explicit developer override is set.
- Must keep working (basic commands) when internet is down.
- Remote support: assume Tailscale for SSH + HTTPS access.

## Hardware
| Part | Notes |
|---|---|
| Main camera: SVBONY SV705C | IMX585 color, 3840×2160, 2.9µm, USB3. SVBony SDK (closed binary). Prefer 8-bit RAW + ROI for planets. Avoid 16-bit initially (known SDK bugs). |
| Finder/solver camera: SVBONY SV905C + 25mm f/1.4 C-mount lens (5mm CS-C ring) | IMX225 color, 1280×960, 3.75µm, 4.8×3.6mm sensor. ≈11°×8° FOV, ≈31″/px. Exposure 64µs–20s. Run lens at f/2–2.8. |
| Computer (arriving): MeLE Quieter 4C, Intel N150, 16GB/512GB, fanless | Target OS: Ubuntu LTS. Powered by USB-C PD (C1 port, 12–20V). |
| Microcontroller: classic Arduino Nano (ATmega328P, CH340) on a screw-terminal board; the firmware also builds for the Nano Every | 5V logic. Reads IntelliScope encoders, drives dew heaters, reads BME280 (+ optional DS18B20). |
| Encoders | IntelliScope Hall-effect (2 Allegro A3515 per axis): two analog sine waves a quarter cycle apart, ~0.2-4.7 V, 36 cycles/rev; the Nano reads them on A2/A3/A6/A7 and interpolates with atan2 to 9216 counts/rev. RJ12 (scope end, white on pin 1): white 5V, blue GND, yellow/black alt, green/red az. Verified on the scope: 90 deg altitude = 2331 counts, repeatable to ~8. Fallback: the IntelliScope handset over RS-232 ("Q" command). |
| Dew | Two 5V USB heater strips (secondary holder, finder lens) via logic-level MOSFET modules on Nano PWM pins; powered from power bank, not the Nano. |
| Power | 100W USB-C PD power bank. |
| Tablet | User's tablet; browser PWA is the mic, speaker, and screen. |

Optics math (put in `astro/optics.py` and unit-test it):
- Plate scale ″/px = 206.265 × pixel_µm / focal_mm.
- Main @1200mm: 0.50″/px, FOV ≈ 32′×18′. With 2× Barlow (2400mm): 0.25″/px, ≈16′×9′.
- Finder @25mm: ≈31″/px, ≈11°×8°.
- Sidereal drift ≈ 15.04″/s × cos(dec) → ~30 px/s on main camera at prime focus.

## Architecture
```
tablet (PWA, HTTPS) ──WebSocket/HTTP── server (Python, FastAPI)
                                         ├─ devices/   main_cam, finder_cam, mcu (serial), sim/ fakes
                                         ├─ pointing/  encoders → alt/az, plate solve sync, mount model
                                         ├─ guidance/  10 Hz loop → cues (voice text, arrows, beeps)
                                         ├─ planner/   ephemeris, catalogs, horizon mask, weather, scoring
                                         ├─ capture/   planetary video (SER), live stack, focus metric
                                         ├─ process/   PSS (planets), Siril + GraXpert (DSO) pipelines
                                         ├─ voice/     whisper.cpp STT, Piper TTS, LLM agent + tools, offline intents
                                         └─ store/     SQLite: locations, horizon masks, sessions, images, calibration
mcu firmware (Arduino Nano)       ──USB serial── devices/mcu
```
Every device has a simulator implementing the same interface so everything runs with no hardware.

Suggested stack (open source): Python 3.11+, FastAPI, numpy, astropy, skyfield, OpenCV,
tetra3/cedar-solve (finder solving), ASTAP CLI (main-camera small-field solving),
PlanetarySystemStacker, Siril (siril-cli scripts), GraXpert (CLI), whisper.cpp,
Piper TTS, Anthropic Python SDK (Claude Haiku for the agent), Open-Meteo (weather, no key),
OpenNGC + Messier/Caldwell catalogs. Ship a single `scripts/setup.sh` for Ubuntu.

## Conventions
- Monorepo: `server/`, `web/`, `firmware/`, `scripts/`, `docs/`, `tests/`.
- Config in `config/*.toml`; secrets (ANTHROPIC_API_KEY) in `.env`, never committed.
- `ASTRO_SIM=1` runs everything against simulators. All tests must pass in sim mode.
- Log everything (structured JSON) to make remote debugging possible; keep the last N sessions.
- Keep the developer's laptop and the Ubuntu MiniPC both working; avoid OS-specific paths.

---

## Phase 1 — Now (laptop, both cameras, finder lens, no telescope)
Goal: everything that doesn't need the PC, Arduino, or scope, built and tested.

1. Repo scaffold, setup script, CI running unit tests in sim mode.
2. Camera layer
   - Python wrapper over the SVBony SDK (ctypes) for both cameras: enumerate, set gain/exposure/ROI/bin,
     8-bit RAW capture, frame streaming, temperature if available. Robust reconnect.
   - Test checklist: connect 10× in a row; ROI video frame rate at 8-bit; 0.2–2s exposures; debayer correct.
   - If the SDK or driver fails these, document it (we may swap to a ZWO ASI585MC).
3. Finder plate solving
   - tetra3/cedar-solve database tuned for ~11°×8° fields; solve from a single frame; return RA/Dec/roll + confidence.
   - Pre-flight finder focus gate: before solving, measure star HFR on the finder frame
     (`astro/capture/focus.py`). If too soft, or the solver finds too few stars, run a voice-guided
     refocus ("turn the finder focus slowly… sharper… stop") instead of failing silently; say why a
     solve failed (soft focus, too few stars, clouds).
   - Test on real sky from the developer's yard: camera + 25mm lens on a tripod. Record solve rate,
     time, and failure modes (clouds, trees, streetlights). Save sample frames to `tests/data/`.
4. Pointing math (pure functions, heavily unit-tested)
   - RA/Dec ↔ Alt/Az for site + time (astropy/skyfield), with refraction.
   - Encoder counts → mount angles (9216 counts/rev, sign/direction configurable).
   - Mount model: sync from plate solves; solve for az zero offset, alt zero offset, and base tilt
     (2-axis) from ≥2 syncs; refine with least squares as more syncs arrive.
   - Finder→main camera offset model (Δalt, Δaz), auto-updated whenever both solve.
5. Guidance engine (simulated scope)
   - Input: current alt/az (10 Hz), target alt/az. Output: cue stream.
   - Coarse (degrees) → fine (arcmin) → "stop" when within the target tolerance (default 4′ at prime focus,
     2′ with Barlow). Hysteresis, rate limiting, never talks over itself.
   - "Left/right" = azimuth as seen by user; direction sign learned on first push.
   - Also emit arrow + distance for UI and an optional beep rate for proximity.
   - Sun exclusion and below-horizon (mask) refusal.
6. Planner
   - "What's good tonight" for a location: visible window, altitude, moon separation/phase, weather
     (Open-Meteo cloud cover), horizon mask, and suitability for Bortle 8 (curated target list with notes).
   - Return ranked choices per category (planet, moon features, nebula, cluster, double star).
7. Voice + agent
   - Tablet PWA: big push-to-talk button, arrows, live view panel, gallery, red night mode.
   - HTTPS on LAN (self-signed cert + install instructions, or Tailscale certs) — required for mic.
   - STT: whisper.cpp (base/small). TTS: Piper. Audio over WebSocket to/from tablet.
   - Claude agent with tools: list_tonight, describe(target), goto(target), stop, where_am_i,
     what_am_i_looking_at, start_capture/stop_capture, focus_assist, set_location, calibrate_horizon,
     session_status. Short, warm, plain-language replies.
   - Offline fallback: small intent grammar for core commands (goto <name>, stop, next, capture, focus).
8. Capture & processing (testable with the main camera on the desk + any lens or none)
   - SER video writer with ROI that follows the brightest blob (planet tracking in-frame).
   - Focus metric (HFR/FWHM for stars, Laplacian variance for planets) → voice "sharper / passed it".
   - Pre-flight main-camera focus gate: run the focus check (with the existing focus coach) at
     session start and after a Barlow change, before any capture.
   - Planet pipeline: PSS stack → wavelets/deconvolution → RGB align → auto-crop → PNG/JPEG.
   - DSO pipeline: register + live stack short subs (with rotation/drift) → GraXpert background +
     denoise → Siril color calibration + stretch.
   - Validate with downloaded sample SER/FITS data.
9. Docs: `docs/user-guide.md` (one page, large print) and `docs/dev-runbook.md`.

Phase 1 is done when: in sim mode a full session works end-to-end via the tablet
(voice → plan → guide → simulated capture → processed image), and the finder solves real sky.

## Phase 2 — MiniPC + Arduino arrive (no telescope)
1. Install Ubuntu LTS on the MeLE; run `scripts/setup.sh`; systemd services; auto-start on boot;
   auto-power-on in BIOS; Tailscale; WiFi + fallback hotspot; log rotation.
2. Firmware (Arduino Nano)
   - Encoder decoding: the IntelliScope sensors turned out analog (see Parts), so the Nano samples
     4 analog pins at ~1 kHz and interpolates with atan2 instead of counting quadrature edges.
   - Line-based serial protocol, e.g. `POS <az_counts> <alt_counts>` at 20 Hz, `ENV <T> <RH> <dewpoint> <optic_T>`,
     commands `HEAT <ch> <0-100>`, `ZERO`, `VER`. Version + checksum. Watchdog.
   - Dew control: Magnus-formula dew point from BME280; PID or simple stepped PWM to keep optics
     ≥2–3°C above dew point; failsafe heaters off on lost host heartbeat.
   - Tested on the scope (2026-10-08): 90 deg of altitude repeats to ~8 counts; full azimuth turns
     read 9041-9216 of 9216 counts once the base bolt was snug.
3. Handset fallback driver: USB-RS232 → IntelliScope "Q" position query, same interface as the MCU encoder driver.
4. Performance on N150: STT latency, solve time, live-stack frame rate, processing time per session.
5. Power test: run a stacking job + both cameras + heaters on the power bank for 1h; confirm no reboot
   when ports are plugged/unplugged.
6. Field test (developer's yard): finder camera on a tripod with the MeLE, full voice loop, plate solving,
   simulated encoders.

## Phase 3 — On the telescope (Florida)
1. Encoder bring-up (done 2026-10-08): RJ12 pinout and voltage found with a multimeter and the
   original controller; analog Hall signals, decoded on the Nano; counts/rev and direction calibrated.
2. Mechanical: finder camera in finder bracket; heater on secondary holder (wire along one spider vane);
   cable strain relief; balance (counterweight if tube droops); dolly.
3. Calibration wizard (voice-led, developer runs it once per location):
   - Name the location; GPS from tablet.
   - First solve → mount model; push to 2–3 other areas to refine tilt.
   - Finder↔main offset from a solve with both cameras.
   - Horizon walk: user sweeps tube along treeline and says "mark" at 8–15 points →
     saved mask (also export Stellarium horizon format).
   - Focus reference positions (prime focus / Barlow) noted for planets vs DSO.
   - Pre-flight gates (finder focus, main focus) wired into the session flow, as built in Phase 1.
4. First-light checklist: back-focus reach (prime vs Barlow), collimation (see below), solve reliability,
   guidance feel (tolerance, voice pacing), dew performance, battery runtime.
   - Collimation check (main camera): defocused-star test measuring ring (donut) evenness and how
     centered the secondary's shadow is, with spoken guidance for the primary-mirror collimation
     screws ("turn the top screw a quarter turn… better… stop"). Run at first light, periodically,
     and prompt for it after the scope has been moved or transported.
5. Teach + tune: simplify prompts based on how he actually talks; tune cue pacing for his
   pushing speed; enlarge UI as needed.
6. Remote support: confirm Tailscale access from developer's home; nightly log upload.

## Phase 3b — Visual workflow (decided 2026-10-10, after the second field night)
The voice-first flow failed in the field: audio can't guide focus or pointing, commands relayed
too slowly for a sky that drifts out of a 32' field in two minutes, and cameras froze because
only one thing could use a camera at a time. New rules: everything works with the volume off,
a few buttons do the few jobs well, and both cameras are live at all times.

1. **Always-on cameras.** Each camera runs its own capture process (the SVBony SDK segfaulted
   with two capture threads in one process) and publishes its latest frame, with a timestamp at
   mid-exposure, to shared memory. Everything else only reads frames: the page's views, plate
   solving, the focus number, stacking, recording. Nothing takes a camera away. Settings: the
   finder stays at its solve settings; the main camera has a view mode and a planet mode (20 ms),
   and while a planet records the main view shows those frames, stretched for display. Sim mode
   uses the same interface with a thread instead of a process.
2. **Screen follows the tablet's rotation.** Portrait: finder above main; landscape: one camera
   full screen, swipe or tap its name for the other. Both always live; one row of buttons below:
   Go to (target list), Align, Focus, Capture (start/stop), Recenter, Pictures, More, STOP. A
   message line (with only what's wrong beside it) and a log replace speech: every message is text.
3. **Voice off.** The page's talk button, hands-free and speech playback, and the session's voice
   routing are removed; astro/voice, wake.py and intents.py stay for later.
4. **Pointing is visual only:** target marker and the main camera's box on the finder view, an
   edge arrow when the target is off screen, distance. No spoken cues.
5. **Align: assume nearly aligned, adjust in one step.** Put a bright (naked-eye) star in the
   main view and tap Align: for 10 s both cameras follow it as it drifts (in the finder, the
   brightest star near where the main camera pointed last, by 3x or it asks for a brighter one).
   main px = A (finder px - c), A a scale-and-rotation, by least squares; the box on the finder
   view comes from inverting it, and one finder plate solve turns c into the alt-az offset
   "go to" aims with (`astro/pointing/align.py`, tested on the 2026-10-10 recording).
6. **Focus number: median half-flux radius** over detected stars, saturated stars skipped, so the
   many faint stars outweigh a few bright ones; shown as a number to maximize (100/HFR) with the
   star count and a short trend. Planets: edge sharpness on the disk. No coach, no voice.
7. **Capture: one Start/Stop button.** A small live-stack preview (JPEG, not the 4.5 MB PNG), the
   raw main view beside it with an edge margin and "time to edge" from the predicted drift.
   Recenter pauses the capture; the finder view and arrows bring the target back; Resume
   continues (deep-sky re-registers to the first frame; planets add a video segment, processed
   together). Option: pause automatically near the edge.

Order: cameras (daytime test on the Mele) → screen and buttons → focus number → capture with
recenter → Align. Each step tested indoors with the simulators before the next night.

**To do**
- Star names on the finder view: small labels beside the brighter stars, from the last plate
  solve, with an on/off button.
- The main camera's box on the finder view: red or orange in night mode.
- Call the main camera "Telescope" everywhere the user sees it (view label, messages, guide).
- Buttons for the setup wizard, horizon walk and collimation (still worded for voice).
- Clear night: a bright-star Align, and focus sweep and wiggle recordings for test data.

## Known risks / open questions
- SV705C Linux SDK stability (fallback: ZWO ASI585MC).
- Azimuth encoder slips if the base's central bolt is loose (the disk turns with the base): keep it snug.
- Camera may not reach focus without the Barlow on this Newtonian.
- No tracking: DSO imaging limited to short subs on bright targets. Phase 2+ option: EQ platform.
- Dew: bring the scope out 30–60 min early; don't move from AC directly to humid air.
- Tablet mic requires HTTPS; verify on the user's actual tablet.
