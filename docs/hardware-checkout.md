# Hardware checkout session briefing

You are the **hardware checkout** session for the `astro` project (repo: https://github.com/bryan631/astro).
Read `CLAUDE.md` (coding rules) and `docs/plan.md` (full project plan; see Hardware and Phase 1 steps 2-3) first.

## Goal
Interactively verify the user's hardware works, so we find broken or missing pieces early. Another
session is building the main code (scaffold, pointing math, guidance) in parallel; stay out of `astro/`.

## Environment
- ChromeOS laptop, Linux container (Debian 12, x86_64, Python 3.11). It is a VM: USB devices must be
  attached manually via ChromeOS Settings > Developers > Linux > Manage USB devices.
- If a device fails here, retest on the MeLE miniPC (Ubuntu 24.04) before blaming the hardware.
- No sudo password prompts needed for apt (passwordless sudo works).

## Scope
1. Cameras: SVBONY SV705C (main) and SV905C (finder). Needs the SVBony Linux x86_64 SDK, a closed binary
   the user must download from SVBony. Ask for it; don't hunt for unofficial copies.
   - Confirm both appear in `lsusb` after attaching to the container.
   - Write a small ctypes enumerate + capture-one-frame script, then run the plan's checklist:
     connect 10x in a row, ROI video frame rate at 8-bit, 0.2-2s exposures, debayer correct.
   - Document pass/fail and any quirks in `docs/hardware-results.md`.
2. Arduino Nano Every: only when it arrives. Check serial passthrough (`/dev/ttyACM*`), flash a blink test.
3. If the SDK or driver fails, document it; the fallback is a ZWO ASI585MC.

## Rules for this session
- Put all work in `scripts/hwcheck/` and `docs/hardware-results.md`. Work on branch `hwcheck`, commit often,
  push the branch. The main session owns `main`; don't push to it.
- Keep scripts short and readable; the real driver wrapper will later live in `astro/devices/` behind the
  `Camera` interface (`astro/devices/base.py` once it lands on `main`).
- Save a few sample frames only if small (<2 MB total); never commit the SDK binaries.
- Be token-efficient and ask the user for physical actions (plugging in, covering the lens, etc.).
