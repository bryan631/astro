# Hardware checkout results

Tested 2026-10-03 on the ChromeOS Linux container (Debian 12, USB passthrough), SVBony SDK v1.13.4 x64.
Scripts: `scripts/hwcheck/` (`enumerate.py`, `checklist.py`). SDK lives in `~/sdk`, not committed.

## Cameras (SV705C main, SV905C2 finder)

| Check | SV705C | SV905C2 |
|---|---|---|
| Appears in `lsusb` (f266:9a0a) | pass | pass |
| Connect x10 in a row | 10/10 | 10/10 |
| Exposure 0.2 / 0.5 / 1 / 2 s | pass (frame interval matches exposure; scene saturated) | pass (saturates at >=0.2 s in a moderately lit room, f/1.4 wide open) |
| ROI fps, RAW8, 2 ms exposure | 640x480: 139, 1280x720: 47, full 3856x2180: 5.4 | 640x480: 75, 1280x720: 46, full 1280x960: 34 |
| Debayer (GR pattern) | plausible, bare sensor sees blue light (SV705C is now capped, no lens) | plausible: smooth, no mosaic artifacts; blue/purple cast because no white balance is applied. Out of focus, so color accuracy not judged |

## Quirks
- **USB2 link.** Both cameras enumerate at 480M inside the container, even the USB3 SV705C. Full-frame rate (5.4 fps) and the 47 fps at 1280x720 look bus-limited. Retest on the MeLE miniPC for real USB3 numbers.
- **Stop/start on one open handle is flaky.** Re-starting video capture on the same open camera intermittently returned `SVB_ERROR_TIMEOUT` (11), once leading to a libusb assert crash at exit. Closing and reopening the camera between runs was reliable. Exposure must be set before `SVBStartVideoCapture`.
- **The SDK's bundled `libusb-1.0.so*` files are empty.** Load the system `libusb-1.0.so.0` with `RTLD_GLOBAL` first.
- Needs udev rule `90-ckusb.rules` (mode 0666) for non-root access.

## Still open
- ~~**Finder focus**~~ done 2026-10-03 (distant object, `viewer.py`).
- **USB3 speed:** retest frame rates on the MeLE (container links at USB2).
- **Restart timeout / viewer recovery:** retest on the MeLE; confirm the viewer reopens cleanly after a real SDK timeout.
- **Arduino Nano Every:** not arrived; serial passthrough and blink test pending.
