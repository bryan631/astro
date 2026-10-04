#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Live camera view in the browser (MJPEG) for focusing and hardware checks.

    scripts/hwcheck/viewer.py finder --exp 0.5 --gain 100     then open http://localhost:8080
Overlay: fps, max pixel, sharpness (higher = sharper), stars and HFR (lower = sharper).
Ctrl+C stops capture cleanly: the current exposure finishes before the camera closes.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from astro.capture.preview import focus_numbers
from astro.devices.base import Roi
from astro.devices.svbony import SvbonyCamera

MODELS = {"finder": "SV905C", "main": "SV705C"}
FOCUS_EVERY_S = 1.0  # focus numbers are slower than display; update them once a second

ap = argparse.ArgumentParser()
ap.add_argument("cam", choices=MODELS, help="finder (SV905C) or main (SV705C)")
ap.add_argument("--exp", type=float, default=0.05, help="exposure, seconds")
ap.add_argument("--gain", type=int, default=100)
ap.add_argument("--roi", default=None, help="WxH, centered, e.g. 1280x720")
ap.add_argument("--width", type=int, default=960, help="max stream width")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
if args.exp > 10:
    ap.error(f"--exp is in seconds; {args.exp:g} s is very long. Did you mean {args.exp / 1000:g}?")

cam = SvbonyCamera(MODELS[args.cam])
cam.connect()
cam.set_exposure(args.exp)
cam.set_gain(args.gain)
if args.roi:
    sw, sh = cam.sensor_size
    w, h = (int(v) for v in args.roi.split("x"))
    w, h = min(w, sw) // 8 * 8, min(h, sh) // 2 * 2
    cam.set_roi(Roi((sw - w) // 4 * 2, (sh - h) // 4 * 2, w, h))
# OpenCV names Bayer codes by row 2, columns 2-3 (e.g. sensor RGGB -> BayerBG).
debayer = getattr(cv2, f"COLOR_Bayer{cam.bayer[3]}{cam.bayer[2]}2BGR")
stop = threading.Event()
latest = {"jpeg": b""}


def grab():
    fps, t, focus, focus_t = 0.0, time.time(), "", 0.0
    while not stop.is_set():
        raw = cam.capture()  # the driver reopens the camera on SDK timeouts
        now = time.time()
        fps, t = 0.9 * fps + 0.1 / max(now - t, 1e-6), now
        if now - focus_t >= FOCUS_EVERY_S:
            f = focus_numbers(raw)
            focus, focus_t = f"sharp {f.sharpness:.0f}  stars {f.stars} hfr {f.hfr_px:.2f}", now
        img = cv2.cvtColor(raw, debayer)
        h, w = img.shape[:2]
        if w > args.width:
            img = cv2.resize(img, (args.width, h * args.width // w), interpolation=cv2.INTER_AREA)
        text = f"{fps:.1f} fps  max {raw.max()}  {focus}"
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        latest["jpeg"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while not stop.is_set():
                jpeg = latest["jpeg"]
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                 % (len(jpeg), jpeg))
                time.sleep(max(0.03, min(args.exp, 0.5)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


grabber = threading.Thread(target=grab)
grabber.start()
server = ThreadingHTTPServer(("", args.port), Handler)
server.daemon_threads = True
print(f"{args.cam} exp {args.exp:g}s gain {args.gain}; open http://localhost:{args.port}", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nstopping after the current exposure...", flush=True)
finally:
    stop.set()
    grabber.join()  # never close the SDK while a capture is in flight (that segfaulted)
    cam.close()
