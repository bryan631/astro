#!/usr/bin/env python3
"""Live MJPEG viewer for an SVBony camera. Open http://localhost:8080 in the ChromeOS browser.

  ./viewer.py finder --exp 50 --gain 100
  ./viewer.py main --roi 1280x720
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import svb

ap = argparse.ArgumentParser()
ap.add_argument("cam", help="finder (SV905C), main (SV705C) or a camera index")
ap.add_argument("--exp", type=float, default=20, help="exposure in ms")
ap.add_argument("--gain", type=int, default=None)
ap.add_argument("--roi", default=None, help="WxH, centered, e.g. 1280x720")
ap.add_argument("--width", type=int, default=960, help="max stream width")
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()

cams = svb.enumerate_cameras()
key = {"finder": "SV905C", "main": "SV705C"}.get(args.cam)
info = next((c for c in cams if key and key in c.name.decode()), None) if key else cams[int(args.cam)]
if info is None:
    raise SystemExit(f"no {args.cam} camera; found {[c.name.decode() for c in cams]}")

cam = svb.Camera(info)
p = cam.prop
cam.set_control(svb.EXPOSURE, int(args.exp * 1000))
if args.gain is not None:
    cam.set_control(svb.GAIN, args.gain)
w, h = map(int, args.roi.split("x")) if args.roi else (p.max_w, p.max_h)
w, h = min(w, p.max_w) // 8 * 8, min(h, p.max_h) // 2 * 2
cam.start((p.max_w - w) // 2 // 2 * 2, (p.max_h - h) // 2 // 2 * 2, w, h)
print(f"{info.name.decode()} {w}x{h} exp={args.exp}ms; http://localhost:{args.port}", flush=True)

jpeg, fps = b"", 0.0
code = getattr(cv2, f"COLOR_Bayer{svb.BAYER[p.bayer]}2BGR")


def grab():
    global jpeg, fps
    t = time.time()
    while True:
        raw = cam.frame(wait_ms=int(args.exp) * 3 + 2000)
        img = cv2.cvtColor(raw, code)
        if w > args.width:
            img = cv2.resize(img, (args.width, h * args.width // w), interpolation=cv2.INTER_AREA)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t, 1e-6)
        t = now
        text = f"{fps:.1f} fps  max={raw.max()}"
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpeg), jpeg))
                time.sleep(max(0.03, args.exp / 1000))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


threading.Thread(target=grab, daemon=True).start()
try:
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()
except KeyboardInterrupt:
    pass
finally:
    cam.stop()
    cam.close()
