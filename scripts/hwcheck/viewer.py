#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Live camera view in the browser (MJPEG).

    scripts/hwcheck/viewer.py finder --exp 0.5 --gain 100     then open http://localhost:8080
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from live import LiveCamera, parse_args

args = parse_args(__doc__, lambda ap: ap.add_argument("--port", type=int, default=8080))
live = LiveCamera(args).start()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while not live.stopped:
                if live.latest is not None:
                    jpeg = cv2.imencode(".jpg", live.latest, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(max(0.03, min(args.exp, 0.5)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


server = ThreadingHTTPServer(("", args.port), Handler)
server.daemon_threads = True
print(f"{args.cam}: open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nstopping after the current exposure...", flush=True)
finally:
    live.stop()
