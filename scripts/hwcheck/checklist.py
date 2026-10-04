#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""Plan checklist per camera: 10x connect, ROI fps (RAW8), exposure sweep, debayer sample."""
import sys
import time

import cv2
import svb

out = sys.argv[1] if len(sys.argv) > 1 else "."
for info in svb.enumerate_cameras():
    name = info.name.decode().replace(" ", "_")
    print(f"== {name}")

    ok = 0
    for i in range(10):
        try:
            with svb.Camera(info) as cam:
                cam.set_control(svb.EXPOSURE, 10_000)
                cam.start(w=640, h=480)
                cam.frame()
            ok += 1
        except RuntimeError as e:
            print(f"  connect {i}: {e}")
    print(f"connect x10: {ok}/10")

    with svb.Camera(info) as probe:
        p = probe.prop
    for sec in (0.2, 0.5, 1, 2):  # exposure must be set before start; changing it mid-stream times out
        wait = int(sec * 1000) * 3 + 3000
        try:
            with svb.Camera(info) as cam:
                cam.set_control(svb.EXPOSURE, int(sec * 1e6))
                cam.start()
                cam.frame(wait_ms=wait)
                t0 = time.time()
                raw = cam.frame(wait_ms=wait)
            print(f"exposure {sec}s: frame interval {time.time() - t0:.2f}s, mean={raw.mean():.1f} max={raw.max()}")
        except RuntimeError as e:
            print(f"exposure {sec}s: {e}")

    for w, h in [(640, 480), (1280, 720), (p.max_w, p.max_h)]:
        w, h = min(w, p.max_w) // 8 * 8, min(h, p.max_h) // 2 * 2
        try:
            with svb.Camera(info) as cam:
                cam.set_control(svb.EXPOSURE, 2_000)
                cam.start(w=w, h=h)
                cam.frame()
                n, t0 = 60, time.time()
                for _ in range(n):
                    cam.frame()
            print(f"ROI {w}x{h} @2ms: {n / (time.time() - t0):.1f} fps")
        except RuntimeError as e:
            print(f"ROI {w}x{h}: {e}")

    with svb.Camera(info) as cam:
        cam.set_control(svb.EXPOSURE, 10_000)
        cam.start()
        cam.frame()
        raw = cam.frame()
    bgr = cv2.cvtColor(raw, getattr(cv2, f"COLOR_Bayer{svb.CV_BAYER[p.bayer]}2BGR"))
    b, g, r = bgr.reshape(-1, 3).mean(0)
    print(f"debayer {svb.BAYER[p.bayer]}: mean R={r:.1f} G={g:.1f} B={b:.1f}")
    cv2.imwrite(f"{out}/{name}_checklist.png", cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
