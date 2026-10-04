#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"'
# Runs with the project .venv (relative to this file), from any directory.
"""List SVBony cameras, grab one RAW8 frame from each, save a debayered PNG."""
import sys

import cv2
import svb

out = sys.argv[1] if len(sys.argv) > 1 else "."
cams = svb.enumerate_cameras()
print(f"{len(cams)} camera(s)")
for info in cams:
    name = info.name.decode()
    with svb.Camera(info) as cam:
        p = cam.prop
        print(f"{name} sn={info.sn.decode()} id={info.camera_id} {p.max_w}x{p.max_h} "
              f"color={p.color} bayer={svb.BAYER[p.bayer]} bits={p.bits}")
        cam.set_control(svb.EXPOSURE, 200_000)  # 0.2 s
        cam.start()
        raw = cam.frame()
    print(f"  frame {raw.shape} min={raw.min()} max={raw.max()} mean={raw.mean():.1f}")
    code = getattr(cv2, f"COLOR_Bayer{svb.CV_BAYER[p.bayer]}2BGR")
    small = cv2.resize(cv2.cvtColor(raw, code), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{out}/{name.replace(' ', '_')}.png", small)
