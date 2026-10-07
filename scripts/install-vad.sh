#!/usr/bin/env bash
# Browser voice-activity detection for hands-free mode: Silero VAD (@ricky0123/vad-web) and its
# onnxruntime-web runtime, served from web/vendor/vad (git-ignored) so it works offline.
set -euo pipefail
VAD_VERSION=0.0.31  # pinned like install-solver.sh; bump deliberately
ORT_VERSION=1.22.0
cd "$(dirname "$0")/.."
dest=web/vendor/vad
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$dest"
npm_get() {  # <package path> <version> <file...>: tarball from the npm registry
  local pkg=$1 ver=$2; shift 2
  curl -fsSL "https://registry.npmjs.org/$pkg/-/${pkg##*/}-$ver.tgz" | tar xz -C "$tmp"
  for f in "$@"; do cp "$tmp/package/dist/$f" "$dest/"; done
  rm -rf "$tmp/package"
}
npm_get @ricky0123/vad-web "$VAD_VERSION" bundle.min.js vad.worklet.bundle.min.js silero_vad_v5.onnx
npm_get onnxruntime-web "$ORT_VERSION" ort.wasm.min.js ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm
echo "installed to $dest"
