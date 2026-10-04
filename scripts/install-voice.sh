#!/usr/bin/env bash
# Offline speech for the MiniPC: whisper.cpp (speech-to-text) + Piper (text-to-speech).
# Writes voice.env, which scripts/run.sh loads. Untested until the MiniPC arrives.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor models
sudo apt-get install -y -q build-essential cmake git ffmpeg
if [ ! -x vendor/whisper.cpp/build/bin/whisper-cli ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
  cmake -S vendor/whisper.cpp -B vendor/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build vendor/whisper.cpp/build -j --target whisper-cli
fi
[ -f models/ggml-base.en.bin ] || (cd models && bash ../vendor/whisper.cpp/models/download-ggml-model.sh base.en .)
.venv/bin/pip install -q piper-tts
VOICE=en_US-lessac-medium
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
[ -f models/$VOICE.onnx ] || curl -L -o models/$VOICE.onnx "$BASE/$VOICE.onnx"
[ -f models/$VOICE.onnx.json ] || curl -L -o models/$VOICE.onnx.json "$BASE/$VOICE.onnx.json"
cat > voice.env <<ENV
ASTRO_WHISPER_BIN=$PWD/vendor/whisper.cpp/build/bin/whisper-cli
ASTRO_WHISPER_MODEL=$PWD/models/ggml-base.en.bin
ASTRO_PIPER_BIN=$PWD/.venv/bin/piper
ASTRO_PIPER_MODEL=$PWD/models/$VOICE.onnx
ENV
echo "Voice installed; restart astro."
