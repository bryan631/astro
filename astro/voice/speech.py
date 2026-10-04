"""Server-side speech: whisper.cpp (STT) and Piper (TTS), both offline.

Both are optional. If the binaries/models aren't configured, `available()` is False and the
tablet falls back to its browser's built-in speech. Configure via environment:
  ASTRO_WHISPER_BIN, ASTRO_WHISPER_MODEL   e.g. whisper-cli, models/ggml-base.en.bin
  ASTRO_PIPER_BIN, ASTRO_PIPER_MODEL       e.g. piper, models/en_US-lessac-medium.onnx
Audio from the tablet (webm/opus) is converted to 16 kHz mono WAV with ffmpeg.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _tool(env_bin: str, env_model: str) -> tuple[str, str] | None:
    binary, model = os.environ.get(env_bin, ""), os.environ.get(env_model, "")
    if binary and model and shutil.which(binary) and Path(model).exists():
        return binary, model
    return None


class Stt:
    def __init__(self) -> None:
        self.tool = _tool("ASTRO_WHISPER_BIN", "ASTRO_WHISPER_MODEL")

    def available(self) -> bool:
        return self.tool is not None and shutil.which("ffmpeg") is not None

    def transcribe(self, audio: bytes) -> str:
        """Browser audio bytes (any ffmpeg-readable format) -> text."""
        assert self.tool
        binary, model = self.tool
        with tempfile.TemporaryDirectory() as d:
            src, wav = Path(d) / "in", Path(d) / "in.wav"
            src.write_bytes(audio)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", wav],
                           check=True, timeout=20)
            out = subprocess.run([binary, "-m", model, "-f", wav, "-nt", "-np"], check=True,
                                 capture_output=True, text=True, timeout=30)
        return " ".join(out.stdout.split())


class Tts:
    def __init__(self) -> None:
        self.tool = _tool("ASTRO_PIPER_BIN", "ASTRO_PIPER_MODEL")

    def available(self) -> bool:
        return self.tool is not None

    def synthesize(self, text: str) -> bytes:
        """Text -> WAV bytes."""
        assert self.tool
        binary, model = self.tool
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "out.wav"
            subprocess.run([binary, "--model", model, "--output_file", wav], input=text,
                           text=True, check=True, capture_output=True, timeout=20)
            return wav.read_bytes()
