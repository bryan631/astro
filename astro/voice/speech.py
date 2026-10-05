"""Server-side speech: whisper.cpp (STT) and Piper (TTS), both offline.

Both are optional. If the binaries/models aren't configured, `available()` is False and the
tablet falls back to its browser's built-in speech. Configure via environment:
  ASTRO_WHISPER_BIN, ASTRO_WHISPER_MODEL   e.g. whisper-cli, models/ggml-base.en.bin
  ASTRO_PIPER_BIN, ASTRO_PIPER_MODEL       e.g. piper, models/en_US-lessac-medium.onnx
Audio from the tablet (webm/opus) is converted to 16 kHz mono WAV with ffmpeg.
"""

import functools
import io
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

CUE_CACHE = 256  # distinct phrases kept as audio
WAV_BYTES_PER_S = 16000 * 2  # ffmpeg output: 16 kHz mono 16-bit
# Whisper encodes a fixed 30 s window (1500 audio frames) unless told otherwise; a short
# command needs only its own length: ~1.24 s -> ~0.4 s per command on this CPU.
WHISPER_FULL_CTX, WHISPER_MIN_CTX, CTX_MARGIN = 1500, 256, 1.5


def audio_ctx(seconds: float) -> int:
    """Whisper audio context covering `seconds` of speech with margin (50 frames per second)."""
    return int(min(WHISPER_FULL_CTX, max(WHISPER_MIN_CTX, seconds * CTX_MARGIN * 50)))


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
            ctx = audio_ctx(wav.stat().st_size / WAV_BYTES_PER_S)
            out = subprocess.run([binary, "-m", model, "-f", wav, "-nt", "-np", "-ac", str(ctx)],
                                 check=True, capture_output=True, text=True, timeout=30)
        return " ".join(out.stdout.split())


class Tts:
    """Piper text-to-speech. The voice stays loaded in-process (the CLI reloads its 63 MB model
    on every call, ~1.2 s, too slow for "stop"), and repeated cues come from a cache."""

    def __init__(self) -> None:
        self.tool = _tool("ASTRO_PIPER_BIN", "ASTRO_PIPER_MODEL")
        self._voice = None  # piper.PiperVoice, loaded on first use
        self._lock = threading.Lock()  # startup warm-up and requests share one voice
        # Guidance repeats a small set of phrases; cache per instance (not on the class).
        self.synthesize = functools.lru_cache(maxsize=CUE_CACHE)(self._synthesize)

    def available(self) -> bool:
        return self.tool is not None

    def warm(self, phrases: list[str]) -> None:
        """Load the voice and pre-render phrases (run at startup, off the request path)."""
        for phrase in phrases:
            self.synthesize(phrase)

    def _synthesize(self, text: str) -> bytes:
        """Text -> WAV bytes."""
        assert self.tool
        try:
            return self._synthesize_in_process(text)
        except ImportError:  # piper's Python package missing: fall back to the CLI
            return self._synthesize_cli(text)

    def _synthesize_in_process(self, text: str) -> bytes:
        from piper import PiperVoice

        with self._lock:  # load once; one synthesis at a time on the shared voice
            if self._voice is None:
                self._voice = PiperVoice.load(self.tool[1])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                self._voice.synthesize_wav(text, wav)
            return buf.getvalue()

    def _synthesize_cli(self, text: str) -> bytes:
        binary, model = self.tool
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "out.wav"
            subprocess.run([binary, "--model", model, "--output_file", wav], input=text,
                           text=True, check=True, capture_output=True, timeout=20)
            return wav.read_bytes()
