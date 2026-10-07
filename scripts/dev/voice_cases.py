#!/usr/bin/env python3
"""Hands-free test audio: spoken commands at three levels over wind, hiss and chatter.

    scripts/dev/voice_cases.py [--transcribe]     # writes data/voice-cases/*.wav (git-ignored)

Speech is the project's own Piper voice; noise is held at a fixed level so quiet speech has a
poor signal-to-noise ratio, as it would on a windy hillside. With --transcribe each case goes
through whisper and the wake-word check, and the table shows what hands-free would act on.
Needs the voice install (scripts/install-voice.sh).
"""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from astro.voice.speech import Stt, Tts
from astro.wake import strip_wake

OUT = Path(__file__).resolve().parents[2] / "data" / "voice-cases"
stt, tts = Stt(), Tts()
RATE = 16000
NOISE_DBFS = -40  # fixed, whatever the speech level
PHRASES = ["Astro, go to Jupiter.", "Astro.", "Astro, stop.", "Go to Saturn.", "What's good tonight?"]
LEVELS = {"loud": -12, "normal": -24, "quiet": -38}  # speech peak, dBFS
rng = np.random.default_rng(1)


def speak(text: str) -> np.ndarray:
    with wave.open(io.BytesIO(tts.synthesize(text))) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
        return resample_poly(x, RATE, w.getframerate())


def scaled(x: np.ndarray, dbfs: float, peak: bool = True) -> np.ndarray:
    ref = np.abs(x).max() if peak else np.sqrt(np.mean(x**2))
    return x * 10 ** (dbfs / 20) / ref


def noise(kind: str, n: int) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "hiss":
        x = white
    elif kind == "wind":  # low rumble in slow gusts
        x = sosfilt(butter(2, 300, "low", fs=RATE, output="sos"), white)
        gust = np.interp(np.arange(n), np.linspace(0, n, 8), rng.uniform(0.2, 1.0, 8))
        x = x * gust
    else:  # chatter: other phrases, talking over each other at a distance
        x = np.zeros(n)
        for p in rng.choice([p for p in PHRASES if strip_wake(p) is None], 4):  # bystanders don't say Astro
            s = speak(p)
            at = rng.integers(0, max(1, n - len(s)))
            x[at:at + len(s)] += s[:n - at]
    return scaled(x, NOISE_DBFS, peak=False)


def save(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    for kind in ("none", "hiss", "wind", "chatter"):
        if kind != "none":  # noise alone: nothing to act on
            cases[f"noise_{kind}"] = (noise(kind, 6 * RATE), None)
        for text in PHRASES:
            voice = speak(text)
            for level, dbfs in LEVELS.items():
                n = len(voice) + 2 * RATE
                bed = noise(kind, n) if kind != "none" else np.zeros(n)
                bed[RATE:RATE + len(voice)] += scaled(voice, dbfs)
                cases[f"{text.strip('.?!').lower().replace(' ', '_').replace(',', '').replace(chr(39), '')}_{level}_{kind}"] = (bed, text)
    manifest = {}
    for name, (x, text) in cases.items():
        save(OUT / f"{name}.wav", x)
        manifest[name] = {"text": text, "wake": bool(text and strip_wake(text) is not None)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(cases)} cases to {OUT}")
    if "--transcribe" in sys.argv:
        report(manifest)


def report(manifest: dict) -> None:
    bad = 0
    for name, want in manifest.items():
        heard = stt.transcribe((OUT / f"{name}.wav").read_bytes())
        acted = strip_wake(heard) is not None  # hands-free acts only on the wake word
        ok = acted == want["wake"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:44} {heard!r}")
    print(f"{len(manifest) - bad}/{len(manifest)} behave correctly")


if __name__ == "__main__":
    main()
