#!/usr/bin/env python3
"""Hands-free test audio: spoken phrases at four levels and two speeds over wind, hiss and chatter.

    scripts/dev/voice_cases.py [--transcribe]     # writes data/voice-cases/*.wav (git-ignored)

Speech is the project's own Piper voice; noise is held at a fixed level so quiet speech has a
poor signal-to-noise ratio, as on a windy hillside. With --transcribe each case goes through
whisper, the wake word and the offline command grammar, and is scored three ways:
  miss           "Astro ..." was spoken but nothing was acted on
  wrong command  acted on, but the intent (or the target) differs from what was said
  false trigger  acted on something that should have been ignored (near-misses like "Castro")
Cases are tiered by signal-to-noise ratio. Obvious ones (clean or loud over the noise, normal
speed) must pass: any failure exits non-zero. Marginal ones (quiet, fast or slow, chatter) are a
benchmark: per-group scores to track as the pipeline changes, never a pass/fail.
Needs the voice install (scripts/install-voice.sh).
"""
import io
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from astro.intents import match_name, parse
from astro.voice.speech import Stt, Tts
from astro.wake import strip_wake

OUT = Path(__file__).resolve().parents[2] / "data" / "voice-cases"
stt, tts = Stt(), Tts()
RATE = 16000
NOISE_DBFS = -40  # fixed, whatever the speech level
LEVELS = {"loud": -12, "normal": -24, "quiet": -38, "faint": -48}  # speech peak, dBFS
SPEEDS = (0.85, 1.15)  # extra speaking rates, at the normal level only
NOISES = ("none", "hiss", "wind", "chatter")
OBVIOUS_SNR_DB = 25  # speech peak over noise level; clean audio is obvious unless faint


def tier(level: str, kind: str, speed: float) -> str:
    snr = LEVELS[level] - NOISE_DBFS
    clean = kind == "none" and level != "faint"
    return "obvious" if speed == 1.0 and kind != "chatter" and (clean or snr >= OBVIOUS_SNR_DB) else "marginal"
ACT = ["Astro, go to Jupiter.", "Astro, stop.", "Astro, what's good tonight?", "Astro, next.",
       "Astro, take a picture.", "Hey Astro, go to Saturn.", "Astro."]  # "Astro." arms the mic
IGNORE = ["Go to Saturn.", "What's good tonight?", "Take a picture.", "Go to Astro.",
          "Astronomy is fun.", "Castro was on the news.", "I bought an astrolabe."]
UNDECIDED = ["Stop."]  # should a bare "stop" work without the wake word? Reported, not scored.
rng = np.random.default_rng(1)


def speak(text: str, speed: float = 1.0) -> np.ndarray:
    with wave.open(io.BytesIO(tts.synthesize(text))) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
        return resample_poly(x, RATE * 100, round(w.getframerate() * speed * 100))


def scaled(x: np.ndarray, dbfs: float, peak: bool = True) -> np.ndarray:
    ref = np.abs(x).max() if peak else np.sqrt(np.mean(x**2))
    return x * 10 ** (dbfs / 20) / ref


def noise(kind: str, n: int) -> np.ndarray:
    if kind == "none":
        return np.zeros(n)
    white = rng.standard_normal(n)
    if kind == "hiss":
        x = white
    elif kind == "wind":  # low rumble in slow gusts
        x = sosfilt(butter(2, 300, "low", fs=RATE, output="sos"), white)
        x = x * np.interp(np.arange(n), np.linspace(0, n, 8), rng.uniform(0.2, 1.0, 8))
    else:  # chatter: bystanders talking over each other at a distance (never saying Astro)
        x = np.zeros(n)
        for p in rng.choice(IGNORE[:3], 4):
            s = speak(p)
            at = rng.integers(0, max(1, n - len(s)))
            x[at:at + len(s)] += s[:n - at]
    return scaled(x, NOISE_DBFS, peak=False)


def save(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def slug(*parts) -> str:
    return "_".join(str(p) for p in parts).lower().replace(" ", "_").translate(
        str.maketrans("", "", ",.?!'"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for kind in NOISES[1:]:  # noise alone: nothing to act on
        name = f"noise_{kind}"
        save(OUT / f"{name}.wav", noise(kind, 6 * RATE))
        manifest[name] = {"text": "", "want": "ignore", "level": "-", "noise": kind, "speed": 1.0, "tier": "obvious"}
    for want, texts in (("act", ACT), ("ignore", IGNORE), ("undecided", UNDECIDED)):
        for text in texts:
            for speed in (1.0, *SPEEDS):
                voice = speak(text, speed)
                for level, dbfs in LEVELS.items():
                    if speed != 1.0 and level != "normal":
                        continue
                    for kind in NOISES:
                        bed = noise(kind, len(voice) + 2 * RATE)
                        bed[RATE:RATE + len(voice)] += scaled(voice, dbfs)
                        name = slug(text, level, kind, speed)
                        save(OUT / f"{name}.wav", bed)
                        manifest[name] = {"text": text, "want": want, "level": level, "noise": kind,
                                     "speed": speed, "tier": tier(level, kind, speed)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} cases to {OUT}")
    if "--transcribe" in sys.argv:
        report(manifest)


def same_command(heard: str, said: str) -> bool:
    a, b = parse(heard), parse(said)
    if a is None or b is None or a.name != b.name:
        return False
    return a.target is None or match_name(a.target, [b.target or ""]) is not None


def score(case: dict, heard: str) -> str:
    command = strip_wake(heard)  # hands-free acts only after the wake word
    if case["want"] == "undecided":
        return "acted" if command is not None else "ignored"
    if case["want"] == "ignore":
        return "ok" if command is None else "false trigger"
    if command is None:
        return "miss"
    said = strip_wake(case["text"])
    return "ok" if (command == "" if said == "" else same_command(command, said)) else "wrong command"


def report(manifest: dict) -> None:
    results = {}
    for name, case in manifest.items():
        heard = stt.transcribe((OUT / f"{name}.wav").read_bytes())
        results[name] = score(case, heard), heard
    cases = [manifest[n] | {"name": n, "result": r, "heard": h}
             for n, (r, h) in results.items() if manifest[n]["want"] != "undecided"]
    obvious = [c for c in cases if c["tier"] == "obvious"]
    failed = [c for c in obvious if c["result"] != "ok"]
    print(f"OBVIOUS (pass/fail): {len(obvious) - len(failed)}/{len(obvious)} pass")
    for c in failed:
        print(f"  FAIL {c['result']:14} {c['name']:52} {c['heard']!r}")

    def credit(c):  # right command 1, right wake word but wrong command 1/2, a miss 0
        return 1.0 if c["result"] == "ok" else 0.5 if c["result"] == "wrong command" else 0.0

    marginal = [c for c in cases if c["tier"] == "marginal"]
    print(f"\nMARGINAL (benchmark, {len(marginal)} cases): command score 0-100 / false-trigger-free %")
    for key in ("level", "noise", "speed"):
        for value in sorted({c[key] for c in marginal}, key=str):
            act = [c for c in marginal if c[key] == value and c["want"] == "act"]
            ign = [c for c in marginal if c[key] == value and c["want"] == "ignore"]
            cmd = f"{100 * sum(map(credit, act)) / len(act):5.0f}" if act else "    -"
            quiet = f"{100 * sum(c['result'] == 'ok' for c in ign) / len(ign):5.0f}" if ign else "    -"
            print(f"  {key:6} {value!s:7} command {cmd}   no false trigger {quiet}")
    act = [c for c in marginal if c["want"] == "act"]
    ign = [c for c in marginal if c["want"] == "ignore"]
    print(f"  overall command {100 * sum(map(credit, act)) / max(1, len(act)):.0f}"
          f"   no false trigger {100 * sum(c['result'] == 'ok' for c in ign) / max(1, len(ign)):.0f}")
    stops = [c for c in cases if c["want"] == "act" and "stop" in c["text"].lower()]
    print(f"  'Astro, stop' correct {sum(c['result'] == 'ok' for c in stops)}/{len(stops)} (all tiers)")
    undecided = Counter(r for n, (r, _) in results.items() if manifest[n]["want"] == "undecided")
    print("bare 'Stop.' without the wake word (undecided, not scored):", dict(undecided))
    (OUT / "results.json").write_text(json.dumps(cases, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
