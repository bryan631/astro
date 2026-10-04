from astro.guidance.engine import Guide, cue_phrases
from astro.voice.speech import Tts


class CountingTts(Tts):
    def __init__(self):
        self.calls = []
        super().__init__()

    def _synthesize(self, text):
        self.calls.append(text)
        return text.encode()


def test_tts_warms_and_caches_per_phrase():
    tts = CountingTts()
    tts.warm(["stop", "push left"])
    assert tts.synthesize("stop") == b"stop"
    assert tts.calls == ["stop", "push left"]  # "stop" came from the cache


def test_cue_phrases_match_everything_the_guide_says():
    """Every phrase spoken in simulated sessions (all directions, both left/right conventions,
    overshoots) is pre-rendered, and every pre-rendered phrase is actually used."""
    from tests.test_guidance import run_session

    seen = set()
    trips = [((20, 10), (55, 80)), ((60, 350), (40, 20)), ((50, 100), (20, 60)),
             ((30, 200), (31, 199)), ((40, 90), (40, 40)), ((70, 180), (30, 220))]
    for start, target in trips:
        for kw in ({}, {"right_is_plus_az": False}):
            seen |= set(run_session(start, target, **kw)[1])
    for below, above in ((44.0, 46.0), (46.0, 44.0)):  # vertical overshoots, both ways
        g = Guide(45, 100)
        g.update(below, 100, 0.0)
        seen.add(g.update(above, 100, 0.1)[1].text)
    assert seen == set(cue_phrases())


def test_audio_ctx_covers_the_utterance():
    from astro.voice.speech import audio_ctx

    assert audio_ctx(2.0) == 256  # short command: minimum window
    assert audio_ctx(8.0) == 600  # 8 s * 1.5 margin * 50 frames/s
    assert audio_ctx(60.0) == 1500  # never more than whisper's full 30 s window
