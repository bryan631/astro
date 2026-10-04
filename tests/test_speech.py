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


def test_cue_phrases_cover_what_the_guide_says():
    phrases = set(cue_phrases())
    g = Guide(45, 100)
    for t, (alt, az) in enumerate([(10, 100), (30, 100), (44.6, 100), (45, 100)]):
        _, cue = g.update(alt, az, t * 5.0)
        if cue:
            assert cue.text in phrases, cue.text


def test_audio_ctx_covers_the_utterance():
    from astro.voice.speech import audio_ctx

    assert audio_ctx(2.0) == 256  # short command: minimum window
    assert audio_ctx(8.0) == 600  # 8 s * 1.5 margin * 50 frames/s
    assert audio_ctx(60.0) == 1500  # never more than whisper's full 30 s window
