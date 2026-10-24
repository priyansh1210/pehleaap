import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.audio.vad import hop_s
from pehleaap.labeling.asr import AsrConfig, WhisperASR, looping, read_words, segments_for, transcribe, write_words
from pehleaap.labeling.config import LabelConfig
from pehleaap.labeling.ipu import IPU
from pehleaap.labeling.schema import BC, HOLD_REQ, read_events

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def test_segments_are_padded_clamped_and_split():
    spans = segments_for([IPU(0.1, 1.0), IPU(5.0, 65.0), IPU(9.9, 10.0)], end_s=10.0 + 60, pad_s=0.3, max_s=28.0)
    assert spans[0] == pytest.approx((0.0, 1.3))  # clamped at the track start
    long = spans[1:-1]
    assert len(long) == 3 and all(b - a <= 28.0 + 1e-9 for a, b in long)
    assert long[0][0] == pytest.approx(4.7) and long[-1][1] == pytest.approx(65.3)


def test_words_get_track_times_and_blank_or_endless_words_are_handled():
    audio = np.zeros(SR * 10, dtype=np.float32)

    def recognize(clips, sr):
        assert sr == SR
        return [[(0.3, 0.6, " haan "), (0.6, None, "ji"), (0.7, 0.8, "  ")] for _ in clips]

    words = transcribe(audio, SR, [IPU(2.0, 2.5)], recognize, AsrConfig(pad_s=0.3))
    assert words == [{"start": 2.0, "end": 2.3, "word": "haan"}, {"start": 2.3, "end": 2.8, "word": "ji"}]  # None end -> clip end (1.1 s)


def test_loops_are_detected_but_short_repeats_are_not():
    assert looping("gaya " * 40)
    assert not looping("haan haan") and not looping("ji ji ji")
    assert not looping("aapka order kal tak pahunch jayega, tracking number aapko SMS par mil jayega")


def test_looping_segments_are_decoded_again_and_the_better_text_kept():
    asr = WhisperASR.__new__(WhisperASR)
    asr.retried = 0
    calls = []

    def decode(clips, sr, extra):
        calls.append(extra)
        if not extra:  # first pass: the second clip loops
            return [[(0.0, 0.3, "haan")], [(0.1 * i, 0.1 * i + 0.1, "gaya") for i in range(50)]]
        return [[(0.0, 0.4, "dekha"), (0.4, 0.8, "gaya")]]

    asr._decode = decode
    out = asr([np.zeros(SR), np.zeros(SR * 5)])
    assert calls == [{}, {"no_repeat_ngram_size": 3}] and asr.retried == 1
    assert out[0] == [(0.0, 0.3, "haan")] and [t for _, _, t in out[1]] == ["dekha", "gaya"]


def test_words_file_round_trip(tmp_path):
    words = {"agent": [{"start": 0.1, "end": 0.4, "word": "नमस्ते"}], "customer": [{"start": 1.0, "end": 1.2, "word": "haan"}]}
    write_words(tmp_path / "words.jsonl", words, AsrConfig(model="m"))
    cfg, back = read_words(tmp_path / "words.jsonl")
    assert cfg["model"] == "m" and back == words


# A scripted session: the agent talks, the customer says "haan ji" (0.8 s, too long to be a
# backchannel by duration alone), later says "ek minute ruko" and goes quiet for 4 s.
AGENT = [(0.0, 3.0), (3.6, 6.0)]
CUSTOMER = [(1.2, 2.0), (7.0, 8.0), (12.0, 13.5)]


class FakeVAD:
    def probs(self, batch, sr):
        hop = hop_s(sr)
        p = np.zeros((2, int(batch.shape[1] / sr / hop)))
        for row, ipus in enumerate((AGENT, CUSTOMER)):
            for a, b in ipus:
                p[row, int(a / hop) : int(b / hop)] = 0.9
        return p


def fake_recognize(clips, sr):
    """Words by clip length (each IPU plus 0.3 s of padding on both sides)."""
    out = []
    for c in clips:
        dur = len(c) / sr
        text = {1.4: "haan ji", 1.6: "ek minute ruko"}.get(round(dur, 1), "aapka order confirm ho gaya hai")
        ws = text.split()
        step = (dur - 0.6) / len(ws)
        out.append([(0.3 + i * step, 0.3 + (i + 1) * step, w) for i, w in enumerate(ws)])
    return out


def _label_session_module():
    spec = importlib.util.spec_from_file_location("label_session", REPO / "scripts" / "label_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_transcript_turns_on_the_text_rules_and_is_cached(tmp_path):
    ls = _label_session_module()
    session = tmp_path / "raw" / "S900"
    session.mkdir(parents=True)
    for role in ("agent", "customer"):
        sf.write(session / f"S900_spk1_{role}.wav", np.zeros(SR * 16, dtype=np.float32), SR)
    cfg = LabelConfig()

    plain = ls.label_session(session, tmp_path / "plain", cfg, FakeVAD())
    assert plain["asr"] is None and plain["counts"]["customer"][BC] == 0  # 0.8 s with no text: not a backchannel

    out = tmp_path / "labels"
    s = ls.label_session(session, out, cfg, FakeVAD(), fake_recognize, AsrConfig(model="fake"))
    events = read_events(out / "S900" / "events.jsonl")
    bc = [e for e in events if e.type == BC]
    assert len(bc) == 1 and bc[0].channel == "customer" and bc[0].text == "haan ji" and bc[0].bc_type is not None
    assert [e.text for e in events if e.type == HOLD_REQ] == ["ek minute ruko"]
    assert s["asr"]["model"] == "fake" and s["n_words"]["customer"] == 2 + 3 + 6

    def fail(clips, sr):
        raise AssertionError("ASR should not run again")

    again = ls.label_session(session, out, cfg, FakeVAD(), fail, AsrConfig(model="fake"))
    assert again["counts"] == s["counts"]  # the cached words.jsonl was used
