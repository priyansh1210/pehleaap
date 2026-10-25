import importlib.util
from pathlib import Path

import numpy as np

from pehleaap.audio.vad import hop_s
from pehleaap.labeling.config import LabelConfig

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def _script():
    spec = importlib.util.spec_from_file_location("make_fleurs_eot", REPO / "scripts" / "make_fleurs_eot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeVAD:
    def __init__(self, segments):
        self.segments = segments

    def probs(self, x, sr):
        hop = hop_s(sr)
        p = np.zeros(int(len(x) / sr / hop))
        for a, b in self.segments:
            p[int(a / hop) : int(b / hop)] = 0.9
        return p


def test_finished_clip_ends_after_the_speech_and_the_pause_clip_inside_the_pause():
    fl = _script()
    x = np.arange(SR * 10, dtype=np.float32)
    out = fl.clips_for(x, "r1.wav", FakeVAD([(0.5, 3.0), (3.6, 6.0), (6.12, 8.0)]), LabelConfig(), 0.25, 0.2, 1.0)
    (fin, clip_f, _), (unf, clip_u, kind) = out
    assert fin and not unf
    assert abs(len(clip_f) / SR - 8.2) < 0.05  # speech end + 0.2 s
    assert kind.startswith("pause@") and abs(len(clip_u) / SR - 3.2) < 0.05  # 0.2 s into the 0.6 s pause; 0.12 s gap is too short


def test_no_long_pause_means_only_a_finished_clip():
    fl = _script()
    x = np.zeros(SR * 6, dtype=np.float32)
    assert [k for _, _, k in fl.clips_for(x, "r2.wav", FakeVAD([(0.3, 0.9), (1.2, 5.0)]), LabelConfig(), 0.25, 0.2, 1.0)] == ["end"]
    assert fl.clips_for(x, "r3.wav", FakeVAD([]), LabelConfig(), 0.25, 0.2, 1.0) == []
