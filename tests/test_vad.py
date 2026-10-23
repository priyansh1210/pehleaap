import numpy as np
import pytest

from pehleaap.audio.vad import SileroVAD, hop_s, speech_segments


def _speechlike(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 2).astype(np.float32)


@pytest.mark.parametrize("sr", [16000, 8000])
def test_probs_match_silero_reference_wrapper(sr):
    torch = pytest.importorskip("torch")
    from silero_vad import load_silero_vad

    x = _speechlike(3.3, sr)  # not a whole number of frames
    ours = SileroVAD().probs(x, sr)
    theirs = load_silero_vad(onnx=True).audio_forward(torch.from_numpy(x), sr).numpy()[0]
    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() < 1e-5


def test_batch_matches_single_tracks():
    vad = SileroVAD()
    x, y = _speechlike(2.0, 16000, seed=1), _speechlike(2.0, 16000, seed=2)
    both = vad.probs(np.stack([x, y]), 16000)
    assert np.allclose(both[0], vad.probs(x, 16000), atol=1e-6)
    assert np.allclose(both[1], vad.probs(y, 16000), atol=1e-6)


def test_silence_has_no_speech():
    p = SileroVAD().probs(np.zeros(16000 * 2, np.float32), 16000)
    assert len(speech_segments(p, hop_s(16000))) == 0


def test_hysteresis_keeps_speech_between_thresholds():
    p = np.array([0.1, 0.6, 0.4, 0.4, 0.2, 0.1, 0.7, 0.9])
    segs = speech_segments(p, 0.032, threshold=0.5, neg_threshold=0.35)
    # 0.4 sits between the thresholds, so speech continues until the 0.2 frame; the last run is open-ended.
    assert np.allclose(segs, [[0.032, 0.128], [0.192, 0.256]])
