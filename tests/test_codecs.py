import numpy as np
import pytest

from pehleaap.audio.align import estimate_offset
from pehleaap.telephony.codecs import CODECS, SR, apply_codec


def _speechlike(seconds=3.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.7 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0) / SR), prepend=0) > 0).astype(np.float32)
    voiced = np.convolve(pulses, np.hanning(32), mode="same")
    gate = (np.sin(2 * np.pi * 1.3 * t) > 0).astype(np.float32)
    return (0.3 * voiced * gate + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_codec_output_is_aligned_and_same_length(codec):
    x = _speechlike()
    y = apply_codec(x, codec)
    assert len(y) == len(x)
    # Labels stay valid only if the codec adds no net delay: within 1 ms.
    assert abs(estimate_offset(x, y, SR, max_lag_s=0.1)) <= 0.001
    # And speech is still where it was. Compare 10 ms log-energy envelopes, not waveforms:
    # CELP codecs (AMR) keep the spectrum and timing but not the waveform's phase.
    assert np.corrcoef(_envelope(x), _envelope(y))[0, 1] > 0.8


def _envelope(x, hop=80):
    n = len(x) // hop
    return np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-8)
