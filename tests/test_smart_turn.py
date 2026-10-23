import numpy as np

from pehleaap.baselines.turn_models import WaveformOnnx, last_n_seconds


def test_short_clips_are_left_padded():
    x = np.ones(16000, dtype=np.float32)
    y = last_n_seconds(x)
    assert len(y) == 8 * 16000
    assert y[: 7 * 16000].sum() == 0  # silence first
    assert y[-16000:].sum() == 16000  # speech at the end


def test_long_clips_keep_the_last_8_seconds():
    x = np.arange(10 * 16000, dtype=np.float32)
    y = last_n_seconds(x)
    assert len(y) == 8 * 16000
    assert y[-1] == x[-1]
    assert y[0] == x[2 * 16000]


def test_waveform_standardisation_uses_real_samples_only():
    # Chhotu's contract: standardise over the real samples, then left-pad with zeros.
    model = WaveformOnnx.__new__(WaveformOnnx)
    model.standardize = True
    x = np.linspace(-1, 3, 16000, dtype=np.float32)
    y = model.prepare(x)
    assert np.all(y[: -16000] == 0)
    assert abs(y[-16000:].mean()) < 1e-5 and abs(y[-16000:].std() - 1) < 1e-3
