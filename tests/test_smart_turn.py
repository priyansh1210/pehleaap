import numpy as np

from pehleaap.baselines.smart_turn import last_n_seconds


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
