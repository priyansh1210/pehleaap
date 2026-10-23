import numpy as np

from pehleaap.audio.align import estimate_offset
from pehleaap.telephony.codecs import apply_codec
from pehleaap.telephony.realtime import PhoneLine, linear_to_ulaw, ulaw_to_linear


def test_ulaw_is_bit_identical_to_ffmpeg_pcm_mulaw():
    rng = np.random.default_rng(0)
    pcm = np.clip(rng.normal(0, 6000, 8000), -32768, 32767).astype(np.int16)
    ours = ulaw_to_linear(linear_to_ulaw(pcm)).astype(np.int32)
    ffmpeg = np.round(apply_codec(pcm.astype(np.float32) / 32768.0, "g711_mulaw") * 32768.0).astype(np.int32)
    assert np.array_equal(ours, ffmpeg)


def test_ulaw_is_monotonic_with_the_standard_peak():
    pcm = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    back = ulaw_to_linear(linear_to_ulaw(pcm)).astype(np.int32)
    assert np.all(np.diff(back) >= 0)
    assert back[-1] == 32124 and back[0] == -32124  # G.711 mu-law peak level


def test_phone_line_is_low_latency_and_band_limited():
    line = PhoneLine(sr=16000)
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0, 3000, 32000), -32768, 32767).astype(np.int16)
    out = np.concatenate([line.process(x[i : i + 320]) for i in range(0, len(x), 320)])  # 20 ms frames
    assert len(out) == len(x)
    # Delay is the two FIR filters only: a few ms, not the 100+ ms of buffering resamplers.
    lag_ms = estimate_offset(x.astype(np.float32), out.astype(np.float32), 16000, max_lag_s=0.05) * 1000
    assert abs(lag_ms - line.delay_samples / 16) < 0.5
    assert line.delay_samples / 16 < 8
    # Narrowband: energy above 4 kHz is gone.
    spec = np.abs(np.fft.rfft(out.astype(np.float32))) ** 2
    freqs = np.fft.rfftfreq(len(out), 1 / 16000)
    assert spec[freqs > 4000].sum() < 1e-3 * spec.sum()


def test_phone_line_frame_size_does_not_matter():
    rng = np.random.default_rng(2)
    x = np.clip(rng.normal(0, 3000, 16000), -32768, 32767).astype(np.int16)
    a = np.concatenate([PhoneLine().process(x)])
    line = PhoneLine()
    b = np.concatenate([line.process(x[i : i + 333]) for i in range(0, len(x), 333)])  # odd frame sizes
    assert np.max(np.abs(a.astype(np.int32) - b.astype(np.int32))) <= 1
