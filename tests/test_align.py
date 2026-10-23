import numpy as np

from pehleaap.audio.align import correct_drift, estimate_offset, measure_drift
from pehleaap.audio.io import resample

SR = 16000


def _session(seconds=120.0, events_s=(5.0, 115.0), seed=0):
    """Low-level noise with 10 ms broadband bursts (claps) at the given times."""
    rng = np.random.default_rng(seed)
    x = 0.001 * rng.standard_normal(int(seconds * SR)).astype(np.float32)
    for t in events_s:
        i = int(t * SR)
        x[i : i + SR // 100] += 0.5 * rng.standard_normal(SR // 100).astype(np.float32)
    return x


def _delay(x, seconds):
    k = int(round(seconds * SR))
    return np.concatenate([np.zeros(k, dtype=x.dtype), x[:-k]])


def test_offset_recovers_known_delay():
    ref = _session()
    sig = _delay(ref, 0.123) + 0.001 * np.random.default_rng(1).standard_normal(len(ref)).astype(np.float32)
    assert abs(estimate_offset(ref[: 20 * SR], sig[: 20 * SR], SR) - 0.123) < 0.0005


def test_drift_is_measured_and_corrected():
    ref = _session()
    # The second device's clock runs 50 ppm fast: events land progressively later.
    sig = _delay(resample(ref, SR, SR * (1 + 50e-6)), 0.123)
    report = measure_drift(ref, sig, SR)
    assert abs(report.offset_start_s - 0.123) < 0.001
    assert abs(report.drift_ppm - 50) < 5

    fixed = correct_drift(sig, SR, report.drift_ppm)
    after = measure_drift(ref, fixed, SR)
    assert abs(after.drift_ms) < 0.5


def test_single_clock_recording_has_no_drift():
    ref = _session()
    sig = _delay(ref, 0.050)
    assert abs(measure_drift(ref, sig, SR).drift_ms) < 0.2
