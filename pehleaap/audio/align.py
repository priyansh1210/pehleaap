"""Track alignment for two-channel recordings.

Each speaker is recorded on their own track. Turn-timing labels are only valid if the
tracks share a clock, so every session is checked with a shared acoustic event (clap or
beep) near the start and the end:

* offset: how much later the event appears in `sig` than in `ref`
* drift:  how much that offset changes between start and end, in parts per million

A single-clock recorder (Craig, Zoom multitrack, server-side relay) should show drift ~0.
"""

from dataclasses import dataclass

import numpy as np

from pehleaap.audio.io import resample


def estimate_offset(ref: np.ndarray, sig: np.ndarray, sr: int, max_lag_s: float = 1.0) -> float:
    """Return the lag in seconds of `sig` relative to `ref` (positive: event later in `sig`).

    Uses GCC-PHAT, which gives a sharp peak for impulsive events such as claps, with
    parabolic interpolation for sub-sample precision.
    """
    n = len(ref) + len(sig)
    nfft = 1 << (n - 1).bit_length()
    cross = np.fft.rfft(sig, nfft) * np.conj(np.fft.rfft(ref, nfft))
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross, nfft)

    max_lag = min(int(max_lag_s * sr), nfft // 2 - 1)
    # Lags -max_lag..+max_lag, with negative lags wrapped at the end of the buffer.
    window = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(window))
    shift = 0.0
    if 0 < k < len(window) - 1:
        y0, y1, y2 = window[k - 1], window[k], window[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
    return (k - max_lag + shift) / sr


@dataclass
class DriftReport:
    offset_start_s: float
    offset_end_s: float
    drift_ppm: float
    span_s: float

    @property
    def drift_ms(self) -> float:
        """Total timing error accumulated across the session."""
        return (self.offset_end_s - self.offset_start_s) * 1000.0


def _loudest_time(x: np.ndarray, sr: int) -> float:
    """Centre (s) of the loudest 10 ms frame: where the clap is within a window."""
    hop = sr // 100
    n = len(x) // hop
    energy = (x[: n * hop].reshape(n, hop) ** 2).sum(axis=1)
    return (int(np.argmax(energy)) + 0.5) * hop / sr


def measure_drift(
    ref: np.ndarray, sig: np.ndarray, sr: int, win_s: float = 20.0, max_lag_s: float = 1.0
) -> DriftReport:
    """Measure offset in the first and last `win_s` seconds (each must contain the shared event)."""
    total = min(len(ref), len(sig))
    win = int(win_s * sr)
    if 2 * win > total:
        raise ValueError(f"recording too short for two {win_s}s windows")
    head, tail = slice(0, win), slice(total - win, total)
    start = estimate_offset(ref[head], sig[head], sr, max_lag_s)
    end = estimate_offset(ref[tail], sig[tail], sr, max_lag_s)
    # Drift is spread over the time between the two events, not between the windows.
    span = (total - win) / sr + _loudest_time(ref[tail], sr) - _loudest_time(ref[head], sr)
    return DriftReport(start, end, (end - start) / span * 1e6, span)


def correct_drift(sig: np.ndarray, sr: int, drift_ppm: float) -> np.ndarray:
    """Undo a constant clock-rate error so events in `sig` stop drifting relative to `ref`."""
    if drift_ppm == 0:
        return sig
    return resample(sig, sr * (1.0 + drift_ppm * 1e-6), sr)


def shift(sig: np.ndarray, sr: int, offset_s: float) -> np.ndarray:
    """Advance `sig` by `offset_s` (positive removes leading samples), keeping its length."""
    k = int(round(offset_s * sr))
    if k > 0:
        return np.concatenate([sig[k:], np.zeros(k, dtype=sig.dtype)])
    if k < 0:
        return np.concatenate([np.zeros(-k, dtype=sig.dtype), sig[:k]])
    return sig
