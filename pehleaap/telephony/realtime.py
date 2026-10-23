"""Real-time phone line for live calls: 16 kHz in -> 8 kHz G.711 mu-law -> 16 kHz out.

ffmpeg round trips (codecs.py) are fine offline but far too slow per 20 ms frame, so the live
demo uses this streaming version, in numpy only:

* G.711 mu-law with ffmpeg's exact encoding table (nearest quantisation level), so live
  audio is bit-identical to the offline `g711_mulaw` training chain.
* Band-limiting with short linear-phase FIR filters cut at 3.4 kHz (the telephone band edge).
  The delay is about 6 ms in total, and every input sample produces an output sample.
  (soxr's streaming resamplers hold back 120-160 ms, which would distort turn timing.)
"""

import numpy as np

NARROWBAND_SR = 8000


def ulaw_to_linear(codes: np.ndarray) -> np.ndarray:
    """8-bit G.711 mu-law codes -> int16 PCM (same formula as ffmpeg's ulaw2linear)."""
    u = ~np.asarray(codes).astype(np.int32) & 0xFF
    mag = (((u & 0x0F) << 3) + 0x84) << ((u >> 4) & 0x07)
    return np.where(u & 0x80, 0x84 - mag, mag - 0x84).astype(np.int16)


def _build_ulaw_table() -> np.ndarray:
    """Port of ffmpeg's build_xlaw_table: 14-bit linear index -> nearest mu-law code."""
    mask = 0xFF
    table = np.zeros(16384, dtype=np.uint8)
    table[8192] = mask
    levels = ulaw_to_linear(np.arange(256) ^ mask).astype(np.int32)  # levels[i] = magnitude of code i
    j = 1
    for i in range(127):
        v = (levels[i] + levels[i + 1] + 4) >> 3
        while j < v:
            table[8192 - j] = i ^ (mask ^ 0x80)
            table[8192 + j] = i ^ mask
            j += 1
    while j < 8192:
        table[8192 - j] = 127 ^ (mask ^ 0x80)
        table[8192 + j] = 127 ^ mask
        j += 1
    table[0] = table[1]
    return table


_ULAW_TABLE = _build_ulaw_table()


def linear_to_ulaw(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM -> 8-bit G.711 mu-law codes."""
    return _ULAW_TABLE[(np.asarray(pcm).astype(np.int32) + 32768) >> 2]


def _lowpass(sr: int, cutoff_hz: float = 3400.0, taps_per_ratio: int = 47) -> np.ndarray:
    ratio = sr // NARROWBAND_SR
    taps = taps_per_ratio * ratio + 1
    n = np.arange(taps) - (taps - 1) / 2
    h = (2 * cutoff_hz / sr) * np.sinc(2 * cutoff_hz / sr * n) * np.kaiser(taps, 8.0)
    return (h / h.sum()).astype(np.float32)


class _StreamingFIR:
    def __init__(self, h: np.ndarray):
        self.h = h
        self.state = np.zeros(len(h) - 1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.state, x])
        self.state = buf[len(buf) - len(self.state) :]
        return np.convolve(buf, self.h, mode="valid")


class PhoneLine:
    """Stateful narrowband G.711 channel for a stream of int16 frames at `sr` (a multiple of 8 kHz)."""

    def __init__(self, sr: int = 16000):
        if sr % NARROWBAND_SR:
            raise ValueError("sample rate must be a multiple of 8000")
        self.sr = sr
        self.ratio = sr // NARROWBAND_SR
        h = _lowpass(sr)
        self._down = _StreamingFIR(h)
        self._up = _StreamingFIR(h * self.ratio)  # zero-stuffing loses 1/ratio of the gain
        self._phase = 0  # index of the next sample to keep when decimating

    @property
    def delay_samples(self) -> int:
        return len(self._down.h) - 1  # two linear-phase filters, (taps-1)/2 each

    def process(self, frame: np.ndarray) -> np.ndarray:
        y = self._down(frame.astype(np.float32))
        kept = y[self._phase :: self.ratio]
        self._phase = (self._phase - len(y)) % self.ratio
        pcm8 = np.clip(np.round(kept), -32768, 32767).astype(np.int16)
        decoded = ulaw_to_linear(linear_to_ulaw(pcm8)).astype(np.float32)
        stuffed = np.zeros(len(decoded) * self.ratio, dtype=np.float32)
        stuffed[:: self.ratio] = decoded
        out = self._up(stuffed)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)
