"""Telephone speech codecs via ffmpeg, with delay compensation.

Encoding and decoding adds a fixed delay (codec lookahead, priming samples). Left in, it
would shift every turn boundary, so each codec's delay is measured once on a calibration
signal and removed. The output is the same length as the input and aligned to it, so
timing labels carry over unchanged.

All audio here is 8 kHz mono float32.
"""

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from pehleaap.audio.align import estimate_offset
from pehleaap.audio.io import resample
from pehleaap.paths import ffmpeg_bin

SR = 8000

# name -> (container extension, encoder arguments)
CODECS: dict[str, tuple[str, list[str]]] = {
    "g711_mulaw": ("wav", ["-c:a", "pcm_mulaw"]),
    "g711_alaw": ("wav", ["-c:a", "pcm_alaw"]),
    "amrnb_4.75": ("amr", ["-c:a", "libopencore_amrnb", "-b:a", "4750"]),
    "amrnb_7.95": ("amr", ["-c:a", "libopencore_amrnb", "-b:a", "7950"]),
    "amrnb_12.2": ("amr", ["-c:a", "libopencore_amrnb", "-b:a", "12200"]),
    "gsm_fr": ("gsm", ["-c:a", "libgsm"]),
    "opus_8k": ("opus", ["-c:a", "libopus", "-b:a", "12k", "-application", "voip"]),
}


def _roundtrip(x: np.ndarray, codec: str) -> np.ndarray:
    """Encode and decode with ffmpeg, without any delay compensation."""
    ext, enc_args = CODECS[codec]
    ff = ffmpeg_bin(encoder=enc_args[enc_args.index("-c:a") + 1])
    with tempfile.TemporaryDirectory() as tmp:
        src, coded, dec = Path(tmp) / "in.wav", Path(tmp) / f"coded.{ext}", Path(tmp) / "out.wav"
        sf.write(src, np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")
        subprocess.run([ff, "-v", "error", "-y", "-i", str(src), "-ar", str(SR), "-ac", "1", *enc_args, str(coded)], check=True)
        subprocess.run([ff, "-v", "error", "-y", "-i", str(coded), "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float32")
    return y


def _calibration_signal(seconds: float = 2.0) -> np.ndarray:
    """Speech-like test signal: pitch-modulated pulse train in bursts, through a smoothing filter."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 40 * np.sin(2 * np.pi * 3 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0) / SR), prepend=0) > 0).astype(np.float32)
    voiced = np.convolve(pulses, np.hanning(40), mode="same")
    envelope = (np.sin(2 * np.pi * 2.5 * t) > -0.3).astype(np.float32)
    return (0.3 * voiced * envelope + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


@lru_cache(maxsize=None)
def codec_delay_samples(codec: str) -> int:
    """Fixed delay this codec adds, in samples at 8 kHz (positive: output lags input)."""
    x = _calibration_signal()
    y = _roundtrip(x, codec)
    n = min(len(x), len(y))
    return int(round(estimate_offset(x[:n], y[:n], SR, max_lag_s=0.1) * SR))


def apply_codec(x: np.ndarray, codec: str) -> np.ndarray:
    """Encode/decode 8 kHz audio and return it delay-compensated, with the same length as `x`."""
    y = _roundtrip(x, codec)
    d = codec_delay_samples(codec)
    y = y[d:] if d >= 0 else np.concatenate([np.zeros(-d, dtype=np.float32), y])
    return fit_length(y, len(x))


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])
    return y[:n].astype(np.float32)


def apply_codec_many(clips: list[np.ndarray], codec: str, gap_s: float = 0.5) -> list[np.ndarray]:
    """`apply_codec` on many 8 kHz clips in one ffmpeg run (much faster than one run per clip).

    The clips are joined with `gap_s` of silence, so codec state can't carry from one clip into
    the next, coded once and split again. Because the codec delay is compensated, the split
    points stay exact.
    """
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts, bounds, pos = [gap], [], len(gap)
    for c in clips:
        parts += [c.astype(np.float32), gap]
        bounds.append((pos, pos + len(c)))
        pos += len(c) + len(gap)
    coded = apply_codec(np.concatenate(parts), codec)
    return [coded[s:e] for s, e in bounds]


def narrowband_many(clips: list[np.ndarray], codec: str, sr: int = 16000, gap_s: float = 0.5) -> list[np.ndarray]:
    """Phone channel for many clips at once: sr -> 8 kHz -> codec -> back to sr, same lengths.

    This is only band-limiting plus codec; noise, echo and packet loss come from the full
    chain in `pipeline.py`.
    """
    coded = apply_codec_many([resample(c.astype(np.float32), sr, SR) for c in clips], codec, gap_s)
    return [fit_length(resample(y, SR, sr), len(c)) for y, c in zip(coded, clips)]
