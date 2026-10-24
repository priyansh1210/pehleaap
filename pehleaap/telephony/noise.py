"""Background noise at a target SNR (plan §4.3).

SNR is measured against the speech's *active* level (frames within 30 dB of the loudest), not
its plain RMS: a customer track is mostly silence, and its RMS would make every SNR too harsh.

Noise comes from audio files under data/external/noise/ (MUSAN, DEMAND, ESC-50, self-recorded
street noise; licenses checked before use) or, when none are there, from synthetic colored
noise, so tests and smoke runs need no download.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from pehleaap.audio.io import load_audio
from pehleaap.paths import EXTERNAL_DIR

NOISE_DIR = EXTERNAL_DIR / "noise"
NOISE_EXTS = {".wav", ".flac"}
SYNTHETIC = ("white", "pink", "brown", "hum")
NOMINAL_LEVEL_DB = -26.0  # active speech level of a nominal phone talker (dBFS); used when a track has no speech


def active_level_db(x: np.ndarray, sr: int, frame_s: float = 0.02, range_db: float = 30.0) -> float | None:
    """Mean power (dBFS) of the frames within `range_db` of the loudest, or None if there's no signal."""
    hop = int(frame_s * sr)
    n = len(x) // hop
    if n == 0:
        return None
    power = (x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1)
    peak = power.max()
    if peak < 1e-10:  # below -100 dBFS: silence
        return None
    return float(10 * np.log10(power[power >= peak * 10 ** (-range_db / 10)].mean()))


def synthetic_noise(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS colored noise: white, pink (1/f), brown (1/f^2) or mains hum (50 Hz and harmonics, plus hiss)."""
    if kind == "hum":
        t = np.arange(n) / sr
        phases = rng.uniform(0, 2 * np.pi, 6)
        y = sum(np.sin(2 * np.pi * 50 * k * t + phases[k - 1]) / k for k in range(1, 7))
        y = y + 0.1 * rng.standard_normal(n)
    else:
        slope = {"white": 0.0, "pink": 0.5, "brown": 1.0}[kind]  # amplitude ~ f^-slope
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / sr)
        spec[1:] /= f[1:] ** slope
        spec[0] = 0
        y = np.fft.irfft(spec, n)
    return (y / (np.sqrt(np.mean(y**2)) + 1e-12)).astype(np.float32)


def held_out(name: str, test_fraction: float = 0.2) -> bool:
    """Whether a noise file belongs to the evaluation part (fixed by a hash of its path)."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) < test_fraction * 16**8


class NoiseBank:
    """Noise sources by name: the files under `root` (relative paths), or "synthetic:<kind>" if there are none.

    `part="train"` or `"test"` keeps 80% or the other 20% of the files, so training and evaluation
    never share a noise recording (synthetic noise is generated fresh, so it isn't split).
    """

    def __init__(self, root: str | Path | None = NOISE_DIR, sr: int = 16000, part: str | None = None):
        self.sr = sr
        self.root = Path(root) if root else None
        files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in NOISE_EXTS) if self.root and self.root.is_dir() else []
        self.files = [p.relative_to(self.root).as_posix() for p in files]
        if part is not None:
            if part not in ("train", "test"):
                raise ValueError(f"part must be 'train' or 'test', not {part!r}")
            self.files = [f for f in self.files if held_out(f) == (part == "test")]
        self.names = self.files or [f"synthetic:{k}" for k in SYNTHETIC]

    def segment(self, name: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """`n` samples of unit-RMS noise from source `name`, from a random offset (looped if the file is short)."""
        if name.startswith("synthetic:"):
            return synthetic_noise(name.split(":", 1)[1], n, self.sr, rng)
        x = self._load(name)
        start = int(rng.integers(len(x)))
        y = np.resize(np.roll(x, -start), n)  # loop from a random offset
        return (y / (np.sqrt(np.mean(y**2)) + 1e-12)).astype(np.float32)

    @lru_cache(maxsize=64)
    def _load(self, name: str) -> np.ndarray:
        x, _ = load_audio(self.root / name, sr=self.sr)
        if not len(x) or np.sqrt(np.mean(x**2)) < 1e-6:
            raise ValueError(f"noise file {name} is empty or silent")
        return x


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, speech_level_db: float | None) -> np.ndarray:
    """speech + noise, the noise scaled to `snr_db` below the speech's active level (nominal if None)."""
    level = NOMINAL_LEVEL_DB if speech_level_db is None else speech_level_db
    gain = 10 ** ((level - snr_db) / 20) / (np.sqrt(np.mean(noise.astype(np.float64) ** 2)) + 1e-12)
    return (speech + gain * noise).astype(np.float32)
