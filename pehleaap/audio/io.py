"""Audio loading, saving and resampling.

Arrays are float32. Mono audio is shape (n,); multichannel audio is shape (channels, n).
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from pehleaap.paths import ffmpeg_bin


def resample(x: np.ndarray, sr_in: float, sr_out: float) -> np.ndarray:
    """Resample mono (n,) or multichannel (ch, n) audio with soxr (VHQ)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if x.ndim == 1:
        return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)
    return soxr.resample(x.T, sr_in, sr_out, quality="VHQ").T.astype(np.float32)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode formats libsndfile can't read (m4a, aac, webm) to (n, ch) float32 via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        subprocess.run(
            [ffmpeg_bin(), "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)],
            check=True,
        )
        return sf.read(str(wav), dtype="float32", always_2d=True)


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32, optionally downmixing to mono and resampling to `sr`."""
    path = Path(path)
    try:
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except RuntimeError:
        x, file_sr = _ffmpeg_decode(path)
    x = x.mean(axis=1) if mono else x.T
    if sr is not None and sr != file_sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return x.astype(np.float32, copy=False), file_sr


def save_audio(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Write mono (n,) or multichannel (ch, n) audio."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x if x.ndim == 1 else x.T, sr, subtype=subtype)
