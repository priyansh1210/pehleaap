"""Repository paths and external tool lookup, shared by the Dev box and the GPU box."""

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
MANIFEST_DIR = DATA_DIR / "manifests"
WEIGHTS_DIR = REPO_ROOT / "weights"
RUNS_DIR = REPO_ROOT / "runs"

# Where the runbooks unzip the ffmpeg full build: D:\tools on the Dev box, ~\tools on the GPU box.
_FFMPEG_BUILD = Path("ffmpeg-9.0.2-full_build-shared") / "bin" / "ffmpeg.exe"
_DEFAULT_FFMPEGS = [Path(r"D:\tools") / _FFMPEG_BUILD, Path.home() / "tools" / _FFMPEG_BUILD]


def _ffmpeg_candidates() -> list[str]:
    found = [os.environ.get("PEHLEAAP_FFMPEG"), shutil.which("ffmpeg")]
    found += [str(p) for p in _DEFAULT_FFMPEGS if p.exists()]
    found += os.environ.get("PEHLEAAP_FFMPEG_EXTRA", "").split(os.pathsep)
    return [f for f in dict.fromkeys(found) if f]


@lru_cache
def _ffmpeg_encoders(ff: str) -> str:
    return subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout


def ffmpeg_bin(encoder: str | None = None) -> str:
    """Return an ffmpeg executable: $PEHLEAAP_FFMPEG, then PATH, then the runbook install locations.

    With `encoder`, return the first of those, or of the builds listed in $PEHLEAAP_FFMPEG_EXTRA
    (os.pathsep-separated), that has it. Kaggle needs two builds: Ubuntu's ffmpeg has libgsm but
    no AMR-NB, and the static builds have AMR-NB but no libgsm.
    """
    candidates = _ffmpeg_candidates()
    if not candidates:
        raise FileNotFoundError("ffmpeg not found. Install a full build (needs libopencore_amrnb) and set PEHLEAAP_FFMPEG.")
    if encoder is None:
        return candidates[0]
    for ff in candidates:
        if re.search(rf"\s{re.escape(encoder)}\s", _ffmpeg_encoders(ff)):
            return ff
    raise FileNotFoundError(f"no ffmpeg with the {encoder} encoder among {candidates}; add one to PEHLEAAP_FFMPEG_EXTRA")
