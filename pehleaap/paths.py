"""Repository paths and external tool lookup, shared by the Dev box and the GPU box."""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
MANIFEST_DIR = DATA_DIR / "manifests"
WEIGHTS_DIR = REPO_ROOT / "weights"
RUNS_DIR = REPO_ROOT / "runs"

# Default install location used on the Dev box; override with PEHLEAAP_FFMPEG on other machines.
_DEFAULT_FFMPEG = Path(r"D:\tools\ffmpeg-9.0.2-full_build-shared\bin\ffmpeg.exe")


def ffmpeg_bin() -> str:
    """Return the ffmpeg executable: $PEHLEAAP_FFMPEG, then PATH, then the Dev box default."""
    env = os.environ.get("PEHLEAAP_FFMPEG")
    if env:
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    if _DEFAULT_FFMPEG.exists():
        return str(_DEFAULT_FFMPEG)
    raise FileNotFoundError(
        "ffmpeg not found. Install a full build (needs libopencore_amrnb) and set PEHLEAAP_FFMPEG."
    )
