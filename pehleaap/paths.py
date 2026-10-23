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

# Where the runbooks unzip the ffmpeg full build: D:\tools on the Dev box, ~\tools on the GPU box.
_FFMPEG_BUILD = Path("ffmpeg-9.0.2-full_build-shared") / "bin" / "ffmpeg.exe"
_DEFAULT_FFMPEGS = [Path(r"D:\tools") / _FFMPEG_BUILD, Path.home() / "tools" / _FFMPEG_BUILD]


def ffmpeg_bin() -> str:
    """Return the ffmpeg executable: $PEHLEAAP_FFMPEG, then PATH, then the runbook install locations."""
    env = os.environ.get("PEHLEAAP_FFMPEG")
    if env:
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    for path in _DEFAULT_FFMPEGS:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "ffmpeg not found. Install a full build (needs libopencore_amrnb) and set PEHLEAAP_FFMPEG."
    )
