"""One pehleaap job on a Kaggle session. scripts/kaggle_job.py fills in COMMIT and CMD and pushes this file.

Clones the repo at COMMIT, installs the pinned environment (Python 3.11.9 + requirements.txt) into a venv
with uv, fetches a static ffmpeg with AMR-NB, then runs CMD from the repo root. Everything in /kaggle/working
is saved as the job's output: job.log, env_check.md, job.json, and the repo's runs/ folder.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = None  # set by kaggle_job.py
CMD = None  # set by kaggle_job.py

REPO_URL = "https://github.com/priyansh1210/pehleaap"
UV = "uv==0.12.24"
PYTHON = "3.11.9"
OUT = Path("/kaggle/working")
SRC = Path("/tmp/pehleaap")  # outside OUT, so the clone and the venv aren't saved as output
VENV = Path("/tmp/venv")
# Ubuntu's ffmpeg has libgsm but no AMR-NB; this static build has AMR-NB but no libgsm (checked on Kaggle
# 2026-10-24, as were BtbN's builds), so codecs.py uses each where needed. Pinned by hash.
FFMPEG_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
FFMPEG_SHA256 = "abda8d77ce8309141f83ab8edf0596834087c52467f6badf376a6a2a4c87cf67"
FFMPEG_BIN = Path("/tmp/ffmpeg-7.0.2-amd64-static/ffmpeg")

LOG = open(OUT / "job.log", "w", encoding="utf-8", buffering=1)


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.write(msg + "\n")


def step(title: str, args: list, cwd=None, env=None, check: bool = True) -> tuple[int, str]:
    args = [str(a) for a in args]
    say(f"\n=== {title}: {' '.join(args)}")
    t = time.time()
    p = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        LOG.write(line)
        lines.append(line)
    rc = p.wait()
    say(f"=== {title}: exit {rc} after {time.time() - t:.0f} s")
    if check and rc:
        sys.exit(rc)
    return rc, "".join(lines)


def static_ffmpeg():
    """Path to the pinned static ffmpeg, or None (system ffmpeg, no AMR-NB) if it can't be fetched or changed upstream."""
    archive = Path("/tmp/ffmpeg.tar.xz")
    rc, _ = step("ffmpeg", ["curl", "-sSL", "--retry", "5", "--max-time", "600", "-o", archive, FFMPEG_URL], check=False)
    if rc or hashlib.sha256(archive.read_bytes()).hexdigest() != FFMPEG_SHA256:
        say("WARNING: static ffmpeg download failed or changed upstream; using the system ffmpeg (no AMR-NB). Update FFMPEG_SHA256.")
        return None
    step("ffmpeg unpack", ["tar", "-xJf", archive, "-C", "/tmp"])
    return str(FFMPEG_BIN)


def main() -> None:
    started = time.time()
    step("inputs", ["find", "/kaggle/input", "-maxdepth", "3"], check=False)
    step("clone", ["git", "clone", "--quiet", REPO_URL, SRC])
    step("checkout", ["git", "-C", SRC, "checkout", "--quiet", COMMIT])
    step("uv", [sys.executable, "-m", "pip", "install", "--quiet", UV])
    uv = [sys.executable, "-m", "uv"]
    step("venv", uv + ["venv", "--quiet", VENV, "--python", PYTHON])
    py = VENV / "bin" / "python"
    # Like pip: look in every index for each package (uv otherwise stops at the PyTorch index).
    step("install", uv + ["pip", "install", "--python", py, "--index-strategy", "unsafe-best-match", "-r", SRC / "requirements.txt"])
    step("install pehleaap", uv + ["pip", "install", "--python", py, "--no-deps", "-e", SRC])

    env = dict(os.environ, PATH=f"{VENV / 'bin'}:{os.environ['PATH']}", VIRTUAL_ENV=str(VENV))
    env.update(WANDB_MODE="offline", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)  # keep Kaggle's own site-packages out of the venv
    ffmpeg = static_ffmpeg()
    if ffmpeg:
        env["PEHLEAAP_FFMPEG_EXTRA"] = ffmpeg  # for AMR-NB only; the system ffmpeg stays first (it has libgsm)

    _, table = step("env_check", ["python", "scripts/env_check.py", "--markdown"], cwd=SRC, env=env, check=False)
    (OUT / "env_check.md").write_text(table, encoding="utf-8")

    rc, _ = step("job", CMD, cwd=SRC, env=env, check=False)
    if (SRC / "runs").is_dir():
        shutil.copytree(SRC / "runs", OUT / "runs", dirs_exist_ok=True)
    job = {"commit": COMMIT, "cmd": CMD, "exit_code": rc, "minutes": round((time.time() - started) / 60, 1)}
    (OUT / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    sys.exit(rc)


if __name__ == "__main__":
    main()
