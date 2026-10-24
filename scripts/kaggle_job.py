"""Run one job on Kaggle from the Dev box, wait for it and download its output (plan §3a).

A job is one command run from the root of a fresh clone of this repo at the current commit, with the
pinned environment installed (scripts/kaggle_job_template.py does that part on Kaggle). Each job is a
private Kaggle script kernel named pehleaap-<name>; running a name again pushes a new version.
Output lands in runs/kaggle/<name>/output/: job.log, env_check.md, job.json and the job's runs/ folder.

    python scripts/kaggle_job.py run smoke --gpu t4 -- python -m pytest
    python scripts/kaggle_job.py run labels --dataset <user>/pehleaap-raw -- python scripts/label_session.py ...
    python scripts/kaggle_job.py status smoke
    python scripts/kaggle_job.py fetch smoke
    python scripts/kaggle_job.py quota

The job clones from GitHub, so the commit must be pushed first. Needs a Kaggle API token on the Dev box
(docs/machines.md). GPU sessions use the weekly quota; CPU sessions (the default) don't.
"""

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.paths import REPO_ROOT, RUNS_DIR  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "kaggle_job_template.py"
GPUS = {"t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100", "none": None}
DONE = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
ACTIVE = {"QUEUED", "RUNNING"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def pushed_commit() -> str:
    commit = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"commit {commit[:7]} isn't on GitHub yet: push it first (the job clones it from there)")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("warning: uncommitted changes are not part of the job")
    return commit


def kaggle_api():
    # Importing the package already tries to log in and prints the help on failure; authenticate() below repeats it.
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    if not api.get_config_value(api.CONFIG_NAME_USER):
        raise SystemExit("not logged in to Kaggle: put an API token in %USERPROFILE%\\.kaggle\\access_token (docs/machines.md)")
    return api


def kernel_id(api, name: str) -> str:
    return f"{api.get_config_value(api.CONFIG_NAME_USER)}/pehleaap-{name}"


def job_dir(name: str) -> Path:
    return RUNS_DIR / "kaggle" / name


def build_kernel(name: str, kernel: str, cmd: list[str], gpu: str | None, datasets: list[str]) -> Path:
    src = TEMPLATE.read_text(encoding="utf-8")
    for line, value in (("COMMIT = None", pushed_commit()), ("CMD = None", cmd)):
        assert src.count(line) == 1, line
        src = src.replace(line, f"{line.split(' = ')[0]} = {value!r}")
    folder = job_dir(name) / "kernel"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.py").write_text(src, encoding="utf-8")
    meta = {
        "id": kernel,
        "title": f"pehleaap-{name}",
        "code_file": "job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu is not None,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def status(api, kernel: str) -> tuple[str, str]:
    r = api.kernels_status(kernel)
    return getattr(r.status, "name", str(r.status)), r.failure_message or ""


def wait(api, kernel: str, poll_s: int = 60) -> str:
    """Poll until the new version finishes. Right after a push the status can still show the previous run."""
    t0 = time.time()
    last, seen_active = None, False
    while True:
        state, msg = status(api, kernel)
        seen_active |= state in ACTIVE
        if state != last:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {state}" + (f": {msg}" if msg else ""), flush=True)
            last = state
        if state in DONE and (seen_active or time.time() - t0 > 300):
            return state
        time.sleep(poll_s)


def fetch(api, name: str, kernel: str) -> None:
    out = job_dir(name) / "output"
    out.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, str(out), force=True, quiet=True)
    log = out / "job.log"
    if log.exists():
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    job = out / "job.json"
    print(f"\noutput: {out}" + (f"\njob: {job.read_text(encoding='utf-8')}" if job.exists() else "\n(no job.json: the job didn't finish)"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="push a job and (by default) wait for it; the command goes after --")
    run.add_argument("name", help="job name: lowercase letters, digits and dashes")
    run.add_argument("--gpu", choices=GPUS, default="none")
    run.add_argument("--hours", type=float, help="session time limit (Kaggle's own maximum otherwise)")
    run.add_argument("--dataset", action="append", default=[], help="attach a dataset, owner/slug (repeatable)")
    run.add_argument("--no-wait", action="store_true")
    for action in ("status", "fetch"):
        sub.add_parser(action).add_argument("name")
    sub.add_parser("quota", help="weekly GPU hours used and left")
    args = ap.parse_args(argv)

    api = kaggle_api()
    if args.action == "quota":
        api.quota_view_cli()
        return
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.name):
        raise SystemExit(f"bad job name {args.name!r}: use lowercase letters, digits and dashes")
    kernel = kernel_id(api, args.name)

    if args.action == "status":
        state, msg = status(api, kernel)
        print(f"{kernel}: {state}" + (f": {msg}" if msg else ""))
    elif args.action == "fetch":
        fetch(api, args.name, kernel)
    else:
        if not cmd:
            raise SystemExit("give the job's command after --, e.g. -- python -m pytest")
        folder = build_kernel(args.name, kernel, cmd, GPUS[args.gpu], args.dataset)
        timeout = str(int(args.hours * 3600)) if args.hours else None
        r = api.kernels_push(str(folder), timeout=timeout, acc=GPUS[args.gpu])
        if r.error:
            raise SystemExit(f"push failed: {r.error}")
        print(f"pushed {kernel} version {r.versionNumber} ({args.gpu}): {r.url}")
        if args.no_wait:
            return
        state = wait(api, kernel)
        fetch(api, args.name, kernel)
        sys.exit(0 if state == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
