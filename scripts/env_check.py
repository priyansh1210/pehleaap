"""Print this machine's specs and toolchain, and optionally verify data manifests.

Run on both the Dev box and the GPU box, and commit the output to docs/machines.md:

    python scripts/env_check.py --markdown >> docs/machines.md
    python scripts/env_check.py --verify-data        # before every training run
"""

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.paths import MANIFEST_DIR, REPO_ROOT, ffmpeg_bin  # noqa: E402


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _version(module: str) -> str:
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except ImportError:
        return "not installed"


def collect() -> dict[str, str]:
    import psutil

    info = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "cpu": _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if platform.system() == "Windows"
        else platform.processor(),
        "cpu_cores": f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical",
        "ram_gb": f"{psutil.virtual_memory().total / 2**30:.1f}",
        "disk_free_gb": f"{shutil.disk_usage(REPO_ROOT).free / 2**30:.0f}",
    }

    # Dedicated VRAM as the driver reports it. Task Manager's "GPU memory" adds shared system
    # RAM on top of this, which is why a card can show "20 GB" while having 8 GB to train with.
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    info["gpu (nvidia-smi)"] = smi or "none"
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda (torch build)"] = torch.version.cuda
            info["gpu dedicated VRAM (GB)"] = f"{props.total_memory / 2**30:.1f}"
            info["gpu compute capability"] = f"{props.major}.{props.minor}"
            # Native bf16 needs Ampere+ (RTX 30/40). Turing (GTX 16xx) only emulates it: use fp16 there.
            info["bf16 native"] = str(torch.cuda.is_bf16_supported(including_emulation=False))
        else:
            info["cuda (torch build)"] = "CUDA not available"
    except ImportError:
        info["torch"] = "not installed"

    for mod in ["numpy", "onnxruntime", "transformers", "datasets", "soxr", "soundfile", "pipecat"]:
        info[mod] = _version(mod)

    try:
        ff = ffmpeg_bin()
        encoders = _run([ff, "-hide_banner", "-encoders"])
        info["ffmpeg"] = _run([ff, "-hide_banner", "-version"]).splitlines()[0]
        info["ffmpeg codecs"] = ", ".join(
            c for c in ["libopencore_amrnb", "libgsm", "libopus", "pcm_mulaw", "pcm_alaw"] if c in encoders
        )
    except FileNotFoundError as e:
        info["ffmpeg"] = f"MISSING: {e}"

    info["git commit"] = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]) or "no commits"
    return info


def verify_data() -> bool:
    from pehleaap.manifest import verify_manifest

    manifests = sorted(MANIFEST_DIR.glob("*.jsonl"))
    if not manifests:
        print("no manifests in data/manifests/")
        return True
    all_ok = True
    for m in manifests:
        r = verify_manifest(m)
        ok = not r["missing"] and not r["changed"]
        all_ok &= ok
        print(f"{'OK  ' if ok else 'FAIL'} {m.name}: {r['ok']} ok, {len(r['missing'])} missing, {len(r['changed'])} changed")
        for p in (r["missing"] + r["changed"])[:10]:
            print(f"     {p}")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="print a table for docs/machines.md")
    ap.add_argument("--verify-data", action="store_true", help="check data/ against data/manifests/")
    args = ap.parse_args()

    if args.verify_data:
        sys.exit(0 if verify_data() else 1)

    info = collect()
    if args.markdown:
        print(f"\n## {info['host']}\n\n| Item | Value |\n|---|---|")
        for k, v in info.items():
            print(f"| {k} | {v} |")
    else:
        width = max(map(len, info))
        for k, v in info.items():
            print(f"{k:<{width}}  {v}")


if __name__ == "__main__":
    main()
