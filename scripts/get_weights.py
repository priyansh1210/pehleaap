"""Download our end-of-turn models from the GitHub release into weights/<name>/, checked against the sha256s kept here.

    python scripts/get_weights.py                  # eot_v1, the shipped model -> weights/eot_v1/model.onnx
    python scripts/get_weights.py eot_v1 eot_v0    # also the first fine-tune, for comparison

A file already on disk with the right checksum isn't downloaded again. The models are Smart-Turn-format ONNX
(Whisper log-mel of the last 8 s in, P(turn complete) out, threshold 0.5). Load one with
`pehleaap.baselines.turn_models.load_any("eot_v1")`, or run it in the demo (demo/bot.py picks it up).
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.manifest import sha256_file  # noqa: E402
from pehleaap.paths import WEIGHTS_DIR  # noqa: E402

# A release that holds only model files and never changes, so code releases can come and go without moving it.
RELEASE = "https://github.com/priyansh1210/pehleaap/releases/download/models-v1"
# model -> [(release asset, file under weights/<model>/, sha256)]
MODELS = {
    "eot_v1": [("eot_v1.onnx", "model.onnx", "0ccac636fc008c04d19d3aa4c1e7b09db874b67bb1ff1f8cab87a306ba89a52b")],
    "eot_v0": [
        ("eot_v0.onnx", "model.onnx", "38fbe51a16d3c4506290d7068a752f164d6ad0f2046a03545c3d2cc972a40202"),
        ("eot_v0_int8.onnx", "model_int8.onnx", "d694d40113e1b8b8b6b3324aac683f4e8ef4c483ed6c019803ef05ba13567a1a"),
    ],
}


def fetch(url: str, out: Path, sha256: str, retries: int = 5) -> str:
    """'present' if `out` already has the right checksum; otherwise download it (through a .part file) and check it."""
    if out.exists() and sha256_file(out) == sha256:
        return "present"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, part)
            break
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            raise SystemExit(f"{url}: HTTP {e.code}" + (" (is the release published?)" if e.code == 404 else ""))
        except OSError as e:
            if attempt == retries - 1:
                part.unlink(missing_ok=True)
                raise SystemExit(f"couldn't download {url}: {e}")
            print(f"network error ({e}); retrying", flush=True)
            time.sleep(2**attempt)
    got = sha256_file(part)
    if got != sha256:
        part.unlink()
        raise SystemExit(f"{url}: checksum {got[:12]}… isn't the released {sha256[:12]}…; nothing was saved")
    part.replace(out)
    return "downloaded"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=["eot_v1"], help=f"any of {', '.join(MODELS)}")
    ap.add_argument("--base-url", default=RELEASE, help="where the release assets are")
    ap.add_argument("--dest", default=str(WEIGHTS_DIR))
    args = ap.parse_args()

    if unknown := [m for m in args.models if m not in MODELS]:
        raise SystemExit(f"unknown models {unknown}; released: {', '.join(MODELS)}")
    for name in args.models:
        for asset, filename, sha256 in MODELS[name]:
            out = Path(args.dest) / name / filename
            status = fetch(f"{args.base_url.rstrip('/')}/{asset}", out, sha256)
            print(f"{name}: {out} {status}, {out.stat().st_size / 1e6:.1f} MB, sha256 OK")


if __name__ == "__main__":
    main()
