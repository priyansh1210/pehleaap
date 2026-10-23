"""Checksummed data manifests, so the Dev box and the GPU box can prove they hold the same data.

Audio stays out of git; `data/manifests/<name>.jsonl` (one line per file: path relative to
data/, size, sha256) is committed instead.
"""

import hashlib
import json
from pathlib import Path

from pehleaap.paths import DATA_DIR, MANIFEST_DIR


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(subdir: str, name: str | None = None) -> Path:
    """Hash every file under data/<subdir> into data/manifests/<name>.jsonl."""
    root = DATA_DIR / subdir
    out = MANIFEST_DIR / f"{name or subdir.replace('/', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(DATA_DIR).as_posix()
                f.write(json.dumps({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)}) + "\n")
    return out


def verify_manifest(manifest: Path) -> dict:
    """Return counts of ok / missing / changed files listed in a manifest."""
    result = {"ok": 0, "missing": [], "changed": []}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        p = DATA_DIR / entry["path"]
        if not p.exists():
            result["missing"].append(entry["path"])
        elif p.stat().st_size != entry["bytes"] or sha256_file(p) != entry["sha256"]:
            result["changed"].append(entry["path"])
        else:
            result["ok"] += 1
    return result
