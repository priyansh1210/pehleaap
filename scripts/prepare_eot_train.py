"""Download Smart Turn's training set and keep its English and Hindi clips as a clip store for train_eot.py.

pipecat-ai/smart-turn-data-v3.2-train (CC BY 4.0) is 83 parquet shards (41 GB) in 23 languages. English
and Hindi are about 29% of the clips, roughly 67k + 13k (every Hindi clip is TTS). Each shard is checked
against the Hub's sha256, read, and its clips' last 8 s appended to data/external/eot-train/
(audio.int16 + index.csv, about 16 GB in all). A rerun skips shards already stored, so an interrupted
run resumes. The next shard downloads while the current one is being stored.

    python scripts/prepare_eot_train.py                    # all shards
    python scripts/prepare_eot_train.py --max-shards 1     # smoke test (about 1k clips)
    python scripts/prepare_eot_train.py --delete-shards    # delete each parquet once stored (saves 41 GB of disk)
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.data.smart_turn_testset import read_shard  # noqa: E402
from pehleaap.manifest import sha256_file  # noqa: E402
from pehleaap.paths import EXTERNAL_DIR  # noqa: E402
from pehleaap.train.eot_data import EotStore  # noqa: E402

DATASET = "pipecat-ai/smart-turn-data-v3.2-train"
SHARD_DIR = EXTERNAL_DIR / "smart-turn-data-v3.2-train"
STORE_DIR = EXTERNAL_DIR / "eot-train"


def fetch(entry) -> Path:
    """Download one shard (unless an identical copy is on disk) and check its sha256."""
    from huggingface_hub import hf_hub_download

    path = SHARD_DIR / entry.path
    if not (path.exists() and path.stat().st_size == entry.size and sha256_file(path) == entry.lfs.sha256):
        hf_hub_download(DATASET, entry.path, repo_type="dataset", local_dir=SHARD_DIR)
        if sha256_file(path) != entry.lfs.sha256:
            raise SystemExit(f"{entry.path}: sha256 doesn't match the Hub's")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["eng", "hin"])
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--delete-shards", action="store_true")
    ap.add_argument("--out", default=str(STORE_DIR))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    entries = sorted(HfApi().list_repo_tree(DATASET, repo_type="dataset", path_in_repo="data", expand=True), key=lambda e: e.path)
    entries = [e for e in entries if e.path.endswith(".parquet")][: args.max_shards]
    store = EotStore(args.out)
    todo = [e for e in entries if Path(e.path).name not in store.done_shards()]
    print(f"{len(entries) - len(todo)} of {len(entries)} shards already stored; {len(todo)} to go", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(1) as pool:
        next_path = pool.submit(fetch, todo[0]) if todo else None
        for k, entry in enumerate(todo):
            path = next_path.result()
            next_path = pool.submit(fetch, todo[k + 1]) if k + 1 < len(todo) else None
            metas, clips = read_shard(path, args.langs)
            store.append(metas, clips, path.name)
            if args.delete_shards:
                path.unlink()
            done = k + 1
            eta = (time.time() - t0) / done * (len(todo) - done) / 60
            print(f"{path.name}: {len(clips)} clips; store has {len(store.index)}; {done}/{len(todo)}, ~{eta:.0f} min left", flush=True)

    counts = store.index.groupby(["language", "split"]).size().to_dict()
    print(f"done: {len(store.index)} clips in {args.out}; by language/split: {counts}")


if __name__ == "__main__":
    main()
