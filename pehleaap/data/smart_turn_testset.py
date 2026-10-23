"""pipecat-ai/smart-turn-data-v3.2-test (CC-BY-4.0): 31,527 clips in 10 parquet shards, about 4.8 GB.

Clips are labelled `endpoint_bool` (turn complete or not), with `language`, `dataset` and a
`synthetic` flag. All Hindi clips are synthetic TTS.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from pehleaap.audio.io import resample
from pehleaap.paths import EXTERNAL_DIR

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
LOCAL_DIR = EXTERNAL_DIR / "smart-turn-data-v3.2-test"
META_COLUMNS = ["id", "language", "endpoint_bool", "synthetic", "dataset", "midfiller", "endfiller"]
SAMPLE_RATE = 16000


def download() -> Path:
    """Fetch the test shards, keeping any already on disk whose sha256 matches the Hub's."""
    from huggingface_hub import HfApi, hf_hub_download

    from pehleaap.manifest import sha256_file

    for entry in HfApi().list_repo_tree(DATASET, repo_type="dataset", path_in_repo="data", expand=True):
        path = LOCAL_DIR / entry.path
        if path.exists() and path.stat().st_size == entry.size and sha256_file(path) == entry.lfs.sha256:
            continue
        print(f"downloading {entry.path}", flush=True)
        hf_hub_download(DATASET, entry.path, repo_type="dataset", local_dir=LOCAL_DIR)
    return LOCAL_DIR


def shard_paths(root: Path = LOCAL_DIR) -> list[Path]:
    return sorted((root / "data").glob("*.parquet"))


def read_shard(shard: Path, langs: list[str]) -> tuple[list[dict], list[np.ndarray]]:
    """(metadata, 16 kHz mono clips) for one shard, filtered to `langs`."""
    table = pq.read_table(shard, columns=META_COLUMNS + ["audio"], filters=[("language", "in", langs)])
    metas = table.select(META_COLUMNS).to_pylist()
    clips = []
    for a in table.column("audio").to_pylist():
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        clips.append(resample(x, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else x)
    return metas, clips


def iter_shards(langs: list[str], root: Path = LOCAL_DIR) -> Iterator[tuple[list[dict], list[np.ndarray]]]:
    """Yield (metadata, 16 kHz mono clips) one shard at a time, to keep memory bounded."""
    for shard in shard_paths(root):
        yield read_shard(shard, langs)
