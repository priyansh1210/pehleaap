"""TurnBench dev (mundo-ai/turn-benchmark-dev) as session folders for label_session.py and eval_sessions.py.

Each conversation becomes TB<conversation_id>/ with its two full-quality FLAC channels as they ship
(speaker_1 -> agent, speaker_2 -> customer; the lossy Opus previews are never used, as the dataset
card asks) and turnbench.json: the conversation's metadata and the three human annotator tracks per
speaker, kept for checking our automatic labels against theirs.

Evaluation only. The licence (Dataset Public License v1.0) allows NonCommercial use, bans voice
cloning and requires attribution: never train on it, never put its audio or transcripts in git.

    python scripts/turnbench_to_sessions.py
    python scripts/label_session.py "data/external/turnbench-dev/sessions/TB*" --out runs/turnbench_labels
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.paths import EXTERNAL_DIR  # noqa: E402

ROLES = {"speaker_1": "agent", "speaker_2": "customer"}
COLUMNS = ["conversation_id", "metadata"] + [f"{s}_audio" for s in ROLES] + [f"{s}_annotation_{a}" for s in ROLES for a in "abc"]


def write_session(row: dict, out_root: Path) -> Path:
    """One TurnBench row -> out_root/TB<id>/ with TB<id>_<speaker>_<role>.flac per channel and turnbench.json."""
    session = f"TB{row['conversation_id']}"
    out = out_root / session
    out.mkdir(parents=True, exist_ok=True)
    for spk, role in ROLES.items():
        # label_session.py reads the speaker from the file name: the actor id (pseudonymous, alphanumeric), cut short.
        actor = row["metadata"][f"{spk}_actor_id"][:8]
        (out / f"{session}_{actor}_{role}.flac").write_bytes(row[f"{spk}_audio"]["bytes"])
    labels = {"conversation_id": row["conversation_id"], "roles": ROLES, "metadata": row["metadata"],
              "annotations": {spk: {a: row[f"{spk}_annotation_{a}"] for a in "abc"} for spk in ROLES}}
    (out / "turnbench.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(EXTERNAL_DIR / "turnbench-dev" / "data"))
    ap.add_argument("--out", default=str(EXTERNAL_DIR / "turnbench-dev" / "sessions"))
    args = ap.parse_args()
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(str(Path(args.data) / "dev-*.parquet")))
    if not shards:
        raise SystemExit(f"no dev-*.parquet in {args.data}: hf download mundo-ai/turn-benchmark-dev --repo-type dataset")
    n = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, columns=COLUMNS):  # one ~100 MB row at a time
            for row in batch.to_pylist():
                print(write_session(row, Path(args.out)).name, flush=True)
                n += 1
    print(f"{n} conversations -> {args.out}")


if __name__ == "__main__":
    main()
