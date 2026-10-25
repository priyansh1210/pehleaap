"""A real-speech Hindi end-of-turn test from Google's FLEURS (CC BY 4.0): human readers, not TTS.

Every Hindi clip in Smart Turn's test set is synthetic TTS. FLEURS has native speakers reading Hindi
sentences; its test and dev splits are used here (train stays free for training). Each recording gives:
- a "finished" clip: the audio up to the end of the sentence's speech, plus `--tail-s` of the silence after;
- at most one "not finished" clip: the audio up to a pause the reader made mid-sentence (VAD silence of at
  least `--min-pause-s`, after which they carried on), plus `--tail-s` of that pause.
Both end the way a live bot hears them: 0.2 s into the silence (the demo's VAD stop_secs). Pauses come
from the labeling engine's VAD and IPU rules (configs/labeling.yaml). This is read speech, not
conversation: readers pause at commas and phrase ends rather than before a verb, so it tests real voices
and recording conditions, not conversational turn-taking.

Writes data/external/fleurs-hi-eot/data/<split>.parquet in Smart Turn's test-set schema, so
eval_clip_baselines.py scores it directly:

    python scripts/make_fleurs_eot.py
    python scripts/eval_clip_baselines.py --data-root data/external/fleurs-hi-eot --langs hin \\
        --models smart_turn_v3.2_int8 --onnx eot_v0=weights/eot_v0/model.onnx --conditions clean g711_mulaw amrnb_7.95 --out fleurs_eot

For training (eot_v1), the train split goes into the clip store instead, with up to two pauses per
recording; the test and dev splits are refused there, so the real-voice test stays unseen:

    python scripts/make_fleurs_eot.py --splits train --to-store data/external/eot-train --max-pauses 2
"""

import argparse
import csv
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.vad import SileroVAD, hop_s  # noqa: E402
from pehleaap.labeling.config import LabelConfig  # noqa: E402
from pehleaap.labeling.ipu import ipus_from_probs  # noqa: E402
from pehleaap.paths import EXTERNAL_DIR  # noqa: E402

DATASET = "google/fleurs"
SRC = EXTERNAL_DIR / "fleurs"
OUT = EXTERNAL_DIR / "fleurs-hi-eot"
SR = 16000


def fetch(split: str) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download(DATASET, f"data/hi_in/{split}.tsv", repo_type="dataset", local_dir=SRC)
    tar = hf_hub_download(DATASET, f"data/hi_in/audio/{split}.tar.gz", repo_type="dataset", local_dir=SRC)
    return Path(tsv), Path(tar)


def clips_for(x: np.ndarray, name: str, vad: SileroVAD, cfg: LabelConfig, min_pause_s: float, tail_s: float, min_before_s: float,
              max_pauses: int = 1):
    """[(endpoint_bool, clip, kind)] for one recording: its finished clip and up to `max_pauses` mid-sentence pauses."""
    ipus = ipus_from_probs(vad.probs(x, SR), hop_s(SR), cfg)
    if not ipus:
        return []
    out = [(True, x[: int(min(len(x) / SR, ipus[-1].end + tail_s) * SR)], "end")]
    pauses = [(a, b) for a, b in zip(ipus[:-1], ipus[1:]) if b.start - a.end >= min_pause_s and a.end >= min_before_s]
    start = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(pauses) if pauses else 0
    for a, _ in (pauses[start:] + pauses[:start])[:max_pauses]:  # a fixed, hash-chosen pick
        out.append((False, x[: int((a.end + tail_s) * SR)], f"pause@{a.end:.2f}"))
    return out


def flac(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--min-pause-s", type=float, default=0.25, help="a mid-sentence pause must be at least this long")
    ap.add_argument("--tail-s", type=float, default=0.2, help="silence kept after the speech: what a live bot has heard")
    ap.add_argument("--min-before-s", type=float, default=1.0, help="ignore pauses in the first second")
    ap.add_argument("--max-pauses", type=int, default=1, help="mid-sentence pauses taken per recording")
    ap.add_argument("--to-store", help="append to this training clip store (train split only) instead of writing a test set")
    args = ap.parse_args()
    if args.to_store and set(args.splits) & {"test", "dev"}:
        raise SystemExit("the test and dev splits are the real-voice test set: never put them in a training store")

    vad, cfg = SileroVAD(), LabelConfig()
    store = None
    if args.to_store:
        from pehleaap.train.eot_data import EotStore

        store = EotStore(args.to_store)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if store is not None and f"fleurs_hi_{split}" in store.done_shards():
            print(f"{split}: already in {args.to_store}")
            continue
        tsv, tar = fetch(split)
        with open(tsv, encoding="utf-8", newline="") as f:
            gender = {row[1]: (row[6] if len(row) > 6 else "") for row in csv.reader(f, delimiter="\t")}
        rows = []
        with tarfile.open(tar, "r:gz") as t:
            for m in t:
                name = Path(m.name).name
                if not m.isfile() or name not in gender:
                    continue
                x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
                if sr != SR:
                    raise SystemExit(f"{name}: {sr} Hz, expected {SR}")
                for finished, clip, kind in clips_for(x.mean(axis=1), name, vad, cfg, args.min_pause_s, args.tail_s, args.min_before_s,
                                                      args.max_pauses):
                    rows.append({"id": f"fleurs_{split}_{Path(name).stem}_{kind}", "language": "hin", "endpoint_bool": finished,
                                 "synthetic": False, "dataset": f"fleurs_hi_{split}", "midfiller": False, "endfiller": False,
                                 "audio": {"bytes": flac(clip), "path": name}})
        n_end = sum(r["endpoint_bool"] for r in rows)
        print(f"{split}: {len(gender)} recordings -> {n_end} finished + {len(rows) - n_end} mid-sentence-pause clips", flush=True)
        if store is not None:
            clips = [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0] for r in rows]
            store.append([{k: r[k] for k in ("id", "language", "endpoint_bool", "synthetic", "dataset")} for r in rows], clips, f"fleurs_hi_{split}")
            print(f"  added to {args.to_store}: it now has {len(store.index)} clips", flush=True)
        else:
            pq.write_table(pa.Table.from_pylist(rows), OUT / "data" / f"{split}.parquet")
    if store is not None:
        return
    (OUT / "README.md").write_text(
        "# FLEURS Hindi read speech by human speakers (google/fleurs, CC BY 4.0), cut at the sentence end (finished) "
        "or 0.2 s into a mid-sentence pause (not finished)\n\nBuilt by scripts/make_fleurs_eot.py; see its docstring.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
