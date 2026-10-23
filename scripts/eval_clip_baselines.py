"""End-of-turn baselines on Smart Turn's test set, on clean audio and over a simulated phone line.

Two jobs:
1. Validate each wrapper: on clean audio, accuracy must match its model card (within 1 pt).
2. Telephony stress test: the same clips band-limited to 8 kHz and passed through G.711 and
   AMR-NB. This is a first look at the phone-line domain shift. It is not the project's main
   evaluation: the community models were trained on this corpus (so it's in-distribution for
   them), and every Hindi clip here is TTS.

Heavy (~0.5 s per clip for all five models on the laptop's 4 cores, ~2.5 h in total), so the full
run belongs on the GPU box. Each shard's predictions are saved as soon as it finishes, and a rerun
skips shards already done, so an interrupted run resumes where it stopped.

    python scripts/eval_clip_baselines.py --threads 8                      # everything
    python scripts/eval_clip_baselines.py --models chhotu_int8 --limit-shards 1 --out smoke
    python scripts/eval_clip_baselines.py --report-only                    # rebuild report from saved shards
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.baselines.turn_models import REGISTRY, load  # noqa: E402
from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.eval.clip_metrics import binary_metrics  # noqa: E402
from pehleaap.paths import RUNS_DIR  # noqa: E402
from pehleaap.telephony.codecs import narrowband_many  # noqa: E402

# Clean accuracy (%) at each card's own threshold on smart-turn-v3.2-test.
CARD_ACCURACY = {
    "smart_turn_v3.2_int8": {"eng": 94.26, "hin": 90.11},
    "deveshu_fp32": {"eng": 93.8, "hin": 93.2},
    "codewithmoin_8s": {"eng": 93.7, "hin": 93.9},
    "chhotu_int8": {"eng": 88.4, "hin": 88.8},
    # abhishek_tiny_attn's card reports only balanced accuracy on 500-clip subsets (eng 67.4, hin 72.4).
}

CODEC_CHUNK = 200  # clips per ffmpeg run


def in_phone_subset(meta: dict) -> bool:
    """All Hindi clips, plus a fixed ~25% hash sample of English, for the codec conditions."""
    if meta["language"] == "hin":
        return True
    return int(hashlib.md5(meta["id"].encode()).hexdigest(), 16) % 4 == 0


def phone_line(clips: list[np.ndarray], codec: str) -> list[np.ndarray]:
    out = []
    for i in range(0, len(clips), CODEC_CHUNK):
        out += narrowband_many(clips[i : i + CODEC_CHUNK], codec)
    return out


def predict_shard(models: dict, metas: list[dict], clips: list[np.ndarray], conditions: list[str]) -> pd.DataFrame:
    records = []
    for cond in conditions:
        idx = list(range(len(metas))) if cond == "clean" else [i for i, m in enumerate(metas) if in_phone_subset(m)]
        audio = [clips[i] for i in idx]
        if cond != "clean":
            audio = phone_line(audio, cond)
        probs = {name: model.predict_batch(audio) for name, model in models.items()}
        for j, i in enumerate(idx):
            m = metas[i]
            records.append(
                {
                    "id": m["id"], "language": m["language"], "dataset": m["dataset"],
                    "synthetic": bool(m["synthetic"]), "label": int(m["endpoint_bool"]),
                    "condition": cond, "phone_subset": in_phone_subset(m),
                    **{f"p_{name}": float(p[j]) for name, p in probs.items()},
                }
            )
    return pd.DataFrame(records)


def load_saved_shard(path: Path, names: list[str], conditions: list[str], langs: list[str]) -> pd.DataFrame | None:
    """A shard saved by an earlier run, if it covers every requested model, condition and language."""
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    df = pd.read_csv(path)
    if not {f"p_{n}" for n in names} <= set(df.columns) or not set(conditions) <= set(df.condition):
        return None
    if not set(langs) <= set(json.loads(path.with_suffix(".json").read_text())["langs"]):
        return None
    return df[df.condition.isin(conditions) & df.language.isin(langs)]


def metrics_table(df: pd.DataFrame, models: list[str], keys: list[str]) -> list[dict]:
    rows = []
    for group, g in df.groupby(keys):
        group = group if isinstance(group, tuple) else (group,)
        for name in models:
            m = binary_metrics(g["label"].to_numpy(), g[f"p_{name}"].to_numpy(), REGISTRY[name].threshold)
            rows.append({**dict(zip(keys, group)), "model": name, **m})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REGISTRY))
    ap.add_argument("--conditions", nargs="+", default=["clean", "g711_mulaw", "amrnb_7.95"])
    ap.add_argument("--langs", nargs="+", default=["eng", "hin"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--out", default="clip_baselines")
    ap.add_argument("--data-root", default=None, help="folder holding data/*.parquet (default: download to data/external)")
    ap.add_argument("--fresh", action="store_true", help="recompute shards even if saved")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from saved shards")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the report has non-ASCII; Windows pipes and logs default to cp1252

    out_dir = RUNS_DIR / args.out
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    names = list(args.models)
    data_root = Path(args.data_root) if args.data_root else smart_turn_testset.LOCAL_DIR
    if not args.report_only and not args.data_root:
        smart_turn_testset.download()

    models, frames, t0 = None, [], time.time()
    shards = smart_turn_testset.shard_paths(data_root)[: args.limit_shards]
    if not shards:
        raise SystemExit(f"no parquet shards under {data_root / 'data'}")
    for shard_i, shard in enumerate(shards):
        saved = shard_dir / f"{shard.stem}.csv"
        df = None if args.fresh else load_saved_shard(saved, names, args.conditions, args.langs)
        if df is None and args.report_only:
            raise SystemExit(f"{saved} is missing or lacks the requested models/conditions; run without --report-only")
        if df is None:
            if models is None:
                models = {name: load(name, threads=args.threads) for name in names}
            df = predict_shard(models, *smart_turn_testset.read_shard(shard, args.langs), args.conditions)
            saved.with_suffix(".json").write_text(json.dumps({"langs": args.langs}))
            tmp = saved.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(saved)  # never leave a half-written shard behind if the run is killed
            status = "done"
        else:
            status = "saved earlier, skipped"
        frames.append(df)
        print(f"shard {shard_i + 1}/{len(shards)} {status}: {len(df)} rows, {time.time() - t0:.0f}s elapsed", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "predictions.csv", index=False)

    lines = ["# End-of-turn baselines: clean vs phone line", ""]
    lines += [
        "Data: pipecat-ai/smart-turn-data-v3.2-test (English + Hindi; **all Hindi clips are synthetic TTS**).",
        "Community models were trained on smart-turn-data-v3.2-train, so this test is in-distribution for them.",
        "Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.",
        "",
        "## 1. Wrapper validation (clean; card numbers are for the full set, so only a full run can match)",
        "",
        "| Model | Lang | n | Acc | Card | Δ | Match |",
        "|---|---|---|---|---|---|---|",
    ]
    clean = df[df.condition == "clean"]
    validation = metrics_table(clean, names, ["language"])
    for r in validation:
        card = CARD_ACCURACY.get(r["model"], {}).get(r["language"])
        delta = f"{r['accuracy'] - card:+.2f}" if card is not None else "n/a"
        match = "n/a" if card is None else ("yes" if abs(r["accuracy"] - card) <= 1.0 else "**NO**")
        lines.append(f"| {r['model']} | {r['language']} | {r['n']} | {r['accuracy']:.2f} | {card if card is not None else '-'} | {delta} | {match} |")

    phone = df[df.phone_subset]
    if len(args.conditions) > 1:
        lines += [
            "",
            "## 2. Phone-line stress test (all Hindi + 25% hash sample of English)",
            "",
            "Narrowband 8 kHz + codec only (no noise, echo or packet loss yet).",
            "",
            "| Model | Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        rows = metrics_table(phone, names, ["language", "condition"])
        clean_acc = {(r["model"], r["language"]): r["accuracy"] for r in rows if r["condition"] == "clean"}
        order = {c: i for i, c in enumerate(args.conditions)}
        for r in sorted(rows, key=lambda r: (r["model"], r["language"], order[r["condition"]])):
            d = r["accuracy"] - clean_acc[(r["model"], r["language"])]
            lines.append(
                f"| {r['model']} | {r['language']} | {r['condition']} | {r['n']} | {r['accuracy']:.2f} | "
                f"{d:+.2f} | {r['fpr']:.3f} | {r['fnr']:.3f} | {r['auc']:.3f} |"
            )
    else:
        rows = []

    (out_dir / "results.json").write_text(json.dumps({"validation": validation, "phone": rows}, indent=2))
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
