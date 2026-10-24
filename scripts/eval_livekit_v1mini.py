"""LiveKit's local audio end-of-turn model (turn-detector v1-mini) on Smart Turn's test set, clean and
over the same simulated phone lines as scripts/eval_clip_baselines.py.

Runs from its own venv (.venv-livekit, see requirements-livekit.txt). The LiveKit Model License only
allows its models to be used with the LiveKit Agents framework and forbids using their outputs to
improve or develop other models, so this script drives the model only through livekit-agents' public
`inference.TurnDetector` stream API, the way AgentSession does, and its predictions are comparison
numbers only: never training data, labels or distillation targets.

How LiveKit uses the model: once the VAD has heard 200 ms of silence, it scores the last 1.2 s of
16 kHz audio and treats the turn as finished unless P(end of turn) is below a per-language
threshold (hi 0.305, en 0.36). Pipecat runs Smart Turn at the same 200 ms point, so each clip is fed
as it is, the same protocol as the other baselines. --pad-ms appends silence for a sensitivity check.

Clips, phone-line conditions, the English phone subset and the metrics all come from
eval_clip_baselines.py, so rows line up one to one with its predictions. Shards are saved as they
finish and a rerun skips them, like the main eval.

    .venv-livekit\\Scripts\\python scripts/eval_livekit_v1mini.py --limit-shards 1 --out livekit_smoke
    .venv-livekit\\Scripts\\python scripts/eval_livekit_v1mini.py --conditions clean g711_mulaw amrnb_7.95
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_clip_baselines import CHAIN, describe, in_phone_subset, phone_line  # noqa: E402
from livekit import rtc  # noqa: E402
from livekit.agents.inference import TurnDetector  # noqa: E402
from livekit.agents.language import LanguageCode  # noqa: E402

from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.eval.clip_metrics import binary_metrics  # noqa: E402
from pehleaap.paths import REPO_ROOT, RUNS_DIR  # noqa: E402
from pehleaap.telephony.codecs import CODECS  # noqa: E402
from pehleaap.telephony.noise import NoiseBank  # noqa: E402
from pehleaap.telephony.pipeline import load_config  # noqa: E402

MODEL = "livekit_v1_mini"
SR = smart_turn_testset.SAMPLE_RATE
FRAME = SR // 100  # 10 ms frames, as a room track delivers them
CONTEXT = int(1.5 * SR)  # the model sees the last 1.2 s; push a little more than that
LANG_CODES = {"hin": "hi", "eng": "en"}
ALL_CODECS = ["g711_mulaw", "g711_alaw", "amrnb_4.75", "amrnb_7.95", "amrnb_12.2", "gsm_fr", "opus_8k"]


async def _settle(stream) -> None:
    """Wait until the stream has moved every pushed frame into the model's audio buffer."""
    while not stream._audio_ch.empty():
        await asyncio.sleep(0)
    for _ in range(3):
        await asyncio.sleep(0)


async def _score_one(stream, x: np.ndarray, pad: np.ndarray) -> float:
    stream.flush()  # drop the previous clip's audio
    pcm = (np.clip(np.concatenate([x, pad])[-CONTEXT:], -1.0, 1.0) * 32767).astype(np.int16)
    for i in range(0, len(pcm), FRAME):
        chunk = pcm[i : i + FRAME]
        stream.push_audio(rtc.AudioFrame(chunk.tobytes(), SR, 1, len(chunk)))
    await _settle(stream)
    event = await stream.predict()
    return float(event.end_of_turn_probability)


async def _score_all(detector: TurnDetector, clips: list[np.ndarray], pad_ms: int, workers: int) -> np.ndarray:
    """P(end of turn) per clip, spread over `workers` independent streams."""
    pad = np.zeros(SR * pad_ms // 1000, np.float32)
    out = np.empty(len(clips))

    async def worker(k: int) -> None:
        stream = detector.stream()
        try:
            for i in range(k, len(clips), workers):
                out[i] = await _score_one(stream, clips[i], pad)
        finally:
            await stream.aclose()

    await asyncio.gather(*(worker(k) for k in range(workers)))
    return out


async def _thresholds(detector: TurnDetector) -> dict[str, float]:
    return {lang: await detector.unlikely_threshold(LanguageCode(code)) for lang, code in LANG_CODES.items()}


def predict_shard(detector, metas, clips, conditions, bank, cfg, pad_ms, workers) -> pd.DataFrame:
    records = []
    for cond in conditions:
        idx = list(range(len(metas))) if cond == "clean" else [i for i, m in enumerate(metas) if in_phone_subset(m)]
        audio = [clips[i] for i in idx]
        if cond != "clean":
            audio = phone_line(audio, [metas[i] for i in idx], cond, bank, cfg)
        probs = asyncio.run(_score_all(detector, audio, pad_ms, workers))
        for j, i in enumerate(idx):
            m = metas[i]
            records.append(
                {
                    "id": m["id"], "language": m["language"], "dataset": m["dataset"],
                    "synthetic": bool(m["synthetic"]), "label": int(m["endpoint_bool"]),
                    "condition": cond, "phone_subset": in_phone_subset(m), f"p_{MODEL}": float(probs[j]),
                }
            )
    return pd.DataFrame(records)


def metrics_rows(df: pd.DataFrame, thresholds: dict[str, float]) -> list[dict]:
    rows = []
    for (lang, cond), g in df.groupby(["language", "condition"]):
        m = binary_metrics(g["label"].to_numpy(), g[f"p_{MODEL}"].to_numpy(), thresholds[lang])
        rows.append({"language": lang, "condition": cond, "threshold": thresholds[lang], **m})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean", *ALL_CODECS])
    ap.add_argument("--langs", nargs="+", default=["eng", "hin"])
    ap.add_argument("--pad-ms", type=int, default=0, help="silence appended to every clip (sensitivity check)")
    ap.add_argument("--workers", type=int, default=8, help="parallel detector streams")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--out", default="livekit_v1mini")
    ap.add_argument("--data-root", default=None, help="folder holding data/*.parquet (default: download to data/external)")
    ap.add_argument("--fresh", action="store_true", help="recompute shards even if saved")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    unknown = [c for c in args.conditions if c != "clean" and c not in CODECS and c not in CHAIN]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}")
    cfg = load_config(REPO_ROOT / "configs" / "augment.yaml")
    bank = None
    if any(c in CHAIN for c in args.conditions):
        bank = NoiseBank(sr=SR)
        if not bank.files:
            raise SystemExit("the chain conditions use MUSAN noise: run scripts/get_noise.py first")

    detector = TurnDetector(version="v1-mini")
    if detector.model != "turn-detector-v1-mini":
        raise SystemExit(f"expected the local v1-mini model, got {detector.model}")
    thresholds = asyncio.run(_thresholds(detector))

    out_dir = RUNS_DIR / args.out
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_info = {"langs": args.langs, "pad_ms": args.pad_ms}
    data_root = Path(args.data_root) if args.data_root else smart_turn_testset.LOCAL_DIR
    if not args.data_root:
        smart_turn_testset.download()

    frames, t0 = [], time.time()
    shards = smart_turn_testset.shard_paths(data_root)[: args.limit_shards]
    for shard_i, shard in enumerate(shards):
        saved = shard_dir / f"{shard.stem}.csv"
        df = None
        if not args.fresh and saved.exists() and saved.with_suffix(".json").exists():
            info = json.loads(saved.with_suffix(".json").read_text())
            df = pd.read_csv(saved)
            if info != run_info or not set(args.conditions) <= set(df.condition):
                df = None
            else:
                df = df[df.condition.isin(args.conditions)]
        status = "saved earlier, skipped"
        if df is None:
            metas, clips = smart_turn_testset.read_shard(shard, args.langs)
            df = predict_shard(detector, metas, clips, args.conditions, bank, cfg, args.pad_ms, args.workers)
            saved.with_suffix(".json").write_text(json.dumps(run_info))
            tmp = saved.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(saved)
            status = "done"
        frames.append(df)
        print(f"shard {shard_i + 1}/{len(shards)} {status}: {len(df)} rows, {time.time() - t0:.0f}s elapsed", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "predictions.csv", index=False)

    validation = metrics_rows(df[df.condition == "clean"], thresholds)
    phone = metrics_rows(df[df.phone_subset], thresholds)
    clean_acc = {r["language"]: r["accuracy"] for r in phone if r["condition"] == "clean"}
    order = {c: i for i, c in enumerate(args.conditions)}
    lines = [
        "# LiveKit turn-detector v1-mini: clean vs phone line",
        "",
        "Data: pipecat-ai/smart-turn-data-v3.2-test (English + Hindi; **all Hindi clips are synthetic TTS**).",
        "This is Smart Turn's own test set: in-distribution for Smart Turn and the community models, not for LiveKit.",
        f"Scored at LiveKit's own per-language thresholds ({', '.join(f'{k} {v}' for k, v in thresholds.items())}): "
        "finished when P(end of turn) is above the threshold. Cut-in = finished called on an unfinished turn.",
        f"Each clip's last 1.2 s, as is{f', plus {args.pad_ms} ms of silence' if args.pad_ms else ''}.",
        "",
        "## 1. Clean, all clips",
        "",
        "| Lang | n | Acc | Cut-in rate | Missed-end rate | AUC |",
        "|---|---|---|---|---|---|",
        *[f"| {r['language']} | {r['n']} | {r['accuracy']:.2f} | {r['fpr']:.3f} | {r['fnr']:.3f} | {r['auc']:.3f} |" for r in validation],
    ]
    if len(args.conditions) > 1:
        lines += [
            "",
            "## 2. Phone-line stress test (all Hindi + 25% hash sample of English)",
            "",
            *[f"- `{c}`: {describe(c)}" for c in args.conditions if c != "clean"],
            "",
            "| Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(phone, key=lambda r: (r["language"], order[r["condition"]])):
            lines.append(
                f"| {r['language']} | {r['condition']} | {r['n']} | {r['accuracy']:.2f} | {r['accuracy'] - clean_acc[r['language']]:+.2f} | "
                f"{r['fpr']:.3f} | {r['fnr']:.3f} | {r['auc']:.3f} |"
            )

    results = {"model": MODEL, "thresholds": thresholds, **run_info, "validation": validation, "phone": phone}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
