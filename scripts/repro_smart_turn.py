"""Week-1 harness check: reproduce Smart Turn v3.2's published accuracy on its own test set.

Published numbers come from huggingface.co/pipecat-ai/smart-turn-v3 benchmarks/smart-turn-v3.2-{cpu,gpu}.md.
Pass criterion: accuracy within 1 point for every language evaluated. Until this passes,
no number this harness produces on our own data can be trusted.

    python scripts/repro_smart_turn.py                 # English + Hindi, int8 CPU model
    python scripts/repro_smart_turn.py --limit 200     # quick smoke run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.baselines.smart_turn import SmartTurnONNX, download_weights  # noqa: E402
from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.eval.clip_metrics import binary_metrics, grouped_metrics  # noqa: E402
from pehleaap.paths import RUNS_DIR  # noqa: E402

# benchmarks/smart-turn-v3.2-cpu.md (int8 model). fp/fn are % of all samples, as Smart Turn defines them.
PUBLISHED = {
    "cpu": {
        "eng": {"n": 7820, "accuracy": 94.26, "fp_pct_total": 3.75, "fn_pct_total": 1.99},
        "hin": {"n": 1284, "accuracy": 90.11, "fp_pct_total": 8.57, "fn_pct_total": 1.32},
    },
}


def iter_clips(langs: list[str], limit: int | None):
    """Yield (metadata dict, 16 kHz audio) for the requested languages."""
    count = 0
    for metas, clips in smart_turn_testset.iter_shards(langs):
        for m, x in zip(metas, clips):
            yield m, x
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["eng", "hin"])
    ap.add_argument("--variant", default="cpu", choices=["cpu", "gpu"], help="cpu = int8, gpu = fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = RUNS_DIR / f"repro_smart_turn_v3.2_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SmartTurnONNX(download_weights("3.2", args.variant), threads=args.threads)
    smart_turn_testset.download()

    metas, probs = [], []
    t0 = time.time()
    batch_meta, batch_audio = [], []
    for m, x in iter_clips(args.langs, args.limit):
        batch_meta.append(m)
        batch_audio.append(x)
        if len(batch_audio) == 32:
            probs.append(model.predict_batch(batch_audio))
            metas += batch_meta
            batch_meta, batch_audio = [], []
            if len(metas) % 1024 == 0:
                print(f"{len(metas)} clips, {time.time() - t0:.0f}s", flush=True)
    if batch_audio:
        probs.append(model.predict_batch(batch_audio))
        metas += batch_meta
    probs = np.concatenate(probs)
    labels = np.array([m["endpoint_bool"] for m in metas], dtype=int)

    results = {
        "model": f"smart-turn-v3.2-{args.variant}.onnx",
        "n": len(metas),
        "seconds": round(time.time() - t0, 1),
        "overall": binary_metrics(labels, probs),
        "by_language": grouped_metrics(labels, probs, [m["language"] for m in metas]),
        "by_language_synthetic": grouped_metrics(
            labels, probs, [f"{m['language']}/{'synthetic' if m['synthetic'] else 'human'}" for m in metas]
        ),
        "by_language_dataset": grouped_metrics(labels, probs, [f"{m['language']}/{m['dataset']}" for m in metas]),
    }

    # Compare with the published benchmark.
    published = PUBLISHED.get(args.variant, {})
    lines = [f"# Smart Turn v3.2 ({args.variant}) reproduction", "", "| Lang | n (ours/pub) | Acc ours | Acc pub | Δ | FP%tot ours/pub | FN%tot ours/pub |", "|---|---|---|---|---|---|---|"]
    passed = True
    for lang, m in results["by_language"].items():
        pub = published.get(lang)
        if not pub:
            continue
        delta = m["accuracy"] - pub["accuracy"]
        passed = passed and abs(delta) <= 1.0 and m["n"] == pub["n"]
        lines.append(
            f"| {lang} | {m['n']}/{pub['n']} | {m['accuracy']:.2f} | {pub['accuracy']:.2f} | {delta:+.2f} | "
            f"{m['fp_pct_total']:.2f}/{pub['fp_pct_total']:.2f} | {m['fn_pct_total']:.2f}/{pub['fn_pct_total']:.2f} |"
        )
    lines += ["", f"**{'PASS' if passed else 'FAIL'}** (accuracy within 1 pt and full sample count)", ""]
    lines += ["## Human vs synthetic audio", "", "| Group | n | Acc | FPR (cut-in) | FNR |", "|---|---|---|---|---|"]
    for g, m in results["by_language_synthetic"].items():
        lines.append(f"| {g} | {m['n']} | {m['accuracy']:.2f} | {m['fpr']:.3f} | {m['fnr']:.3f} |")
    lines += ["", "## By source dataset", "", "| Group | n | Acc | FPR (cut-in) | FNR |", "|---|---|---|---|---|"]
    for g, m in results["by_language_dataset"].items():
        lines.append(f"| {g} | {m['n']} | {m['accuracy']:.2f} | {m['fpr']:.3f} | {m['fnr']:.3f} |")

    results["passed"] = bool(passed)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    np.savez(out_dir / "predictions.npz", ids=[m["id"] for m in metas], probs=probs, labels=labels)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
