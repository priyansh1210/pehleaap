"""Int8 copy of a fine-tuned EOT model, made the way Pipecat made smart-turn-v3.2-cpu.onnx.

1. The best checkpoint's weights go into Pipecat's own fp32 graph (smart-turn-v3.2-gpu.onnx), so the
   model ships in exactly the graph Pipecat publishes: <run>/model_pipecat_graph.onnx, checked against
   the run's own model.onnx on the dev clips.
2. Upstream's recipe (pipecat-ai/smart-turn train.py, quantize_onnx_model; pehleaap/models/quantize.py):
   QDQ, entropy calibration on 1024 clean training clips (never dev or test), Conv/MatMul/Gemm only.
   Quantising every op type, or calibrating on phone-line audio, shifted probabilities by -0.1 to -0.2
   and cost up to 23 dev points; this recipe reproduces Pipecat's published int8 file from its fp32 one.
3. Checks int8 against fp32 on the dev split (clean and phone line, the keys train_eot.py scores) and
   times both, and Smart Turn's int8, on the CPU.

Writes <run>/model_int8.onnx and <run>/quantize.json. The test-set check is a separate
eval_clip_baselines.py run with --onnx (docs/machines.md).

    python scripts/quantize_eot.py --run eot_v0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.eval.clip_metrics import binary_metrics  # noqa: E402
from pehleaap.models.quantize import quantize_eot  # noqa: E402
from pehleaap.models.smart_turn import PUBLISHED, onnx_with_state_dict  # noqa: E402
from pehleaap.baselines.turn_models import fetch  # noqa: E402
from pehleaap.paths import REPO_ROOT, RUNS_DIR  # noqa: E402
from pehleaap.train.eot_data import DEV_CLEAN, DEV_PHONE, Augmenter, EotStore  # noqa: E402

BATCH = 64


def features_for(augmenter: Augmenter, keys: list[tuple[int, int]]) -> dict:
    """Augmenter output for `keys`, in batches, concatenated."""
    parts = [augmenter(keys[i : i + BATCH]) for i in range(0, len(keys), BATCH)]
    return {
        "features": np.concatenate([p["features"].numpy() for p in parts]),
        "labels": np.concatenate([p["labels"].numpy() for p in parts]).astype(int),
        "language": np.array(sum((p["language"] for p in parts), [])),
        "phone": np.concatenate([p["phone"].numpy() for p in parts]),
    }


def session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def probs(path: Path, features: np.ndarray, threads: int = 8) -> np.ndarray:
    s = session(path, threads)
    return np.concatenate([s.run(None, {"input_features": features[i : i + BATCH]})[0].reshape(-1) for i in range(0, len(features), BATCH)])


def latency_ms(path: Path, threads: int, runs: int = 100) -> float:
    """Median time of one clip (batch 1), as a live agent calls the model."""
    s = session(path, threads)
    x = np.random.default_rng(0).standard_normal((1, 80, 800)).astype(np.float32)
    for _ in range(10):
        s.run(None, {"input_features": x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        s.run(None, {"input_features": x})
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eot_v0", help="train_eot.py output folder under runs/")
    ap.add_argument("--calib-clips", type=int, default=1024)
    ap.add_argument("--method", default="entropy", choices=["entropy", "minmax", "percentile"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run = RUNS_DIR / args.run
    exported, fp32, int8 = run / "model.onnx", run / "model_pipecat_graph.onnx", run / "model_int8.onnx"
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    onnx_with_state_dict(fetch(*PUBLISHED), best["model"], fp32)
    cfg = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
    store = EotStore(REPO_ROOT / cfg["data_dir"])
    augmenter = Augmenter(store, REPO_ROOT / cfg["augment_config"], cfg["p_phone"], cfg["seed"])

    calib_rows = np.random.default_rng(args.seed).choice(store.rows("train", tuple(cfg["langs"])), args.calib_clips, replace=False)
    t0 = time.time()
    calib = features_for(augmenter, [(DEV_CLEAN, int(r)) for r in calib_rows])  # the DEV_CLEAN key = no augmentation
    print(f"calibration features: {len(calib_rows)} clean train clips ({np.mean(calib['language'] == 'hin'):.0%} Hindi), {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    quantize_eot(fp32, int8, calib["features"], args.method)
    print(f"quantized ({args.method}) in {time.time() - t0:.0f}s: {fp32.stat().st_size / 1e6:.1f} MB -> {int8.stat().st_size / 1e6:.1f} MB", flush=True)

    dev_rows = store.rows("dev", tuple(cfg["langs"]))
    dev = features_for(augmenter, [(DEV_CLEAN, int(r)) for r in dev_rows] + [(DEV_PHONE, int(r)) for r in dev_rows])
    p32, p8 = probs(fp32, dev["features"]), probs(int8, dev["features"])
    graft_diff = float(np.abs(p32 - probs(exported, dev["features"])).max())
    print(f"Pipecat graph vs {exported.name}: max |dp| {graft_diff:.2e} on {len(p32)} dev clips", flush=True)
    if graft_diff > 1e-5:
        raise SystemExit("the weights did not transfer into Pipecat's graph")
    rows = []
    for cond, is_phone in (("clean", False), ("phone", True)):
        for lang in cfg["langs"]:
            sel = (dev["phone"] == is_phone) & (dev["language"] == lang)
            m32, m8 = binary_metrics(dev["labels"][sel], p32[sel], 0.5), binary_metrics(dev["labels"][sel], p8[sel], 0.5)
            rows.append({"condition": cond, "language": lang, "n": int(sel.sum()),
                         "acc_fp32": round(m32["accuracy"], 2), "acc_int8": round(m8["accuracy"], 2),
                         "auc_fp32": round(m32["auc"], 4), "auc_int8": round(m8["auc"], 4),
                         "cut_in_fp32": round(m32["fpr"], 4), "cut_in_int8": round(m8["fpr"], 4),
                         "decision_agreement": round(float(((p32[sel] > 0.5) == (p8[sel] > 0.5)).mean()), 4),
                         "max_abs_diff": round(float(np.abs(p32[sel] - p8[sel]).max()), 4)})
            print(" ".join(f"{k} {v}" for k, v in rows[-1].items()), flush=True)

    from pehleaap.baselines.smart_turn import download_weights

    smart_turn_int8 = download_weights("3.2", "cpu")
    latency = {name: {t: round(latency_ms(path, t), 2) for t in (1, 4)}
               for name, path in (("smart_turn_v3.2_int8", smart_turn_int8), (f"{args.run}_fp32", fp32), (f"{args.run}_int8", int8))}
    for name, t in latency.items():
        print(f"latency {name}: {t[1]} ms (1 thread), {t[4]} ms (4 threads)", flush=True)

    summary = {"source": str(fp32.relative_to(REPO_ROOT)), "graph": "pipecat smart-turn-v3.2-gpu.onnx", "best_step": best["step"],
               "graft_max_abs_diff": graft_diff, "method": args.method, "calib_clips": int(len(calib_rows)), "calib": "clean train clips",
               "seed": args.seed, "size_mb": {"fp32": round(fp32.stat().st_size / 1e6, 2), "int8": round(int8.stat().st_size / 1e6, 2)},
               "dev": rows, "latency_ms_batch1": latency}
    (run / "quantize.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {int8} and {run / 'quantize.json'}")


if __name__ == "__main__":
    main()
