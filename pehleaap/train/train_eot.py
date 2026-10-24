"""Fine-tune Smart Turn v3.2 for the phone line (plan §6, Tier A EOT v0).

Starts from the published v3.2 weights (`models/smart_turn.py`) and trains on Smart Turn's own
training clips (English + Hindi), most of them sent through the phone-line augmentation chain
on the fly (`eot_data.py`). Upstream's recipe otherwise: AdamW, BCE with a per-batch positive
weight, effective batch 384.

Runs unchanged on the GPU box (bf16), Kaggle's T4 (fp16 + GradScaler), the Dev box's GTX 1650
(fp16, smoke runs) or the CPU. Every `checkpoint_minutes` it saves runs/<out>/ckpt.pt; rerunning
the same command resumes from it, on exactly the batches it would have seen next.

Outputs in runs/<out>/: meta.json (commit, config, device), train.csv (loss, lr, speed), dev.csv
(accuracy, AUC and cut-in rate per language, clean and phone line, every eval; step 0 is the
published model), best.pt (best dev phone-line accuracy), model.onnx (best, same inputs and
outputs as Smart Turn's ONNX, so `WhisperMelOnnx` scores it like the baselines), summary.json.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pehleaap.eval.calibration import best_threshold
from pehleaap.eval.clip_metrics import binary_metrics
from pehleaap.models.smart_turn import SmartTurnV3, from_published
from pehleaap.paths import REPO_ROOT, RUNS_DIR
from pehleaap.train.eot_data import DEV_CLEAN, DEV_PHONE, Augmenter, EotStore, EpochKeys, Keys


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/external/eot-train"  # relative to the repo root
    langs: tuple[str, ...] = ("eng", "hin")
    hin_repeat: int = 2  # Hindi clips appear this many times per epoch (each with fresh augmentation)
    augment_config: str = "configs/augment.yaml"
    p_phone: float = 0.8  # share of training clips sent through the phone-line chain; the rest stay clean 16 kHz
    epochs: int = 4
    batch_size: int = 384  # effective (upstream's); micro-batches are accumulated up to it
    micro_batch: int = 192
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # linear warmup, then cosine decay to 0
    grad_clip: float = 1.0
    num_workers: int = 20  # CPU processes running augmentation + features
    eval_every_steps: int = 100
    dev_limit: int | None = None  # score only this many dev clips (smoke runs)
    checkpoint_minutes: float = 30.0
    seed: int = 0
    precision: str = "auto"  # auto, bf16, fp16 or fp32

    def __post_init__(self):
        if self.batch_size % self.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")


def load_config(path: str | Path | None = None, overrides: list[str] = ()) -> TrainConfig:
    """Defaults, then a YAML file, then key=value overrides (values parsed as YAML)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {} if path else {}
    for item in overrides:
        key, _, value = item.partition("=")
        raw[key] = yaml.safe_load(value)
    return TrainConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def precision_for(device: torch.device, setting: str = "auto") -> str:
    """bf16 on Ampere and newer (the GPU box's Ada), fp16 on older GPUs (Kaggle's T4), fp32 on the CPU.

    GTX 16xx cards (the Dev box's GTX 1650) get fp32: their fp16 convolutions return NaN for this
    model from the first layer on (checked 2026-10-24), a known quirk of those cards.
    """
    if setting != "auto":
        return setting
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.get_device_capability(device) >= (8, 0):
        return "bf16"
    return "fp32" if "GTX 16" in torch.cuda.get_device_name(device) else "fp16"


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    return f


def bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Upstream's loss: positives weighted by the batch's negative/positive ratio, clamped to [0.1, 10]."""
    pos_weight = ((labels == 0).sum() / (labels == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def loader(keys, augmenter: Augmenter, batch: int, workers: int, pin: bool) -> DataLoader:
    return DataLoader(
        Keys(), batch_size=batch, sampler=keys, collate_fn=augmenter, num_workers=workers,
        persistent_workers=False, prefetch_factor=4 if workers else None, pin_memory=pin,
    )


@torch.no_grad()
def evaluate(model: SmartTurnV3, dev: DataLoader, device: torch.device, autocast) -> list[dict]:
    """Accuracy, AUC and cut-in rate on the dev clips, per language, clean and through the phone line,
    plus the accuracy-optimal threshold for each (what the model should ship with, see eval/calibration.py)."""
    model.eval()
    probs, labels, langs, phone = [], [], [], []
    for b in dev:
        with autocast():
            probs.append(torch.sigmoid(model(b["features"].to(device)).float()).cpu().numpy())
        labels.append(b["labels"].numpy())
        langs += b["language"]
        phone.append(b["phone"].numpy())
    model.train()
    probs, labels, phone, langs = np.concatenate(probs), np.concatenate(labels), np.concatenate(phone), np.array(langs)
    rows = []
    for cond, is_phone in (("clean", False), ("phone", True)):
        for lang in sorted(set(langs)):
            sel = (phone == is_phone) & (langs == lang)
            m = binary_metrics(labels[sel], probs[sel], 0.5)
            t = best_threshold(labels[sel], probs[sel])
            rows.append({"condition": cond, "language": lang, "n": m["n"], "accuracy": round(m["accuracy"], 2),
                         "auc": round(m["auc"], 4), "cut_in": round(m["fpr"], 4), "missed_end": round(m["fnr"], 4),
                         "best_threshold": round(t, 4), "accuracy_at_best": round(binary_metrics(labels[sel], probs[sel], t)["accuracy"], 2)})
    return rows


def dev_score(rows: list[dict]) -> float:
    """What "best" means: phone-line accuracy at the 0.5 threshold, averaged over languages."""
    return float(np.mean([r["accuracy"] for r in rows if r["condition"] == "phone"]))


def export_onnx(model: SmartTurnV3, path: Path) -> None:
    """Same graph interface as Smart Turn's published ONNX: input_features (B, 80, 800) -> P(complete) (B, 1)."""

    class Probs(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_features):
            return torch.sigmoid(self.inner(input_features)).unsqueeze(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch 2.14 calls the TorchScript exporter legacy; it needs no onnxscript
        torch.onnx.export(
            Probs(model).eval().cpu(), (torch.zeros(2, 80, 800),), str(path), dynamo=False, opset_version=18,
            input_names=["input_features"], output_names=["logits"], dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
        )


def append_csv(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def train(cfg: TrainConfig, out: Path, max_steps: int | None = None, device: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    dev_type = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = precision_for(dev_type, cfg.precision)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def autocast():
        return torch.autocast(dev_type.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    store = EotStore(REPO_ROOT / cfg.data_dir)
    train_rows = store.rows("train", cfg.langs)
    dev_rows = store.rows("dev", cfg.langs)[: cfg.dev_limit]
    if not len(train_rows):
        raise SystemExit(f"no training clips in {store.root}: run scripts/prepare_eot_train.py first")
    repeat = np.where(store.index.language.to_numpy()[train_rows] == "hin", cfg.hin_repeat, 1)
    keys = EpochKeys(train_rows, repeat, cfg.epochs, cfg.batch_size, cfg.seed)
    steps_per_epoch = keys.epoch_len // cfg.batch_size
    total_steps = min(cfg.epochs * steps_per_epoch, max_steps or 10**9)
    accum = cfg.batch_size // cfg.micro_batch

    model = from_published().to(dev_type).train()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]  # biases and norms
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(cfg.warmup_ratio * total_steps), total_steps))
    scaler = torch.amp.GradScaler(dev_type.type, enabled=precision == "fp16")

    ckpt_path, best_path = out / "ckpt.pt", out / "best.pt"
    step, best = 0, -1.0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=dev_type, weights_only=False)
        if state["config"] != asdict(cfg):
            raise SystemExit(f"{ckpt_path} was made with a different config; use a new --out or delete it")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, best = state["step"], state["best"]
        print(f"resumed at step {step}/{total_steps} (best dev phone accuracy so far {best:.2f})", flush=True)
    else:
        meta = {"commit": git_commit(), "config": asdict(cfg), "device": torch.cuda.get_device_name(dev_type) if dev_type.type == "cuda" else "cpu",
                "precision": precision, "torch": torch.__version__, "n_train": len(train_rows), "n_dev": len(dev_rows),
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "max_steps": max_steps, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pin = dev_type.type == "cuda"
    augment_config = REPO_ROOT / cfg.augment_config
    dev_keys = [(DEV_CLEAN, int(r)) for r in dev_rows] + [(DEV_PHONE, int(r)) for r in dev_rows]
    # A few workers are enough for the dev set, and keep memory down while the training workers are alive too.
    dev = loader(dev_keys, Augmenter(store, augment_config, cfg.p_phone, cfg.seed), cfg.micro_batch, min(cfg.num_workers, 4), pin)

    def save_checkpoint():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "config": asdict(cfg)}, tmp)
        tmp.replace(ckpt_path)

    def run_eval():
        nonlocal best
        rows = evaluate(model, dev, dev_type, autocast)
        append_csv(out / "dev.csv", [{"step": step, **r} for r in rows])
        score = dev_score(rows)
        line = "  ".join(f"{r['condition']}/{r['language']} acc {r['accuracy']:.1f} auc {r['auc']:.3f} cut-in {100 * r['cut_in']:.1f}%" for r in rows)
        print(f"[dev @ step {step}] {line}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "step": step, "dev": rows}, best_path)

    if step == 0:
        run_eval()  # the published model: the bar to beat
    train_keys = EpochKeys(train_rows, repeat, cfg.epochs, cfg.batch_size, cfg.seed, start=step * cfg.batch_size)
    t_ckpt = t_log = time.time()
    seen, micro, loss_sum = 0, 0, 0.0
    if step < total_steps:
        print(f"training {total_steps - step} steps on {dev_type} ({precision}), {len(train_rows)} clips, {steps_per_epoch} steps/epoch", flush=True)
    for b in loader(train_keys, Augmenter(store, augment_config, cfg.p_phone, cfg.seed), cfg.micro_batch, cfg.num_workers, pin) if step < total_steps else []:
        with autocast():
            logits = model(b["features"].to(dev_type, non_blocking=True))
        loss = bce(logits, b["labels"].to(dev_type)) / accum
        if not torch.isfinite(loss):
            raise SystemExit(f"loss is {loss.item()} at step {step} ({precision}); try --set precision=fp32")
        scaler.scale(loss).backward()
        loss_sum += loss.item()
        seen += len(b["labels"])
        micro += 1
        if micro % accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 10 == 0 or step == total_steps:
            rate = seen / (time.time() - t_log)
            append_csv(out / "train.csv", [{"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": round(loss_sum / (seen / cfg.batch_size), 5),
                                            "lr": f"{sched.get_last_lr()[0]:.3e}", "clips_per_s": round(rate, 1)}])
            eta = (total_steps - step) * cfg.batch_size / rate / 60
            print(f"step {step}/{total_steps}  loss {loss_sum / (seen / cfg.batch_size):.4f}  {rate:.0f} clips/s  ~{eta:.0f} min left", flush=True)
            t_log, seen, loss_sum = time.time(), 0, 0.0
        if step % cfg.eval_every_steps == 0 or step == total_steps:
            run_eval()
        if time.time() - t_ckpt > 60 * cfg.checkpoint_minutes or step == total_steps:
            save_checkpoint()
            t_ckpt = time.time()
        if step >= total_steps:
            break

    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    final = SmartTurnV3()
    final.load_state_dict(best_state["model"])
    export_onnx(final, out / "model.onnx")
    # .get: a run started before thresholds were recorded (best.pt from older code) still finishes
    thresholds = {f"{r['condition']}/{r['language']}": r.get("best_threshold") for r in best_state["dev"]}
    summary = {"best_step": best_state["step"], "best_dev": best_state["dev"], "dev_score": dev_score(best_state["dev"]), "steps": step,
               "dev_thresholds": thresholds}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"best dev phone accuracy {summary['dev_score']:.2f} at step {summary['best_step']}; exported {out / 'model.onnx'}", flush=True)
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_eot.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields, e.g. --set micro_batch=8 num_workers=2")
    ap.add_argument("--out", required=True, help="run name: outputs go to runs/<out>/")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke runs)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg, RUNS_DIR / args.out, args.max_steps, args.device)
