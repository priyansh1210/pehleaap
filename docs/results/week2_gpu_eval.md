# Evals on the GPU: `--device cuda` (GPU box, 2026-11-04)

`eval_sessions.py` and `eval_clip_baselines.py` take `--device cuda` (default `cpu`, unchanged). fp32 models then run on
ONNX Runtime's CUDA provider with TF32 off. Quantized models stay on the CPU. Setup is in `docs/machines.md` (`.venv-gpu`).

Setup: RTX 4000 SFF Ada, driver 581.42 (CUDA 13.0), onnxruntime-gpu 1.31.0 with its `[cuda,cudnn]` pip wheels, and torch
2.14.1+cu126 in the same venv. Without those wheels the CUDA provider fails to load, and ORT silently runs on the CPU.
`turn_models` now raises instead.

## 1. Per model: parity and speed

The first 512 Hindi clips of the FLEURS EOT test split, clean and G.711. Batches of 16. CPU = 4 threads. Time is the model
step only, per clip; the Whisper features take another 1.6 ms per clip on the CPU either way. |dp| = |P_cpu - P_gpu|.
Flips = decisions that change at the model's threshold. Measured with a throwaway script, not kept; section 2 uses only
repo scripts.

| Model | CPU ms | GPU ms | max \|dp\| clean / G.711 | Flips |
|---|---|---|---|---|
| eot_v1 | 15.8 | 1.5 | 4.8e-6 / 3.9e-6 | 0 |
| eot_v0 | 16.2 | 1.5 | 1.1e-5 / 5.4e-6 | 0 |
| Smart Turn v3.2 fp32 | 18.6 | 1.9 | 5.4e-6 / 6.1e-6 | 0 |
| abhishek_tiny_attn | 121.4 | 8.7 | 4.3e-6 / 4.3e-6 | 0 |
| codewithmoin_8s | 67.5 | 52.5 | 3.2e-6 / 2.6e-6 | 0 |
| deveshu_fp32 (batch fixed at 1, not timed) | - | - | 3.9e-6 / 4.9e-6 | 0 |

Two settings would break parity, and the code avoids both:

- **TF32** (ORT's CUDA default on this GPU). With it on, fp32 models moved by up to 6.3e-3 (eot_v0), 3.9e-3 (Smart Turn
  fp32) and 2.8e-3 (eot_v1), and abhishek_tiny_attn flipped 1 decision. That's above the 1e-3 bar.
- **Quantized models on the CUDA provider.** Smart Turn v3.2 int8 moved by up to 0.45 and flipped 20 and 18 of 512 decisions;
  chhotu_int8 moved by up to 0.11 and flipped 5 and 6. They were validated on the CPU, so `turn_models` keeps them there. It
  spots them by their quantization ops: QuantizeLinear, MatMulInteger and the like.

## 2. End to end: TurnBench, 3 sessions

```
python scripts/eval_sessions.py runs/turnbench_labels/TB105 runs/turnbench_labels/TB106 runs/turnbench_labels/TB107
    --channels agent customer --models eot_v1 smart_turn_v3.2_int8 weights/pipecat-ai__smart-turn-v3/smart-turn-v3.2-gpu.onnx
    --conditions clean --threads 20 [--device cuda] --out <out>
```

599 pauses, 1,783 probabilities per model (the 6 checkpoints, minus those after a pause ended).

| | Wall time | vs the saved CPU run (`week2_turnbench_dev.md`) |
|---|---|---|
| `--device cpu` (new code, `.venv`) | 134.5 s | all probabilities bit-identical |
| `--device cuda` (`.venv-gpu`) | 48.5 s | eot_v1 max \|dp\| 3.8e-6, Smart Turn fp32 6.2e-6, int8 bit-identical (CPU); `results.json` identical |

That's 2.8x faster overall. The int8 model, VAD and feature extraction stay on the CPU. At this rate the 38-session dev run
would drop from about 22 to about 8 minutes (extrapolated, not run).

## Known limits

- With torch in the process, ORT runs on torch's cuDNN, its CUDA 12 build. ORT's own cuDNN wheel clashes with torch's by
  DLL name, and loading ORT's first breaks `import torch`. The setup works, as section 1 shows, but it wasn't designed this
  way; a CUDA 13 torch would remove the clash.
- `eval_clip_baselines.py` reuses saved shards. When switching devices, use a new `--out` or `--fresh`, so that one table
  doesn't mix CPU and GPU shards. The difference is within 1.1e-5, but the device isn't recorded per shard.
- Only measured on the GPU box. Kaggle's T4 has no TF32, so the TF32 setting doesn't matter there, but parity there is unchecked.
