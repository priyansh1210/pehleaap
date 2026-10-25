---
name: compute-routing
description: "Priyansh's rule: route each part of a job to the device it is faster on (GPU for GPU-friendly work, CPU for the rest), not all-CPU or all-GPU"
metadata:
  type: feedback
---

For every run, put each piece of work on the device it runs faster on: GPU-friendly work (fp32/fp16 model inference,
training, batch ASR) on the GPU; CPU-friendly work (int8-quantized ONNX models, ffmpeg codec simulation, audio
preprocessing, VAD) on the CPU, and run both at once when the job allows it.

**Why:** on 2026-10-24 the GPU box's baseline eval (Job 1) ran CPU-only at 100% CPU while its 20 GB GPU sat at 1%.
Priyansh said: "the process requires GPU or faster on it - can use GPU and others who are better on CPU runs on CPU".
**How to apply:**
- Before a heavy run, say which parts go to the GPU and which stay on the CPU, and why.
- Measure, don't guess. Time a model on both devices on a small slice, and check GPU vs CPU parity (probabilities within
  ~1e-3), before switching a model.
- Don't mix devices for one model within a single reported table, or the model-card comparison gets muddied.
- `pehleaap/baselines/turn_models.py` is CPU-only for now (`providers=["CPUExecutionProvider"]`, plain `onnxruntime`).
  GPU inference needs `onnxruntime-gpu` (its own venv or swapped in carefully) and a per-model provider choice.
- This applies on the GPU box (until 2026-10-28) and Kaggle (T4). The Laptop's GTX 1650 4 GB rarely helps.

Related: [[two-machine-setup]], [[dev-environment]].
