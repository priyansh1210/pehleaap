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
- GPU inference (Nov 4, branch gpu-eval): `turn_models` `device="cuda"` / the eval scripts' `--device cuda`, run from
  `.venv-gpu` (onnxruntime-gpu[cuda,cudnn] 1.31.0). fp32 models: TF32 off, |dp| <= 1.1e-5, 0 flips, ~10x faster.
  Quantized models stay on the CPU automatically, since CUDA EP int8 drifted by up to 0.45 with 20/512 flips. torch must be
  imported before `ort.preload_dlls()` (their cuDNN DLLs clash); `_load_cuda_dlls` does that. Default stays CPU.
- Batch evals: pass `--no-spin` (opt-in, Priyansh's choice): spinning ORT threads starved torch feature extraction.
- This applies on the GPU box (until 2026-10-28) and Kaggle (T4). The Laptop's GTX 1650 4 GB rarely helps.

Related: [[two-machine-setup]], [[dev-environment]].
