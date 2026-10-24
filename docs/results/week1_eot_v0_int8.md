# eot_v0 int8 (GPU box, Oct 25 2026)

`scripts/quantize_eot.py --run eot_v0` (`pehleaap/models/quantize.py`). Numbers: `week1_eot_v0_int8_quantize.json`.
The int8 file (8.7 MB, `runs/eot_v0/model_int8.onnx`) and the fp32 Pipecat-graph file (32.4 MB) stay on the GPU box, not in git.

How it is made:
1. eot_v0's best checkpoint (step 900) goes into Pipecat's own fp32 graph (`smart-turn-v3.2-gpu.onnx`) with
   `onnx_with_state_dict`, the reverse of `state_dict_from_onnx`. It matches our TorchScript export within 2.95e-6 on
   all 4622 dev clips, so it ships in exactly the graph Pipecat publishes.
2. Upstream's recipe (pipecat-ai/smart-turn train.py, `quantize_onnx_model`): `quant_pre_process` (symbolic shape inference
   skipped), static QDQ, uint8 activations, int8 per-channel weights, entropy calibration, **Conv/MatMul/Gemm only**,
   calibrated on 1024 **clean** training clips. These are never dev or test clips.

## Dev split (the clips train_eot.py scores; threshold 0.5)

| Dev set | n | Acc fp32 | Acc int8 | AUC fp32 | AUC int8 | Cut-in fp32 | Cut-in int8 | Same decision |
|---|---|---|---|---|---|---|---|---|
| phone/hin | 344 | 92.15 | 90.12 | 0.976 | 0.975 | 8.7% | 16.9% | 92.7% |
| phone/eng | 1967 | 93.34 | 91.66 | 0.980 | 0.971 | 8.5% | 10.7% | 95.9% |
| clean/hin | 344 | 95.06 | 93.02 | 0.989 | 0.986 | 6.0% | 9.8% | 96.8% |
| clean/eng | 1967 | 97.36 | 96.75 | 0.993 | 0.992 | 3.2% | 3.8% | 98.4% |

The fp32 numbers are ONNX fp32. They differ slightly from training's bf16 dev scores (for example, phone/eng 93.34 against 93.75).

**Reference: Pipecat's own int8 loses more on these clips.**

| Dev set | Smart Turn fp32 | Smart Turn int8 (published) |
|---|---|---|
| phone/hin | 81.98 | 71.80 |
| phone/eng | 87.90 | 85.00 |
| clean/hin | 97.09 | 93.02 |
| clean/eng | 98.53 | 97.41 |

**CPU latency, one clip, batch 1, median of 100 runs on the GPU box's CPU:**

| Model | 1 thread | 4 threads |
|---|---|---|
| eot_v0 fp32 | 56.9 ms | 17.1 ms |
| eot_v0 int8 | 28.6 ms | 9.9 ms |
| Smart Turn int8 (published) | 28.6 ms | 9.9 ms |

## What went wrong first, and the diagnostics

Our first pipeline quantised every op type onnxruntime supports and calibrated on training's own mix (80% phone line).
It shifted eot_v0's probabilities by -0.1 to -0.2, and dev accuracy fell by up to 23 points while AUC held. Entropy,
min-max and percentile calibration all failed this way. Keeping the pooling and classifier head in float only halved
the shift.

Two checks found the cause:
- **Our pipeline with upstream's recipe reproduces Pipecat's published int8 from their fp32 file:** mean shift +0.034 against
  +0.033, and identical phone/hin accuracy (71.80).
- **With upstream's recipe, the graph hardly matters:** eot_v0's TorchScript graph and Pipecat's graph give the same int8
  accuracy within 0.9 points.

The fix was the recipe itself: quantising only Conv/MatMul/Gemm, and calibrating on clean clips.
The histogram calibrator also keeps every activation of every clip in RAM, about 150 MB per clip, so 1024 clips ran out of
memory. `quantize.py` feeds it 32 clips at a time, and the histograms merge across calls.

## Read with care

- **int8 costs 0.6-2.0 dev points, and Hindi phone-line cut-ins nearly double (8.7% → 16.9%).** int8 eot_v0 is still far above
  Smart Turn on the phone line (90.1 against 82.0 fp32 or 71.8 int8, on phone/hin). But the fp32 model is the one to use where
  latency allows. A dev-tuned threshold for int8 would likely win back some of the cut-in rate. That is not done yet.
- The Hindi dev set is only 344 clips. A test-set run (`eval_clip_baselines.py --onnx eot_v0_int8=runs\eot_v0\model_int8.onnx`,
  11 conditions, about 75 min on CPU) is the real check, and it has not been run yet.
