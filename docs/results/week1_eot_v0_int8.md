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
  latency allows. Dev-tuned Hindi thresholds win back about half of the loss (test set below).
- The Hindi dev set is only 344 clips. The test set below is the real check.

## Test set (Smart Turn's test set, the same clips and 11 conditions as week1_eot_v0_report.md)

Run: `eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx eot_v0=runs\eot_v0\model_pipecat_graph.onnx
eot_v0_int8=runs\eot_v0\model_int8.onnx` (108 min on CPU; `week1_eot_v0_int8_eval_report.md`, `_results.json`).
Calibration: `calibrate_thresholds.py --runs eot_v0_int8_eval --shipped eot_v0=runs\eot_v0\summary.json
eot_v0_int8=runs\eot_v0\int8_summary.json --out calibration_int8` (`week1_calibration_int8_report.md`, `_results.json`).

Sanity checks:
- eot_v0 in Pipecat's graph scores the test clips within 4.5e-6 of the TorchScript export in `eot_v0_eval`.
- Smart Turn's predictions are identical to `eot_v0_eval`.

**int8 dev-tuned thresholds** (`week1_eot_v0_int8_dev_thresholds.json`) are accuracy-optimal on the dev split, which has no test
clips. They were computed after quantising, with `best_threshold` on the same dev keys as training: clean/eng 0.391, clean/hin 0.814,
phone/eng 0.369, phone/hin 0.713.

### Hindi (n = 1284): accuracy, with the cut-in rate in brackets

| Condition | eot_v0 fp32 @0.5 | eot_v0 fp32 shipped | int8 @0.5 | int8 dev-tuned | Smart Turn int8 @0.5 |
|---|---|---|---|---|---|
| noise 5 dB + AMR 7.95 | 88.6 (15%) | 88.8 (16%) | 86.3 (26%) | 87.5 (19%) | 69.5 (63%) |
| noise 10 dB + AMR 7.95 | 91.2 (10%) | 91.4 (11%) | 89.0 (21%) | 89.6 (16%) | 73.7 (55%) |
| noise 20 dB + AMR 7.95 | 93.6 (8%) | 93.5 (9%) | 90.0 (18%) | 91.4 (14%) | 77.4 (47%) |
| mixed chain | 91.9 (9%) | 92.2 (10%) | 89.0 (20%) | 90.3 (13%) | 73.5 (55%) |
| tandem AMR 4.75 → G.711 | 93.1 (7%) | 93.1 (8%) | 90.7 (17%) | 91.8 (12%) | 76.4 (49%) |
| G.711 + 3% loss | 94.2 (6%) | 93.7 (7%) | 91.7 (13%) | 92.8 (10%) | 82.1 (36%) |
| AGC + clip + G.711 | 94.7 (6%) | 94.3 (7%) | 92.0 (13%) | 93.3 (9%) | 82.6 (35%) |
| AMR-NB 4.75 | 93.3 (7%) | 93.3 (8%) | 91.0 (16%) | 92.1 (12%) | 80.5 (40%) |
| AMR-NB 7.95 | 93.2 (8%) | 93.2 (8%) | 91.5 (15%) | 92.5 (11%) | 82.8 (35%) |
| G.711 μ-law | 94.4 (6%) | 93.9 (7%) | 92.8 (12%) | 93.7 (9%) | 86.4 (27%) |
| clean | 94.1 (6%) | 94.1 (6%) | 92.8 (12%) | 93.6 (7%) | 90.3 (18%) |

### English phone subset (n = 1947)

| Condition | eot_v0 fp32 @0.5 | eot_v0 fp32 shipped | int8 @0.5 | int8 dev-tuned | Smart Turn int8 @0.5 |
|---|---|---|---|---|---|
| noise 5 dB + AMR 7.95 | 90.6 (11%) | 90.5 (11%) | 89.6 (13%) | 89.5 (16%) | 81.7 (32%) |
| noise 10 dB + AMR 7.95 | 92.1 (10%) | 92.2 (10%) | 91.6 (10%) | 91.1 (13%) | 85.4 (24%) |
| noise 20 dB + AMR 7.95 | 92.2 (10%) | 92.2 (9%) | 92.5 (9%) | 92.4 (11%) | 88.1 (19%) |
| mixed chain | 92.5 (9%) | 92.6 (9%) | 91.2 (11%) | 90.8 (14%) | 84.3 (27%) |
| tandem AMR 4.75 → G.711 | 92.8 (9%) | 92.7 (9%) | 91.5 (11%) | 91.6 (12%) | 89.5 (17%) |
| G.711 + 3% loss | 93.6 (8%) | 93.5 (8%) | 93.4 (9%) | 93.2 (10%) | 89.5 (16%) |
| AGC + clip + G.711 | 93.4 (8%) | 93.3 (8%) | 93.0 (9%) | 93.1 (10%) | 89.4 (16%) |
| AMR-NB 4.75 | 92.4 (8%) | 92.4 (8%) | 92.1 (9%) | 92.6 (10%) | 90.8 (14%) |
| AMR-NB 7.95 | 93.5 (8%) | 93.5 (8%) | 93.2 (8%) | 93.4 (9%) | 91.7 (12%) |
| G.711 μ-law | 93.5 (7%) | 93.5 (7%) | 93.5 (8%) | 93.6 (9%) | 92.1 (11%) |
| clean | 94.3 (7%) | 94.4 (7%) | 94.0 (7%) | 93.9 (8%) | 93.0 (9%) |

Findings:
- **The dev split was right: int8 costs about 2 points on Hindi.** At 0.5, int8 is 1.3-3.7 points below fp32 on Hindi (mean -2.3),
  and Hindi cut-ins roughly double (6-15% → 12-26%). English loses 0.5 on average (at most 1.3).
- **The Hindi dev-tuned thresholds recover about half the loss.** With them, Hindi is 0.5-2.2 points below fp32 (mean -1.2),
  and cut-ins fall to 7-19%. The English dev-tuned thresholds (0.37-0.39) don't help on the test set: they add cut-ins and cost
  up to 0.5 points. **Ship int8 with Hindi thresholds 0.71 (phone) / 0.81 (clean) and English at 0.5.**
- **int8 eot_v0 still beats Smart Turn int8 everywhere.** On Hindi, at the same int8 speed (28.6 ms, one thread), it is
  +18.0 points at 5 dB SNR, +16.8 on the mixed chain and +3.3 on clean, with the dev-tuned thresholds.
- **AUC barely moves** (Hindi int8 0.953-0.985 against fp32 0.964-0.987), so most of the int8 loss is calibration, as on dev.

**Recommendation.** Use **fp32 eot_v0** wherever about 57 ms (one thread) or 17 ms (four threads) per decision is affordable, as in
the demo. Use **int8 with the Hindi dev-tuned thresholds** only where CPU time is tight. It keeps most of the gain at Smart
Turn's own cost.
