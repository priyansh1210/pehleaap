# eot_v1 int8 (GPU box, Oct 25 2026)

`scripts/quantize_eot.py --run eot_v1`, the same recipe as eot_v0's int8 (`week1_eot_v0_int8.md`). eot_v1's best checkpoint
(step 400) goes into Pipecat's fp32 graph, matching our export within 2.9e-6 on all 4,832 dev clips. Then QDQ with entropy
calibration on 1,024 clean training clips, Conv/MatMul/Gemm only.

Numbers: `week1_eot_v1_int8_quantize.json` (dev + latency), `week1_eot_v1_int8_dev_thresholds.json` (dev thresholds per
group, made by a one-off script with the same dev keys and seed as training), `week1_eot_v1_int8_fleurs_results.json`.
Files (GPU box, not in git):
- `runs/eot_v1/model_int8.onnx`: 8.7 MB, sha256 fe3c6e0a…0356f7.
- `runs/eot_v1/model_pipecat_graph.onnx`: 32.4 MB, sha256 d30e81b4…1e4f.

## Dev split (threshold 0.5)

| Dev set | n | fp32 | int8 | Cut-in fp32 | Cut-in int8 | int8 dev-tuned threshold | int8 at that threshold |
|---|---|---|---|---|---|---|---|
| phone / Hindi TTS | 344 | 91.6 | 85.8 | 8.7% | 25.0% | 0.75 | 90.1 (12.5%) |
| phone / Hindi real | 105 | 85.7 | 78.1 | 23.3% | 44.2% | 0.85 | 82.9 (9.3%) |
| phone / English | 1967 | 93.0 | 91.1 | 8.7% | 12.8% | 0.44 | 91.4 (14.1%) |
| clean / Hindi TTS | 344 | 93.9 | 89.8 | 6.5% | 15.8% | 0.89 | 93.6 (5.4%) |
| clean / Hindi real | 105 | 93.3 | 84.8 | 9.3% | 37.2% | 0.91 | 92.4 (4.7%) |
| clean / English | 1967 | 96.6 | 95.7 | 4.2% | 5.8% | 0.46 | 95.9 (6.1%) |

int8 costs eot_v1 more than it cost eot_v0 (Hindi −4 to −6 points against −2). AUC barely moves (phone/hin 0.969 → 0.959):
int8 shifts Hindi probabilities **up**, so at 0.5 it calls many unfinished turns finished. A higher Hindi threshold undoes most of it.

**CPU latency** (batch 1, median of 100 runs, GPU box CPU): eot_v1 fp32 88.5 ms (1 thread) / 25.0 ms (4 threads), int8 39.2 / 12.5 ms,
Smart Turn int8 39.5 / 12.1 ms. int8 is 2.3× faster than fp32, the same as Smart Turn's int8. All three are slower than in the eot_v0 run
(57 / 29 ms), on the same machine and model size, so the CPU was slower during this run. Compare the ratios, not the milliseconds.

## Real voices: FLEURS Hindi test (n = 1046), accuracy (cut-in rate)

`eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin --models smart_turn_v3.2_int8 --onnx
eot_v1=runs\eot_v1\model_pipecat_graph.onnx eot_v1_int8=runs\eot_v1\model_int8.onnx` (14 min). eot_v1 in Pipecat's graph
gives exactly the same numbers as the TorchScript export in `week1_eot_v1_report.md`.

Dev-tuned Hindi thresholds come from the Hindi TTS dev clips (clean 0.89, phone 0.75). They see no FLEURS test clips.

| Condition | eot_v1 fp32 @0.5 | int8 @0.5 | int8, Hindi dev thresholds | int8, real-voice dev thresholds (0.91 / 0.85) | Smart Turn int8 @0.5 |
|---|---|---|---|---|---|
| clean | **88.8** (18%) | 86.1 (34%) | 84.4 (14%, missed 16%) | 83.2 (13%) | 72.8 (60%) |
| G.711 μ-law | **86.7** (23%) | 84.3 (39%) | 85.5 (29%) | 85.2 (20%) | 69.4 (73%) |
| AMR-NB 7.95 | **86.9** (25%) | 81.5 (48%) | 85.7 (32%) | 83.9 (20%) | 68.4 (77%) |
| noise 10 dB + AMR 7.95 | **83.4** (33%) | 78.3 (56%) | 81.6 (34%) | 77.9 (22%) | 65.2 (84%) |
| mixed chain | **82.8** (31%) | 79.4 (52%) | 81.5 (32%) | 77.3 (20%) | 65.7 (79%) |

- **At 0.5, int8 loses 2.4-5.4 points on real voices and its cut-ins nearly double** (18-33% → 34-56%). Missed ends fall to
  1-2%: the model now leans towards "finished".
- **The Hindi dev thresholds bring cut-ins back to fp32's level** (14-34% against 18-33%). On the phone conditions this also recovers
  most of the accuracy (−1.1 to −1.8 against fp32). On clean audio, 0.89 is too strict for real voices: cut-ins are lowest
  (14%), but 16% of finished sentences are missed, and accuracy is 84.4.
- The thresholds from the 105 real-voice dev clips are stricter still: fewest cut-ins (13-22%), but accuracy drops under noise.
  They are tuned on too few clips to trust.
- **int8 eot_v1 still beats Smart Turn's published int8 by 11-17 points everywhere on real voices**, with half or fewer of its cut-ins.

## Decision guide

- **Use fp32 eot_v1 where its CPU cost is fine** (the demo, a few calls per core). It is the best model on every test.
- **Use int8 where the CPU is the limit**, with Hindi thresholds **0.75 on phone audio** (every real call) and 0.5 for English.
  On phone-line real voices it then sits 1-2 points below fp32 with the same cut-in rate, 2.3× faster.
- The TTS test set below confirms it: with the Hindi thresholds, int8 is 1.0-2.0 points below fp32 on Hindi, and English needs no change.

## TTS test set (Smart Turn's test set, the same clips and 7 conditions as `week1_eot_v1_report.md`)

Run: `eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx eot_v1=runs\eot_v1\model_pipecat_graph.onnx
eot_v1_int8=runs\eot_v1\model_int8.onnx --conditions clean g711_mulaw amrnb_7.95 amrnb_4.75 noise10_amr7.95 noise5_amr7.95
mixed --threads 24` (GPU box, 54 min; `week1_eot_v1_int8_eval_report.md`, `_results.json`).
Sanity checks: eot_v1 in Pipecat's graph scores all 28,490 clip-conditions within 5.1e-6 of the TorchScript export in `eot_v1_eval`,
and Smart Turn's predictions are identical to `eot_v1_eval`.

The Hindi thresholds are the dev-tuned ones from above (clean 0.89, phone 0.75). English stays at 0.5. p is an exact two-sided
McNemar test, fp32 @0.5 against int8 with Hindi thresholds, on the same clips.

| Condition | Hindi fp32 @0.5 | Hindi int8 @0.5 | Hindi int8, Hindi thresholds | Smart Turn int8 | p | English fp32 | English int8 @0.5 | Smart Turn int8 |
|---|---|---|---|---|---|---|---|---|
| clean | 93.7 (7%) | 91.1 (15%) | 92.7 (5%) | 90.3 (18%) | 0.10 | 94.7 | 94.3 | 94.3 |
| G.711 μ-law | 93.8 (8%) | 89.9 (18%) | 92.1 (11%) | 86.4 (27%) | 0.002 | 92.7 | 93.0 | 92.1 |
| AMR-NB 7.95 | 93.9 (7%) | 89.6 (20%) | 92.4 (10%) | 82.8 (35%) | 0.01 | 93.0 | 92.3 | 91.7 |
| AMR-NB 4.75 | 93.1 (8%) | 88.2 (23%) | 91.3 (13%) | 80.5 (40%) | 0.003 | 92.3 | 91.8 | 90.8 |
| noise 10 dB + AMR 7.95 | 91.3 (12%) | 87.7 (25%) | 89.6 (16%) | 73.7 (55%) | 0.01 | 91.1 | 90.2 | 85.4 |
| noise 5 dB + AMR 7.95 | 89.3 (15%) | 84.9 (30%) | 87.3 (19%) | 69.5 (63%) | 0.004 | 88.9 | 87.8 | 81.7 |
| mixed chain | 91.2 (11%) | 87.5 (24%) | 89.9 (14%) | 73.5 (55%) | 0.05 | 91.3 | 89.8 | 84.3 |

Hindi n = 1284 per condition. English n = 7820 clean and 1947 (25% hash sample) per phone condition.

- **At 0.5, int8 loses 2.6-4.9 points on Hindi TTS, and Hindi cut-ins double** (7-15% → 15-30%), as the dev split predicted.
- **The Hindi thresholds recover most of it.** int8 is 1.0-2.0 points below fp32 (clean −1.0, p = 0.10; phone conditions −1.3 to
  −2.0, p = 0.002-0.05), with cut-ins at 5-19% against fp32's 7-15%. eot_v0's int8 with its own Hindi thresholds was 0.5-2.2
  below its fp32: the same size of loss.
- **English needs no special threshold.** int8 at 0.5 is within 1.5 points of fp32 everywhere (clean −0.4; mixed chain −1.5, p = 0.004).
- **int8 eot_v1 with Hindi thresholds beats Smart Turn's int8 in every condition**: +2.4 on clean Hindi and +16 to +18 under noise.
