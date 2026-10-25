# The other baselines on real human Hindi (FLEURS), Laptop, Oct 25 2026

The FLEURS real-voice test (`week1_fleurs_eot_report.md`, `week1_eot_v1_report.md`) had scored Smart Turn int8, codewithmoin,
eot_v0 and eot_v1. This run adds the other community models and Smart Turn's fp32 file, so that every model we compare has
been through the same test. The data, conditions and thresholds are the same as before: n = 1046 (657 finished, 389 mid-sentence pauses).

Command: `scripts/eval_clip_baselines.py --data-root data/external/fleurs-hi-eot --langs hin --models deveshu_fp32 chhotu_int8
abhishek_tiny_attn --onnx smart_turn_v3.2_fp32=weights/pipecat-ai__smart-turn-v3/smart-turn-v3.2-gpu.onnx --conditions clean
g711_mulaw amrnb_7.95 noise10_amr7.95 mixed --threads 4 --out fleurs_more_baselines` (Laptop, 4 cores, about 50 min).

**Every model on FLEURS** (accuracy %, cut-in rate in brackets). The eot_v1, codewithmoin and Smart Turn int8 columns come
from the GPU box run (`week1_eot_v1_report.md`); the fp32 models score the same on both machines on the four fixed conditions.
*Correction (Oct 25):* the mixed chain's audio is not identical across the two machines (`week1_fleurs_significance.md`:
deveshu and abhishek differ by 1–2 clips; codewithmoin scored 75.43 on the Laptop in `week1_fleurs_eot_results.json` and 75.05
on the GPU box, 4 clips), so gaps in the mixed row between GPU-box and Laptop columns can be off by up to about 0.4 points.

| Condition | eot_v1 | codewithmoin | deveshu | abhishek | Smart Turn fp32 | Smart Turn int8 | chhotu |
|---|---|---|---|---|---|---|---|
| clean | **88.8** (18%) | 81.6 (37%) | 80.9 (36%) | 79.5 (49%) | 75.0 (36%) | 72.8 (60%) | 67.2 (55%) |
| G.711 μ-law | **86.7** (23%) | 80.7 (39%) | 79.9 (39%) | 78.3 (51%) | 71.8 (49%) | 69.4 (73%) | 54.4 (40%) |
| AMR-NB 7.95 | **86.9** (25%) | 80.5 (41%) | 77.5 (43%) | 78.4 (49%) | 70.6 (57%) | 68.4 (77%) | 60.2 (62%) |
| noise 10 dB + AMR 7.95 | **83.4** (33%) | 76.3 (52%) | 71.1 (47%) | 76.8 (45%) | 67.7 (66%) | 65.2 (84%) | 64.1 (92%) |
| mixed chain | **82.8** (31%) | 75.1 (54%) | 70.6 (41%) | 76.8 (47%) | 68.6 (59%) | 65.7 (79%) | 62.9 (90%) |
| AUC clean / mixed | **0.93 / 0.87** | 0.82 / 0.75 | 0.81 / 0.73 | 0.86 / 0.80 | 0.79 / 0.67 | 0.77 / 0.60 | 0.67 / 0.52 |

- **eot_v1 still leads every model in every condition**, by 6.0 to 7.2 points over the next best. That is codewithmoin on clean
  and codec-only audio, and abhishek under noise and on the mixed chain.
- **abhishek is the surprise.** It scores 68.5% on Smart Turn's TTS Hindi, but it is third here (79.5% clean). Its AUC is
  second only to eot_v1's (0.86 clean), and it is the most noise-robust community model (−2.7 points under noise, against
  −5 to −10 for the others).
- **deveshu**, the strongest community model on TTS Hindi (93.2%), drops to 80.9% on real voices and loses 10 points under noise.
- **Smart Turn's int8 file**, the one Pipecat ships, cuts in far more than its own fp32 file on real voices: 60% vs 36% clean,
  and 73% vs 49% on G.711. Accuracy differs by only 2-3 points; the int8 file shifts probabilities up, toward "finished".
- **chhotu** fails on phone audio (54% on G.711; under noise it calls 92% of pauses finished). Its card allows research and
  evaluation use only.
- No significance tests here: the eot_v1 predictions are on the GPU box. The smallest lead over the next best model is 6.0
  points on 1,046 paired clips. Against codewithmoin, leads of 7.2-10.4 points gave McNemar p ≤ 3e-8.
  *Since done (GPU box, Job 6):* `week1_fleurs_significance.md` tests eot_v1 against codewithmoin, deveshu and abhishek in all
  five conditions, all on one machine's audio: every one of the 15 tests gives p ≤ 4e-5.

---

# End-of-turn baselines: clean vs phone line

Data: `data/external/fleurs-hi-eot`: FLEURS Hindi read speech by human speakers (google/fleurs, CC BY 4.0), cut at the sentence end (finished) or 0.2 s into a mid-sentence pause (not finished).
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Clean audio (model-card accuracy is for Smart Turn's test set, so no match check)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| deveshu_fp32 | hin | 1046 | 80.88 | - | n/a | n/a |
| chhotu_int8 | hin | 1046 | 67.21 | - | n/a | n/a |
| abhishek_tiny_attn | hin | 1046 | 79.45 | - | n/a | n/a |
| smart_turn_v3.2_fp32 | hin | 1046 | 74.95 | - | n/a | n/a |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `g711_mulaw`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_7.95`: Band-limited to 8 kHz, then this codec; nothing else
- `noise10_amr7.95`: MUSAN noise at 10 dB SNR, AMR-NB 7.95
- `mixed`: One random chain per clip from configs/augment.yaml (no speakerphone echo: the clips have no agent track)
- Noise: the 181 held-out MUSAN noise files (CC BY 4.0; training uses the other 80%), checked against `data/manifests/noise_musan.jsonl`

| Model | Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|---|
| abhishek_tiny_attn | hin | clean | 1046 | 79.45 | +0.00 | 0.488 | 0.038 | 0.858 |
| abhishek_tiny_attn | hin | g711_mulaw | 1046 | 78.30 | -1.15 | 0.509 | 0.044 | 0.848 |
| abhishek_tiny_attn | hin | amrnb_7.95 | 1046 | 78.39 | -1.05 | 0.491 | 0.053 | 0.831 |
| abhishek_tiny_attn | hin | noise10_amr7.95 | 1046 | 76.77 | -2.68 | 0.445 | 0.107 | 0.778 |
| abhishek_tiny_attn | hin | mixed | 1046 | 76.77 | -2.68 | 0.473 | 0.090 | 0.800 |
| chhotu_int8 | hin | clean | 1046 | 67.21 | +0.00 | 0.553 | 0.195 | 0.674 |
| chhotu_int8 | hin | g711_mulaw | 1046 | 54.40 | -12.81 | 0.398 | 0.490 | 0.587 |
| chhotu_int8 | hin | amrnb_7.95 | 1046 | 60.23 | -6.98 | 0.620 | 0.266 | 0.542 |
| chhotu_int8 | hin | noise10_amr7.95 | 1046 | 64.05 | -3.15 | 0.920 | 0.027 | 0.447 |
| chhotu_int8 | hin | mixed | 1046 | 62.91 | -4.30 | 0.900 | 0.058 | 0.518 |
| deveshu_fp32 | hin | clean | 1046 | 80.88 | +0.00 | 0.360 | 0.091 | 0.806 |
| deveshu_fp32 | hin | g711_mulaw | 1046 | 79.92 | -0.96 | 0.386 | 0.091 | 0.797 |
| deveshu_fp32 | hin | amrnb_7.95 | 1046 | 77.53 | -3.35 | 0.429 | 0.104 | 0.779 |
| deveshu_fp32 | hin | noise10_amr7.95 | 1046 | 71.13 | -9.75 | 0.465 | 0.184 | 0.686 |
| deveshu_fp32 | hin | mixed | 1046 | 70.55 | -10.33 | 0.406 | 0.228 | 0.729 |
| smart_turn_v3.2_fp32 | hin | clean | 1046 | 74.95 | +0.00 | 0.360 | 0.186 | 0.787 |
| smart_turn_v3.2_fp32 | hin | g711_mulaw | 1046 | 71.80 | -3.15 | 0.491 | 0.158 | 0.737 |
| smart_turn_v3.2_fp32 | hin | amrnb_7.95 | 1046 | 70.55 | -4.40 | 0.566 | 0.134 | 0.699 |
| smart_turn_v3.2_fp32 | hin | noise10_amr7.95 | 1046 | 67.69 | -7.27 | 0.661 | 0.123 | 0.661 |
| smart_turn_v3.2_fp32 | hin | mixed | 1046 | 68.55 | -6.41 | 0.589 | 0.152 | 0.665 |