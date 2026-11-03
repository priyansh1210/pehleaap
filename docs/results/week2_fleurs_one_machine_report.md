# Every FLEURS cell from one machine (GPU box, Nov 3 2026)

**Why.** The README's real-voice table mixed two machines: eot_v1, codewithmoin and Smart Turn int8 were scored on the
GPU box (Job 5), and deveshu, abhishek, Smart Turn fp32 and chhotu on the Laptop. The "mixed phone chain" audio is not
bit-identical across machines, so that row could be off by up to about 0.4 points. Here all 7 models run in all 5
conditions in one run on the GPU box.

**Result.** 27 of the 35 cells are unchanged at full precision. eot_v1's lead and every AUC in the README stay the same.
- **Same machine, same numbers.** The three Job 5 columns (eot_v1, codewithmoin, Smart Turn int8) repeat exactly in
  every condition.
- **The fp32 models the Laptop scored** (deveshu, abhishek, Smart Turn fp32) match it exactly in the 4 fixed conditions
  and differ only on the mixed chain, by 0 to 3 clips: deveshu 70.55 -> 70.36, Smart Turn fp32 68.55 -> 68.83,
  abhishek's cut-in rate 47.3% -> 47.6%. These GPU-box mixed values are the ones the significance tests
  (`week1_fleurs_significance.md`) already used. The cause is the ffmpeg build: 9.0.1 here and 9.0.2 on the Laptop encode
  Opus differently, and with 9.0.2 this machine reproduces the Laptop's mixed numbers exactly (`week2_ffmpeg_opus.md`).
- **chhotu, the one int8 model scored on both machines, differs in every condition, clean audio included**
  (67.21 -> 67.30 clean, 54.40 -> 54.02 G.711, 60.23 -> 59.85 AMR-NB, 64.05 -> 63.96 noise, 62.91 -> 63.10 mixed).
  Clean audio is identical on both machines, so the int8 inference itself differs between the two CPUs (i5-11300H on the
  Laptop, Core Ultra 9 285 here). The likely cause is onnxruntime picking different int8 kernels for each CPU's
  instruction set (not checked). So int8 scores, ours included, are comparable only within one machine.

Command (GPU box, `--threads 20`, about 20 min):

```
.venv\Scripts\python scripts\eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin
    --models smart_turn_v3.2_int8 deveshu_fp32 abhishek_tiny_attn codewithmoin_8s chhotu_int8
    --onnx eot_v1=weights\eot_v1\model.onnx smart_turn_v3.2_fp32=weights\pipecat-ai__smart-turn-v3\smart-turn-v3.2-gpu.onnx
    --conditions clean g711_mulaw amrnb_7.95 noise10_amr7.95 mixed --threads 20 --out fleurs_mixed_one_machine
```

The full report follows.

---

# End-of-turn baselines: clean vs phone line

Data: `data/external/fleurs-hi-eot`: FLEURS Hindi read speech by human speakers (google/fleurs, CC BY 4.0), cut at the sentence end (finished) or 0.2 s into a mid-sentence pause (not finished).
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Clean audio (model-card accuracy is for Smart Turn's test set, so no match check)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | hin | 1046 | 72.75 | - | n/a | n/a |
| deveshu_fp32 | hin | 1046 | 80.88 | - | n/a | n/a |
| abhishek_tiny_attn | hin | 1046 | 79.45 | - | n/a | n/a |
| codewithmoin_8s | hin | 1046 | 81.55 | - | n/a | n/a |
| chhotu_int8 | hin | 1046 | 67.30 | - | n/a | n/a |
| eot_v1 | hin | 1046 | 88.81 | - | n/a | n/a |
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
| abhishek_tiny_attn | hin | mixed | 1046 | 76.77 | -2.68 | 0.476 | 0.088 | 0.800 |
| chhotu_int8 | hin | clean | 1046 | 67.30 | +0.00 | 0.553 | 0.193 | 0.674 |
| chhotu_int8 | hin | g711_mulaw | 1046 | 54.02 | -13.29 | 0.401 | 0.495 | 0.588 |
| chhotu_int8 | hin | amrnb_7.95 | 1046 | 59.85 | -7.46 | 0.622 | 0.271 | 0.542 |
| chhotu_int8 | hin | noise10_amr7.95 | 1046 | 63.96 | -3.35 | 0.923 | 0.027 | 0.446 |
| chhotu_int8 | hin | mixed | 1046 | 63.10 | -4.21 | 0.900 | 0.055 | 0.520 |
| codewithmoin_8s | hin | clean | 1046 | 81.55 | +0.00 | 0.370 | 0.075 | 0.821 |
| codewithmoin_8s | hin | g711_mulaw | 1046 | 80.69 | -0.86 | 0.391 | 0.076 | 0.814 |
| codewithmoin_8s | hin | amrnb_7.95 | 1046 | 80.50 | -1.05 | 0.411 | 0.067 | 0.809 |
| codewithmoin_8s | hin | noise10_amr7.95 | 1046 | 76.29 | -5.26 | 0.524 | 0.067 | 0.754 |
| codewithmoin_8s | hin | mixed | 1046 | 75.05 | -6.50 | 0.540 | 0.078 | 0.754 |
| deveshu_fp32 | hin | clean | 1046 | 80.88 | +0.00 | 0.360 | 0.091 | 0.806 |
| deveshu_fp32 | hin | g711_mulaw | 1046 | 79.92 | -0.96 | 0.386 | 0.091 | 0.797 |
| deveshu_fp32 | hin | amrnb_7.95 | 1046 | 77.53 | -3.35 | 0.429 | 0.104 | 0.779 |
| deveshu_fp32 | hin | noise10_amr7.95 | 1046 | 71.13 | -9.75 | 0.465 | 0.184 | 0.686 |
| deveshu_fp32 | hin | mixed | 1046 | 70.36 | -10.52 | 0.406 | 0.231 | 0.727 |
| eot_v1 | hin | clean | 1046 | 88.81 | +0.00 | 0.183 | 0.070 | 0.926 |
| eot_v1 | hin | g711_mulaw | 1046 | 86.71 | -2.10 | 0.231 | 0.075 | 0.914 |
| eot_v1 | hin | amrnb_7.95 | 1046 | 86.90 | -1.91 | 0.254 | 0.058 | 0.907 |
| eot_v1 | hin | noise10_amr7.95 | 1046 | 83.37 | -5.45 | 0.334 | 0.067 | 0.880 |
| eot_v1 | hin | mixed | 1046 | 82.79 | -6.02 | 0.314 | 0.088 | 0.873 |
| smart_turn_v3.2_fp32 | hin | clean | 1046 | 74.95 | +0.00 | 0.360 | 0.186 | 0.787 |
| smart_turn_v3.2_fp32 | hin | g711_mulaw | 1046 | 71.80 | -3.15 | 0.491 | 0.158 | 0.737 |
| smart_turn_v3.2_fp32 | hin | amrnb_7.95 | 1046 | 70.55 | -4.40 | 0.566 | 0.134 | 0.699 |
| smart_turn_v3.2_fp32 | hin | noise10_amr7.95 | 1046 | 67.69 | -7.27 | 0.661 | 0.123 | 0.661 |
| smart_turn_v3.2_fp32 | hin | mixed | 1046 | 68.83 | -6.12 | 0.584 | 0.151 | 0.668 |
| smart_turn_v3.2_int8 | hin | clean | 1046 | 72.75 | +0.00 | 0.604 | 0.076 | 0.774 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1046 | 69.41 | -3.35 | 0.728 | 0.056 | 0.728 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1046 | 68.36 | -4.40 | 0.766 | 0.050 | 0.682 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1046 | 65.20 | -7.55 | 0.843 | 0.055 | 0.589 |
| smart_turn_v3.2_int8 | hin | mixed | 1046 | 65.68 | -7.07 | 0.794 | 0.076 | 0.602 |