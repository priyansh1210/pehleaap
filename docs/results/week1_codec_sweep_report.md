# End-of-turn baselines: wider codec sweep (GPU box, Oct 24 2026)

Follow-up to `week1_clip_baselines_report.md` (Job 1: G.711 μ-law and AMR-NB 7.95). Job 2 adds five more codecs from
`pehleaap/telephony/codecs.py` on the same clips and models: G.711 A-law, AMR-NB 4.75 and 12.2 kbps, GSM-FR and Opus 8k.
Command: `scripts/eval_clip_baselines.py --conditions clean g711_alaw amrnb_4.75 amrnb_12.2 gsm_fr opus_8k --threads 24
--out clip_baselines_codecs` (96 min, CPU only). Clean rows match Job 1 exactly.

Scope: narrowband 8 kHz + codec only, no noise, echo or packet loss. All Hindi clips are synthetic TTS, and this is the
models' own test distribution.

## Summary: Smart Turn v3.2, Hindi (n = 1284), all 7 phone conditions

| Condition | Acc (card thr) | Δ vs clean | Cut-in rate | AUC | Best-threshold acc* |
|---|---|---|---|---|---|
| clean | 90.26 | +0.00 | 17.7% | 0.982 | 92.83 |
| G.711 A-law | 86.21 | -4.05 | 27.8% | 0.973 | 90.26 |
| G.711 μ-law (Job 1) | 86.37 | -3.89 | 27.5% | 0.971 | 90.26 |
| Opus 8k | 84.03 | -6.23 | 32.2% | 0.969 | 89.95 |
| GSM-FR | 83.88 | -6.39 | 32.8% | 0.968 | 89.80 |
| AMR-NB 12.2 | 83.10 | -7.17 | 34.3% | 0.968 | 89.95 |
| AMR-NB 7.95 (Job 1) | 82.79 | -7.48 | 35.3% | 0.962 | 88.86 |
| AMR-NB 4.75 | 80.53 | -9.74 | 40.0% | 0.951 | 87.46 |

\* Accuracy at the single best threshold, chosen on these same clips (an optimistic upper bound, not a fair score).
Best thresholds are 0.91-0.95 for Hindi in every condition, against the card's 0.5.

Findings:
- **AMR-NB 4.75 is the worst case.** Smart Turn loses 9.7 points on Hindi and calls 40% of unfinished turns finished.
  Every non-G.711 codec costs 6-7.5 points. Bitrate matters little within AMR (12.2 and 7.95 are close), so the codec's
  processing hurts, not only the bitrate.
- **Most of the drop is a calibration shift, not lost information.** AUC only falls 0.982 → 0.951, and the mean score
  on unfinished Hindi turns rises 0.18 → 0.40. A higher threshold would recover about 7 of the 10 points on AMR 4.75. Even
  clean Hindi is under-thresholded at 0.5. This argues for per-language and per-channel calibration in our recipe, and
  for reporting AUC alongside accuracy.
- **English barely moves** (best case -0.7, worst -2.2 points, AMR 4.75), and the best English threshold stays near 0.5-0.7.
- **Community models:** deveshu and codewithmoin hold up again (worst -2.7 points, AMR 4.75 Hindi). chhotu collapses in every
  narrowband condition (-23 to -27 points). abhishek stays weak and flat.

---

## Full run report

Data: pipecat-ai/smart-turn-data-v3.2-test (English + Hindi; **all Hindi clips are synthetic TTS**).
Community models were trained on smart-turn-data-v3.2-train, so this test is in-distribution for them.
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Wrapper validation (clean; card numbers are for the full set, so only a full run can match)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | eng | 7820 | 94.26 | 94.26 | -0.00 | yes |
| deveshu_fp32 | eng | 7820 | 93.86 | 93.8 | +0.06 | yes |
| abhishek_tiny_attn | eng | 7820 | 60.23 | - | n/a | n/a |
| codewithmoin_8s | eng | 7820 | 93.67 | 93.7 | -0.03 | yes |
| chhotu_int8 | eng | 7820 | 88.35 | 88.4 | -0.05 | yes |
| smart_turn_v3.2_int8 | hin | 1284 | 90.26 | 90.11 | +0.15 | yes |
| deveshu_fp32 | hin | 1284 | 93.22 | 93.2 | +0.02 | yes |
| abhishek_tiny_attn | hin | 1284 | 68.46 | - | n/a | n/a |
| codewithmoin_8s | hin | 1284 | 93.93 | 93.9 | +0.03 | yes |
| chhotu_int8 | hin | 1284 | 88.79 | 88.8 | -0.01 | yes |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

Narrowband 8 kHz + codec only (no noise, echo or packet loss yet).

| Model | Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|---|
| abhishek_tiny_attn | eng | clean | 1947 | 60.91 | +0.00 | 0.490 | 0.287 | 0.665 |
| abhishek_tiny_attn | eng | g711_alaw | 1947 | 60.45 | -0.46 | 0.496 | 0.290 | 0.658 |
| abhishek_tiny_attn | eng | amrnb_4.75 | 1947 | 58.45 | -2.47 | 0.538 | 0.288 | 0.647 |
| abhishek_tiny_attn | eng | amrnb_12.2 | 1947 | 59.58 | -1.34 | 0.514 | 0.290 | 0.653 |
| abhishek_tiny_attn | eng | gsm_fr | 1947 | 60.09 | -0.82 | 0.510 | 0.284 | 0.653 |
| abhishek_tiny_attn | eng | opus_8k | 1947 | 60.35 | -0.56 | 0.503 | 0.286 | 0.655 |
| abhishek_tiny_attn | hin | clean | 1284 | 68.46 | +0.00 | 0.403 | 0.235 | 0.782 |
| abhishek_tiny_attn | hin | g711_alaw | 1284 | 68.77 | +0.31 | 0.398 | 0.233 | 0.775 |
| abhishek_tiny_attn | hin | amrnb_4.75 | 1284 | 67.29 | -1.17 | 0.361 | 0.296 | 0.748 |
| abhishek_tiny_attn | hin | amrnb_12.2 | 1284 | 67.60 | -0.86 | 0.389 | 0.265 | 0.765 |
| abhishek_tiny_attn | hin | gsm_fr | 1284 | 68.22 | -0.23 | 0.389 | 0.253 | 0.763 |
| abhishek_tiny_attn | hin | opus_8k | 1284 | 67.45 | -1.01 | 0.400 | 0.257 | 0.766 |
| chhotu_int8 | eng | clean | 1947 | 88.70 | +0.00 | 0.157 | 0.067 | 0.951 |
| chhotu_int8 | eng | g711_alaw | 1947 | 65.79 | -22.91 | 0.436 | 0.244 | 0.713 |
| chhotu_int8 | eng | amrnb_4.75 | 1947 | 62.04 | -26.66 | 0.656 | 0.092 | 0.723 |
| chhotu_int8 | eng | amrnb_12.2 | 1947 | 64.66 | -24.04 | 0.530 | 0.170 | 0.711 |
| chhotu_int8 | eng | gsm_fr | 1947 | 63.64 | -25.06 | 0.550 | 0.170 | 0.705 |
| chhotu_int8 | eng | opus_8k | 1947 | 65.90 | -22.80 | 0.455 | 0.222 | 0.716 |
| chhotu_int8 | hin | clean | 1284 | 88.79 | +0.00 | 0.148 | 0.079 | 0.961 |
| chhotu_int8 | hin | g711_alaw | 1284 | 63.16 | -25.62 | 0.509 | 0.239 | 0.669 |
| chhotu_int8 | hin | amrnb_4.75 | 1284 | 63.08 | -25.70 | 0.678 | 0.085 | 0.685 |
| chhotu_int8 | hin | amrnb_12.2 | 1284 | 61.37 | -27.41 | 0.615 | 0.176 | 0.660 |
| chhotu_int8 | hin | gsm_fr | 1284 | 63.63 | -25.16 | 0.571 | 0.173 | 0.670 |
| chhotu_int8 | hin | opus_8k | 1284 | 62.15 | -26.64 | 0.551 | 0.220 | 0.652 |
| codewithmoin_8s | eng | clean | 1947 | 93.07 | +0.00 | 0.089 | 0.049 | 0.978 |
| codewithmoin_8s | eng | g711_alaw | 1947 | 91.53 | -1.54 | 0.101 | 0.068 | 0.975 |
| codewithmoin_8s | eng | amrnb_4.75 | 1947 | 91.22 | -1.85 | 0.112 | 0.063 | 0.971 |
| codewithmoin_8s | eng | amrnb_12.2 | 1947 | 92.19 | -0.87 | 0.098 | 0.058 | 0.975 |
| codewithmoin_8s | eng | gsm_fr | 1947 | 91.22 | -1.85 | 0.110 | 0.065 | 0.973 |
| codewithmoin_8s | eng | opus_8k | 1947 | 91.73 | -1.34 | 0.103 | 0.062 | 0.975 |
| codewithmoin_8s | hin | clean | 1284 | 93.93 | +0.00 | 0.067 | 0.055 | 0.985 |
| codewithmoin_8s | hin | g711_alaw | 1284 | 93.22 | -0.70 | 0.076 | 0.060 | 0.984 |
| codewithmoin_8s | hin | amrnb_4.75 | 1284 | 91.82 | -2.10 | 0.081 | 0.082 | 0.978 |
| codewithmoin_8s | hin | amrnb_12.2 | 1284 | 92.99 | -0.93 | 0.080 | 0.061 | 0.982 |
| codewithmoin_8s | hin | gsm_fr | 1284 | 93.07 | -0.86 | 0.070 | 0.069 | 0.982 |
| codewithmoin_8s | hin | opus_8k | 1284 | 92.76 | -1.17 | 0.076 | 0.069 | 0.982 |
| deveshu_fp32 | eng | clean | 1947 | 93.32 | +0.00 | 0.079 | 0.055 | 0.980 |
| deveshu_fp32 | eng | g711_alaw | 1947 | 92.71 | -0.62 | 0.085 | 0.061 | 0.977 |
| deveshu_fp32 | eng | amrnb_4.75 | 1947 | 91.11 | -2.21 | 0.104 | 0.073 | 0.970 |
| deveshu_fp32 | eng | amrnb_12.2 | 1947 | 92.24 | -1.08 | 0.094 | 0.061 | 0.975 |
| deveshu_fp32 | eng | gsm_fr | 1947 | 92.14 | -1.18 | 0.092 | 0.065 | 0.974 |
| deveshu_fp32 | eng | opus_8k | 1947 | 92.40 | -0.92 | 0.083 | 0.069 | 0.976 |
| deveshu_fp32 | hin | clean | 1284 | 93.22 | +0.00 | 0.059 | 0.076 | 0.984 |
| deveshu_fp32 | hin | g711_alaw | 1284 | 92.68 | -0.55 | 0.078 | 0.069 | 0.981 |
| deveshu_fp32 | hin | amrnb_4.75 | 1284 | 90.58 | -2.65 | 0.078 | 0.109 | 0.972 |
| deveshu_fp32 | hin | amrnb_12.2 | 1284 | 92.13 | -1.09 | 0.080 | 0.078 | 0.978 |
| deveshu_fp32 | hin | gsm_fr | 1284 | 91.59 | -1.64 | 0.081 | 0.087 | 0.979 |
| deveshu_fp32 | hin | opus_8k | 1284 | 92.45 | -0.78 | 0.075 | 0.076 | 0.979 |
| smart_turn_v3.2_int8 | eng | clean | 1947 | 92.96 | +0.00 | 0.094 | 0.046 | 0.979 |
| smart_turn_v3.2_int8 | eng | g711_alaw | 1947 | 92.24 | -0.72 | 0.111 | 0.043 | 0.973 |
| smart_turn_v3.2_int8 | eng | amrnb_4.75 | 1947 | 90.76 | -2.21 | 0.137 | 0.046 | 0.969 |
| smart_turn_v3.2_int8 | eng | amrnb_12.2 | 1947 | 92.14 | -0.82 | 0.117 | 0.039 | 0.972 |
| smart_turn_v3.2_int8 | eng | gsm_fr | 1947 | 91.22 | -1.75 | 0.134 | 0.040 | 0.970 |
| smart_turn_v3.2_int8 | eng | opus_8k | 1947 | 92.19 | -0.77 | 0.113 | 0.042 | 0.974 |
| smart_turn_v3.2_int8 | hin | clean | 1284 | 90.26 | +0.00 | 0.177 | 0.024 | 0.982 |
| smart_turn_v3.2_int8 | hin | g711_alaw | 1284 | 86.21 | -4.05 | 0.278 | 0.009 | 0.973 |
| smart_turn_v3.2_int8 | hin | amrnb_4.75 | 1284 | 80.53 | -9.74 | 0.400 | 0.006 | 0.951 |
| smart_turn_v3.2_int8 | hin | amrnb_12.2 | 1284 | 83.10 | -7.17 | 0.343 | 0.009 | 0.968 |
| smart_turn_v3.2_int8 | hin | gsm_fr | 1284 | 83.88 | -6.39 | 0.328 | 0.007 | 0.968 |
| smart_turn_v3.2_int8 | hin | opus_8k | 1284 | 84.03 | -6.23 | 0.322 | 0.010 | 0.969 |