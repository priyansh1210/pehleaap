# End-of-turn baselines: realistic phone-line chains (GPU box, Oct 24-25 2026)

Job 3. Follow-up to `week1_codec_sweep_report.md` (codec only). This run adds the impairments real calls stack on top of
the codec: transcoding, packet loss, AGC with clipping, background noise and a random mixed chain. It covers the five
community baselines and LiveKit turn-detector v1-mini, on the same clips.

- Baselines: `scripts/eval_clip_baselines.py --conditions clean tandem_amr4.75 loss3_g711 agc_clip_g711 noise20_amr7.95
  noise10_amr7.95 noise5_amr7.95 mixed --threads 24 --out clip_baselines_chain` (about 3 h of CPU time, 19 min per shard;
  interrupted after shard 3 and resumed from the saved shards).
- LiveKit: `.venv-livekit\Scripts\python scripts/eval_livekit_v1mini.py` with the same `--conditions` and
  `--out livekit_v1mini_chain`. LiveKit's predictions are comparison numbers only (LiveKit Model License).
- Noise: the 181 held-out MUSAN noise files (CC BY 4.0). Training uses the other 80%. The files are checked against
  `data/manifests/noise_musan.jsonl`.
- Scope: all Hindi clips are synthetic TTS, and this is the community models' own test distribution. Clean rows match
  Jobs 1 and 2 exactly.

## Summary: Hindi (n = 1284), accuracy at each model's own threshold / AUC

| Condition | Smart Turn v3.2 | deveshu | codewithmoin | chhotu | abhishek | LiveKit v1-mini |
|---|---|---|---|---|---|---|
| clean | 90.26 / 0.982 | 93.22 / 0.984 | 93.93 / 0.985 | 88.79 / 0.961 | 68.46 / 0.782 | 55.14 / 0.708 |
| tandem AMR 4.75 → G.711 | 76.40 / 0.932 | 89.33 / 0.961 | 90.50 / 0.973 | 56.54 / 0.649 | 66.28 / 0.736 | 59.89 / 0.708 |
| G.711 + 3% loss | 82.09 / 0.956 | 90.26 / 0.973 | 92.29 / 0.981 | 55.53 / 0.614 | 66.51 / 0.758 | 61.14 / 0.700 |
| AGC + clip + G.711 | 82.63 / 0.961 | 91.12 / 0.977 | 92.60 / 0.982 | 54.91 / 0.600 | 66.43 / 0.760 | 60.51 / 0.674 |
| noise 20 dB + AMR 7.95 | 77.41 / 0.929 | 87.77 / 0.961 | 90.97 / 0.973 | 54.91 / 0.622 | 66.90 / 0.727 | 59.35 / 0.660 |
| noise 10 dB + AMR 7.95 | 73.68 / 0.897 | 82.55 / 0.931 | 89.25 / 0.960 | 54.98 / 0.598 | 63.08 / 0.691 | 57.17 / 0.624 |
| noise 5 dB + AMR 7.95 | 69.55 / 0.858 | 77.88 / 0.900 | 86.29 / 0.944 | 54.36 / 0.575 | 61.14 / 0.662 | 56.23 / 0.595 |
| mixed | 73.52 / 0.895 | 82.63 / 0.931 | 89.41 / 0.960 | 54.28 / 0.588 | 62.85 / 0.696 | 57.32 / 0.659 |

## Smart Turn v3.2 in detail: Hindi, chain conditions

| Condition | Acc (card thr) | Δ vs clean | Cut-in rate | AUC | Best-threshold acc* | Best thr | Mean score, unfinished turns |
|---|---|---|---|---|---|---|---|
| clean | 90.26 | +0.00 | 17.7% | 0.982 | 92.83 | 0.91 | 0.18 |
| tandem AMR 4.75 → G.711 | 76.40 | -13.86 | 49.1% | 0.932 | 85.36 | 0.96 | 0.47 |
| G.711 + 3% loss | 82.09 | -8.18 | 35.8% | 0.956 | 88.40 | 0.95 | 0.35 |
| AGC + clip + G.711 | 82.63 | -7.63 | 35.1% | 0.961 | 89.17 | 0.93 | 0.35 |
| noise 20 dB + AMR 7.95 | 77.41 | -12.85 | 46.5% | 0.929 | 84.89 | 0.95 | 0.46 |
| noise 10 dB + AMR 7.95 | 73.68 | -16.59 | 54.8% | 0.897 | 80.92 | 0.95 | 0.54 |
| noise 5 dB + AMR 7.95 | 69.55 | -20.72 | 62.6% | 0.858 | 78.35 | 0.95 | 0.60 |
| mixed | 73.52 | -16.74 | 54.8% | 0.895 | 81.23 | 0.92 | 0.53 |

\* Accuracy at the single best threshold, chosen on these same clips (an optimistic upper bound, not a fair score).

Findings:
- **Realistic chains hurt Smart Turn far more than codecs alone.** On Hindi it loses 14 points to tandem transcoding
  (codec-only worst case: 9.7, AMR 4.75), 17 on the mixed chain and 21 at 5 dB SNR. Cut-in rate reaches 55-63%: the bot
  would interrupt more than half of unfinished Hindi turns. English loses 3.4-11.3 points.
- **Noise is where real information is lost.** Under codec, loss and AGC chains AUC stays at 0.93-0.96, so most of the
  drop is still calibration (best threshold 0.93-0.96 against the card's 0.5). With noise, AUC falls to 0.86-0.90, and even
  the best threshold leaves Smart Turn 12-14 points below clean. Recalibration alone won't fix noisy calls. That needs
  training on noisy phone-line audio, which is exactly what Job 4 (eot_v0) does.
- **codewithmoin is the most robust model** (worst case -7.6 points and AUC 0.944, at 5 dB). deveshu is close on
  channel effects, but noise pushes its missed-end rate to 24-31%: it starts to wait too long instead of cutting in.
- **chhotu collapses in every chain** (-32 to -37 points, AUC 0.58-0.70), worse than on codecs alone. abhishek stays weak
  and roughly flat.
- **LiveKit v1-mini is weak here and degrades further.** Hindi AUC is 0.708 clean, 0.595 at 5 dB SNR and 0.659 on the
  mixed chain, with cut-in rates of 76-82%. Its accuracy "gains" over clean are a threshold artifact, as in the codec run
  (`week1_livekit_v1mini_report.md`). This test set is out of distribution for LiveKit, so this measures it on Smart
  Turn's data, not on its own.

---

## A1. Baselines: Wrapper validation (clean; card numbers are for the full set, so only a full run can match)

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

## A2. Baselines: Phone-line stress test (all Hindi + 25% hash sample of English)

- `tandem_amr4.75`: AMR-NB 4.75 kbit/s, then G.711 mu-law (transcoding at the operator's gateway)
- `loss3_g711`: G.711 mu-law, 3% of 20 ms packets lost in bursts of about 2, concealed
- `agc_clip_g711`: Phone AGC towards -16 dBFS (up to +20 dB), clipped at 0.25 of full scale, G.711 mu-law
- `noise20_amr7.95`: MUSAN noise at 20 dB SNR, AMR-NB 7.95
- `noise10_amr7.95`: MUSAN noise at 10 dB SNR, AMR-NB 7.95
- `noise5_amr7.95`: MUSAN noise at 5 dB SNR, AMR-NB 7.95
- `mixed`: One random chain per clip from configs/augment.yaml (no speakerphone echo: the clips have no agent track)
- Noise: the 181 held-out MUSAN noise files (CC BY 4.0; training uses the other 80%), checked against `data/manifests/noise_musan.jsonl`

| Model | Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|---|
| abhishek_tiny_attn | eng | clean | 1947 | 60.91 | +0.00 | 0.490 | 0.287 | 0.665 |
| abhishek_tiny_attn | eng | tandem_amr4.75 | 1947 | 59.32 | -1.59 | 0.517 | 0.292 | 0.644 |
| abhishek_tiny_attn | eng | loss3_g711 | 1947 | 60.50 | -0.41 | 0.476 | 0.310 | 0.650 |
| abhishek_tiny_attn | eng | agc_clip_g711 | 1947 | 59.68 | -1.23 | 0.499 | 0.303 | 0.644 |
| abhishek_tiny_attn | eng | noise20_amr7.95 | 1947 | 59.53 | -1.39 | 0.512 | 0.294 | 0.644 |
| abhishek_tiny_attn | eng | noise10_amr7.95 | 1947 | 59.84 | -1.08 | 0.501 | 0.299 | 0.640 |
| abhishek_tiny_attn | eng | noise5_amr7.95 | 1947 | 59.53 | -1.39 | 0.497 | 0.308 | 0.635 |
| abhishek_tiny_attn | eng | mixed | 1947 | 59.53 | -1.39 | 0.505 | 0.301 | 0.638 |
| abhishek_tiny_attn | hin | clean | 1284 | 68.46 | +0.00 | 0.403 | 0.235 | 0.782 |
| abhishek_tiny_attn | hin | tandem_amr4.75 | 1284 | 66.28 | -2.18 | 0.390 | 0.288 | 0.736 |
| abhishek_tiny_attn | hin | loss3_g711 | 1284 | 66.51 | -1.95 | 0.420 | 0.257 | 0.758 |
| abhishek_tiny_attn | hin | agc_clip_g711 | 1284 | 66.43 | -2.02 | 0.431 | 0.248 | 0.760 |
| abhishek_tiny_attn | hin | noise20_amr7.95 | 1284 | 66.90 | -1.56 | 0.363 | 0.302 | 0.727 |
| abhishek_tiny_attn | hin | noise10_amr7.95 | 1284 | 63.08 | -5.37 | 0.369 | 0.369 | 0.691 |
| abhishek_tiny_attn | hin | noise5_amr7.95 | 1284 | 61.14 | -7.32 | 0.343 | 0.430 | 0.662 |
| abhishek_tiny_attn | hin | mixed | 1284 | 62.85 | -5.61 | 0.403 | 0.342 | 0.696 |
| chhotu_int8 | eng | clean | 1947 | 88.70 | +0.00 | 0.157 | 0.067 | 0.951 |
| chhotu_int8 | eng | tandem_amr4.75 | 1947 | 56.81 | -31.90 | 0.818 | 0.030 | 0.702 |
| chhotu_int8 | eng | loss3_g711 | 1947 | 53.36 | -35.34 | 0.898 | 0.017 | 0.649 |
| chhotu_int8 | eng | agc_clip_g711 | 1947 | 52.03 | -36.67 | 0.913 | 0.028 | 0.634 |
| chhotu_int8 | eng | noise20_amr7.95 | 1947 | 52.29 | -36.41 | 0.925 | 0.010 | 0.656 |
| chhotu_int8 | eng | noise10_amr7.95 | 1947 | 51.62 | -37.08 | 0.931 | 0.019 | 0.622 |
| chhotu_int8 | eng | noise5_amr7.95 | 1947 | 51.72 | -36.98 | 0.905 | 0.043 | 0.594 |
| chhotu_int8 | eng | mixed | 1947 | 52.23 | -36.47 | 0.901 | 0.037 | 0.612 |
| chhotu_int8 | hin | clean | 1284 | 88.79 | +0.00 | 0.148 | 0.079 | 0.961 |
| chhotu_int8 | hin | tandem_amr4.75 | 1284 | 56.54 | -32.24 | 0.885 | 0.021 | 0.649 |
| chhotu_int8 | hin | loss3_g711 | 1284 | 55.53 | -33.26 | 0.907 | 0.019 | 0.614 |
| chhotu_int8 | hin | agc_clip_g711 | 1284 | 54.91 | -33.88 | 0.920 | 0.019 | 0.600 |
| chhotu_int8 | hin | noise20_amr7.95 | 1284 | 54.91 | -33.88 | 0.935 | 0.006 | 0.622 |
| chhotu_int8 | hin | noise10_amr7.95 | 1284 | 54.98 | -33.80 | 0.928 | 0.010 | 0.598 |
| chhotu_int8 | hin | noise5_amr7.95 | 1284 | 54.36 | -34.42 | 0.919 | 0.031 | 0.575 |
| chhotu_int8 | hin | mixed | 1284 | 54.28 | -34.50 | 0.935 | 0.018 | 0.588 |
| codewithmoin_8s | eng | clean | 1947 | 93.07 | +0.00 | 0.089 | 0.049 | 0.978 |
| codewithmoin_8s | eng | tandem_amr4.75 | 1947 | 90.09 | -2.98 | 0.130 | 0.067 | 0.965 |
| codewithmoin_8s | eng | loss3_g711 | 1947 | 91.27 | -1.80 | 0.108 | 0.066 | 0.970 |
| codewithmoin_8s | eng | agc_clip_g711 | 1947 | 91.47 | -1.59 | 0.104 | 0.066 | 0.970 |
| codewithmoin_8s | eng | noise20_amr7.95 | 1947 | 91.83 | -1.23 | 0.108 | 0.055 | 0.967 |
| codewithmoin_8s | eng | noise10_amr7.95 | 1947 | 89.52 | -3.54 | 0.139 | 0.069 | 0.957 |
| codewithmoin_8s | eng | noise5_amr7.95 | 1947 | 87.31 | -5.75 | 0.172 | 0.080 | 0.948 |
| codewithmoin_8s | eng | mixed | 1947 | 88.75 | -4.31 | 0.143 | 0.081 | 0.954 |
| codewithmoin_8s | hin | clean | 1284 | 93.93 | +0.00 | 0.067 | 0.055 | 0.985 |
| codewithmoin_8s | hin | tandem_amr4.75 | 1284 | 90.50 | -3.43 | 0.115 | 0.076 | 0.973 |
| codewithmoin_8s | hin | loss3_g711 | 1284 | 92.29 | -1.64 | 0.081 | 0.073 | 0.981 |
| codewithmoin_8s | hin | agc_clip_g711 | 1284 | 92.60 | -1.32 | 0.080 | 0.069 | 0.982 |
| codewithmoin_8s | hin | noise20_amr7.95 | 1284 | 90.97 | -2.96 | 0.102 | 0.079 | 0.973 |
| codewithmoin_8s | hin | noise10_amr7.95 | 1284 | 89.25 | -4.67 | 0.130 | 0.087 | 0.960 |
| codewithmoin_8s | hin | noise5_amr7.95 | 1284 | 86.29 | -7.63 | 0.205 | 0.075 | 0.944 |
| codewithmoin_8s | hin | mixed | 1284 | 89.41 | -4.52 | 0.132 | 0.082 | 0.960 |
| deveshu_fp32 | eng | clean | 1947 | 93.32 | +0.00 | 0.079 | 0.055 | 0.980 |
| deveshu_fp32 | eng | tandem_amr4.75 | 1947 | 90.24 | -3.08 | 0.112 | 0.083 | 0.958 |
| deveshu_fp32 | eng | loss3_g711 | 1947 | 90.96 | -2.36 | 0.093 | 0.088 | 0.964 |
| deveshu_fp32 | eng | agc_clip_g711 | 1947 | 90.76 | -2.57 | 0.100 | 0.085 | 0.963 |
| deveshu_fp32 | eng | noise20_amr7.95 | 1947 | 90.81 | -2.52 | 0.098 | 0.086 | 0.962 |
| deveshu_fp32 | eng | noise10_amr7.95 | 1947 | 88.55 | -4.78 | 0.117 | 0.112 | 0.952 |
| deveshu_fp32 | eng | noise5_amr7.95 | 1947 | 86.24 | -7.09 | 0.133 | 0.143 | 0.937 |
| deveshu_fp32 | eng | mixed | 1947 | 85.93 | -7.40 | 0.115 | 0.168 | 0.943 |
| deveshu_fp32 | hin | clean | 1284 | 93.22 | +0.00 | 0.059 | 0.076 | 0.984 |
| deveshu_fp32 | hin | tandem_amr4.75 | 1284 | 89.33 | -3.89 | 0.091 | 0.121 | 0.961 |
| deveshu_fp32 | hin | loss3_g711 | 1284 | 90.26 | -2.96 | 0.067 | 0.126 | 0.973 |
| deveshu_fp32 | hin | agc_clip_g711 | 1284 | 91.12 | -2.10 | 0.063 | 0.112 | 0.977 |
| deveshu_fp32 | hin | noise20_amr7.95 | 1284 | 87.77 | -5.45 | 0.099 | 0.143 | 0.961 |
| deveshu_fp32 | hin | noise10_amr7.95 | 1284 | 82.55 | -10.67 | 0.104 | 0.239 | 0.931 |
| deveshu_fp32 | hin | noise5_amr7.95 | 1284 | 77.88 | -15.34 | 0.120 | 0.314 | 0.900 |
| deveshu_fp32 | hin | mixed | 1284 | 82.63 | -10.59 | 0.093 | 0.248 | 0.931 |
| smart_turn_v3.2_int8 | eng | clean | 1947 | 92.96 | +0.00 | 0.094 | 0.046 | 0.979 |
| smart_turn_v3.2_int8 | eng | tandem_amr4.75 | 1947 | 89.47 | -3.49 | 0.169 | 0.039 | 0.953 |
| smart_turn_v3.2_int8 | eng | loss3_g711 | 1947 | 89.52 | -3.44 | 0.161 | 0.046 | 0.959 |
| smart_turn_v3.2_int8 | eng | agc_clip_g711 | 1947 | 89.37 | -3.60 | 0.164 | 0.046 | 0.957 |
| smart_turn_v3.2_int8 | eng | noise20_amr7.95 | 1947 | 88.14 | -4.83 | 0.193 | 0.041 | 0.952 |
| smart_turn_v3.2_int8 | eng | noise10_amr7.95 | 1947 | 85.36 | -7.60 | 0.243 | 0.046 | 0.936 |
| smart_turn_v3.2_int8 | eng | noise5_amr7.95 | 1947 | 81.66 | -11.30 | 0.316 | 0.045 | 0.909 |
| smart_turn_v3.2_int8 | eng | mixed | 1947 | 84.28 | -8.68 | 0.274 | 0.036 | 0.930 |
| smart_turn_v3.2_int8 | hin | clean | 1284 | 90.26 | +0.00 | 0.177 | 0.024 | 0.982 |
| smart_turn_v3.2_int8 | hin | tandem_amr4.75 | 1284 | 76.40 | -13.86 | 0.491 | 0.001 | 0.932 |
| smart_turn_v3.2_int8 | hin | loss3_g711 | 1284 | 82.09 | -8.18 | 0.358 | 0.015 | 0.956 |
| smart_turn_v3.2_int8 | hin | agc_clip_g711 | 1284 | 82.63 | -7.63 | 0.351 | 0.010 | 0.961 |
| smart_turn_v3.2_int8 | hin | noise20_amr7.95 | 1284 | 77.41 | -12.85 | 0.465 | 0.006 | 0.929 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1284 | 73.68 | -16.59 | 0.548 | 0.001 | 0.897 |
| smart_turn_v3.2_int8 | hin | noise5_amr7.95 | 1284 | 69.55 | -20.72 | 0.626 | 0.009 | 0.858 |
| smart_turn_v3.2_int8 | hin | mixed | 1284 | 73.52 | -16.74 | 0.548 | 0.004 | 0.895 |

## B. LiveKit v1-mini (thresholds hin 0.305, eng 0.36; finished when P(end of turn) is above the threshold)

### B1. Clean, all clips

| Lang | n | Acc | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|
| eng | 7820 | 56.45 | 0.830 | 0.025 | 0.712 |
| hin | 1284 | 55.14 | 0.937 | 0.000 | 0.708 |

### B2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `tandem_amr4.75`: AMR-NB 4.75 kbit/s, then G.711 mu-law (transcoding at the operator's gateway)
- `loss3_g711`: G.711 mu-law, 3% of 20 ms packets lost in bursts of about 2, concealed
- `agc_clip_g711`: Phone AGC towards -16 dBFS (up to +20 dB), clipped at 0.25 of full scale, G.711 mu-law
- `noise20_amr7.95`: MUSAN noise at 20 dB SNR, AMR-NB 7.95
- `noise10_amr7.95`: MUSAN noise at 10 dB SNR, AMR-NB 7.95
- `noise5_amr7.95`: MUSAN noise at 5 dB SNR, AMR-NB 7.95
- `mixed`: One random chain per clip from configs/augment.yaml (no speakerphone echo: the clips have no agent track)

| Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|
| eng | clean | 1947 | 56.24 | +0.00 | 0.832 | 0.027 | 0.725 |
| eng | tandem_amr4.75 | 1947 | 61.33 | +5.08 | 0.674 | 0.088 | 0.688 |
| eng | loss3_g711 | 1947 | 60.86 | +4.62 | 0.619 | 0.154 | 0.692 |
| eng | agc_clip_g711 | 1947 | 61.02 | +4.78 | 0.597 | 0.174 | 0.684 |
| eng | noise20_amr7.95 | 1947 | 63.12 | +6.88 | 0.544 | 0.187 | 0.694 |
| eng | noise10_amr7.95 | 1947 | 61.02 | +4.78 | 0.491 | 0.284 | 0.660 |
| eng | noise5_amr7.95 | 1947 | 58.86 | +2.62 | 0.477 | 0.343 | 0.622 |
| eng | mixed | 1947 | 57.83 | +1.59 | 0.606 | 0.230 | 0.632 |
| hin | clean | 1284 | 55.14 | +0.00 | 0.937 | 0.000 | 0.708 |
| hin | tandem_amr4.75 | 1284 | 59.89 | +4.75 | 0.808 | 0.027 | 0.708 |
| hin | loss3_g711 | 1284 | 61.14 | +6.00 | 0.759 | 0.048 | 0.700 |
| hin | agc_clip_g711 | 1284 | 60.51 | +5.37 | 0.787 | 0.034 | 0.674 |
| hin | noise20_amr7.95 | 1284 | 59.35 | +4.21 | 0.780 | 0.063 | 0.660 |
| hin | noise10_amr7.95 | 1284 | 57.17 | +2.02 | 0.793 | 0.093 | 0.624 |
| hin | noise5_amr7.95 | 1284 | 56.23 | +1.09 | 0.766 | 0.136 | 0.595 |
| hin | mixed | 1284 | 57.32 | +2.18 | 0.823 | 0.063 | 0.659 |
