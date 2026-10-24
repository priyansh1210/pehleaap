# End-of-turn baselines: clean vs phone line

Data: pipecat-ai/smart-turn-data-v3.2-test (English + Hindi; **all Hindi clips are synthetic TTS**).
Community models were trained on smart-turn-data-v3.2-train, so this test is in-distribution for them.
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Wrapper validation (clean; card numbers are for the full set, so only a full run can match)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | eng | 7820 | 94.26 | 94.26 | -0.00 | yes |
| eot_v0 | eng | 7820 | 95.40 | - | n/a | n/a |
| smart_turn_v3.2_int8 | hin | 1284 | 90.26 | 90.11 | +0.15 | yes |
| eot_v0 | hin | 1284 | 94.08 | - | n/a | n/a |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `g711_mulaw`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_7.95`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_4.75`: Band-limited to 8 kHz, then this codec; nothing else
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
| eot_v0 | eng | clean | 1947 | 94.30 | +0.00 | 0.066 | 0.047 | 0.987 |
| eot_v0 | eng | g711_mulaw | 1947 | 93.53 | -0.77 | 0.075 | 0.055 | 0.984 |
| eot_v0 | eng | amrnb_7.95 | 1947 | 93.53 | -0.77 | 0.076 | 0.053 | 0.983 |
| eot_v0 | eng | amrnb_4.75 | 1947 | 92.45 | -1.85 | 0.083 | 0.068 | 0.981 |
| eot_v0 | eng | tandem_amr4.75 | 1947 | 92.76 | -1.54 | 0.094 | 0.050 | 0.976 |
| eot_v0 | eng | loss3_g711 | 1947 | 93.58 | -0.72 | 0.081 | 0.047 | 0.982 |
| eot_v0 | eng | agc_clip_g711 | 1947 | 93.43 | -0.87 | 0.082 | 0.049 | 0.981 |
| eot_v0 | eng | noise20_amr7.95 | 1947 | 92.24 | -2.05 | 0.096 | 0.059 | 0.978 |
| eot_v0 | eng | noise10_amr7.95 | 1947 | 92.14 | -2.16 | 0.102 | 0.055 | 0.973 |
| eot_v0 | eng | noise5_amr7.95 | 1947 | 90.60 | -3.70 | 0.112 | 0.075 | 0.967 |
| eot_v0 | eng | mixed | 1947 | 92.50 | -1.80 | 0.088 | 0.062 | 0.975 |
| eot_v0 | hin | clean | 1284 | 94.08 | +0.00 | 0.060 | 0.058 | 0.987 |
| eot_v0 | hin | g711_mulaw | 1284 | 94.39 | +0.31 | 0.060 | 0.052 | 0.986 |
| eot_v0 | hin | amrnb_7.95 | 1284 | 93.22 | -0.86 | 0.080 | 0.057 | 0.985 |
| eot_v0 | hin | amrnb_4.75 | 1284 | 93.30 | -0.78 | 0.073 | 0.061 | 0.982 |
| eot_v0 | hin | tandem_amr4.75 | 1284 | 93.15 | -0.93 | 0.075 | 0.063 | 0.984 |
| eot_v0 | hin | loss3_g711 | 1284 | 94.24 | +0.16 | 0.055 | 0.060 | 0.986 |
| eot_v0 | hin | agc_clip_g711 | 1284 | 94.70 | +0.62 | 0.059 | 0.048 | 0.987 |
| eot_v0 | hin | noise20_amr7.95 | 1284 | 93.61 | -0.47 | 0.083 | 0.046 | 0.985 |
| eot_v0 | hin | noise10_amr7.95 | 1284 | 91.20 | -2.88 | 0.104 | 0.073 | 0.976 |
| eot_v0 | hin | noise5_amr7.95 | 1284 | 88.63 | -5.45 | 0.146 | 0.084 | 0.964 |
| eot_v0 | hin | mixed | 1284 | 91.90 | -2.18 | 0.093 | 0.070 | 0.976 |
| smart_turn_v3.2_int8 | eng | clean | 1947 | 92.96 | +0.00 | 0.094 | 0.046 | 0.979 |
| smart_turn_v3.2_int8 | eng | g711_mulaw | 1947 | 92.14 | -0.82 | 0.109 | 0.047 | 0.974 |
| smart_turn_v3.2_int8 | eng | amrnb_7.95 | 1947 | 91.73 | -1.23 | 0.123 | 0.041 | 0.971 |
| smart_turn_v3.2_int8 | eng | amrnb_4.75 | 1947 | 90.76 | -2.21 | 0.137 | 0.046 | 0.969 |
| smart_turn_v3.2_int8 | eng | tandem_amr4.75 | 1947 | 89.47 | -3.49 | 0.169 | 0.039 | 0.953 |
| smart_turn_v3.2_int8 | eng | loss3_g711 | 1947 | 89.52 | -3.44 | 0.161 | 0.046 | 0.959 |
| smart_turn_v3.2_int8 | eng | agc_clip_g711 | 1947 | 89.37 | -3.60 | 0.164 | 0.046 | 0.957 |
| smart_turn_v3.2_int8 | eng | noise20_amr7.95 | 1947 | 88.14 | -4.83 | 0.193 | 0.041 | 0.952 |
| smart_turn_v3.2_int8 | eng | noise10_amr7.95 | 1947 | 85.36 | -7.60 | 0.243 | 0.046 | 0.936 |
| smart_turn_v3.2_int8 | eng | noise5_amr7.95 | 1947 | 81.66 | -11.30 | 0.316 | 0.045 | 0.909 |
| smart_turn_v3.2_int8 | eng | mixed | 1947 | 84.28 | -8.68 | 0.274 | 0.036 | 0.930 |
| smart_turn_v3.2_int8 | hin | clean | 1284 | 90.26 | +0.00 | 0.177 | 0.024 | 0.982 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1284 | 86.37 | -3.89 | 0.275 | 0.009 | 0.971 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1284 | 82.79 | -7.48 | 0.353 | 0.006 | 0.962 |
| smart_turn_v3.2_int8 | hin | amrnb_4.75 | 1284 | 80.53 | -9.74 | 0.400 | 0.006 | 0.951 |
| smart_turn_v3.2_int8 | hin | tandem_amr4.75 | 1284 | 76.40 | -13.86 | 0.491 | 0.001 | 0.932 |
| smart_turn_v3.2_int8 | hin | loss3_g711 | 1284 | 82.09 | -8.18 | 0.358 | 0.015 | 0.956 |
| smart_turn_v3.2_int8 | hin | agc_clip_g711 | 1284 | 82.63 | -7.63 | 0.351 | 0.010 | 0.961 |
| smart_turn_v3.2_int8 | hin | noise20_amr7.95 | 1284 | 77.41 | -12.85 | 0.465 | 0.006 | 0.929 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1284 | 73.68 | -16.59 | 0.548 | 0.001 | 0.897 |
| smart_turn_v3.2_int8 | hin | noise5_amr7.95 | 1284 | 69.55 | -20.72 | 0.626 | 0.009 | 0.858 |
| smart_turn_v3.2_int8 | hin | mixed | 1284 | 73.52 | -16.74 | 0.548 | 0.004 | 0.895 |