# End-of-turn baselines: clean vs phone line

Data: pipecat-ai/smart-turn-data-v3.2-test (English + Hindi; **all Hindi clips are synthetic TTS**).
Community models were trained on smart-turn-data-v3.2-train, so this test is in-distribution for them.
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Wrapper validation (clean; card numbers are for the full set, so only a full run can match)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | eng | 7820 | 94.26 | 94.26 | -0.00 | yes |
| eot_v0 | eng | 7820 | 95.40 | - | n/a | n/a |
| eot_v1 | eng | 7820 | 94.69 | - | n/a | n/a |
| smart_turn_v3.2_int8 | hin | 1284 | 90.26 | 90.11 | +0.15 | yes |
| eot_v0 | hin | 1284 | 94.08 | - | n/a | n/a |
| eot_v1 | hin | 1284 | 93.69 | - | n/a | n/a |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `g711_mulaw`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_7.95`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_4.75`: Band-limited to 8 kHz, then this codec; nothing else
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
| eot_v0 | eng | noise10_amr7.95 | 1947 | 92.14 | -2.16 | 0.102 | 0.055 | 0.973 |
| eot_v0 | eng | noise5_amr7.95 | 1947 | 90.60 | -3.70 | 0.112 | 0.075 | 0.967 |
| eot_v0 | eng | mixed | 1947 | 92.50 | -1.80 | 0.088 | 0.062 | 0.975 |
| eot_v0 | hin | clean | 1284 | 94.08 | +0.00 | 0.060 | 0.058 | 0.987 |
| eot_v0 | hin | g711_mulaw | 1284 | 94.39 | +0.31 | 0.060 | 0.052 | 0.986 |
| eot_v0 | hin | amrnb_7.95 | 1284 | 93.22 | -0.86 | 0.080 | 0.057 | 0.985 |
| eot_v0 | hin | amrnb_4.75 | 1284 | 93.30 | -0.78 | 0.073 | 0.061 | 0.982 |
| eot_v0 | hin | noise10_amr7.95 | 1284 | 91.20 | -2.88 | 0.104 | 0.073 | 0.976 |
| eot_v0 | hin | noise5_amr7.95 | 1284 | 88.63 | -5.45 | 0.146 | 0.084 | 0.964 |
| eot_v0 | hin | mixed | 1284 | 91.90 | -2.18 | 0.093 | 0.070 | 0.976 |
| eot_v1 | eng | clean | 1947 | 93.79 | +0.00 | 0.076 | 0.048 | 0.984 |
| eot_v1 | eng | g711_mulaw | 1947 | 92.71 | -1.08 | 0.088 | 0.058 | 0.979 |
| eot_v1 | eng | amrnb_7.95 | 1947 | 93.01 | -0.77 | 0.086 | 0.053 | 0.978 |
| eot_v1 | eng | amrnb_4.75 | 1947 | 92.35 | -1.44 | 0.091 | 0.062 | 0.977 |
| eot_v1 | eng | noise10_amr7.95 | 1947 | 91.06 | -2.72 | 0.114 | 0.064 | 0.967 |
| eot_v1 | eng | noise5_amr7.95 | 1947 | 88.91 | -4.88 | 0.135 | 0.086 | 0.960 |
| eot_v1 | eng | mixed | 1947 | 91.32 | -2.47 | 0.107 | 0.066 | 0.966 |
| eot_v1 | hin | clean | 1284 | 93.69 | +0.00 | 0.070 | 0.057 | 0.987 |
| eot_v1 | hin | g711_mulaw | 1284 | 93.77 | +0.08 | 0.075 | 0.051 | 0.986 |
| eot_v1 | hin | amrnb_7.95 | 1284 | 93.93 | +0.23 | 0.070 | 0.052 | 0.985 |
| eot_v1 | hin | amrnb_4.75 | 1284 | 93.15 | -0.55 | 0.078 | 0.060 | 0.983 |
| eot_v1 | hin | noise10_amr7.95 | 1284 | 91.28 | -2.41 | 0.115 | 0.061 | 0.972 |
| eot_v1 | hin | noise5_amr7.95 | 1284 | 89.25 | -4.44 | 0.154 | 0.064 | 0.961 |
| eot_v1 | hin | mixed | 1284 | 91.20 | -2.49 | 0.106 | 0.072 | 0.974 |
| smart_turn_v3.2_int8 | eng | clean | 1947 | 92.96 | +0.00 | 0.094 | 0.046 | 0.979 |
| smart_turn_v3.2_int8 | eng | g711_mulaw | 1947 | 92.14 | -0.82 | 0.109 | 0.047 | 0.974 |
| smart_turn_v3.2_int8 | eng | amrnb_7.95 | 1947 | 91.73 | -1.23 | 0.123 | 0.041 | 0.971 |
| smart_turn_v3.2_int8 | eng | amrnb_4.75 | 1947 | 90.76 | -2.21 | 0.137 | 0.046 | 0.969 |
| smart_turn_v3.2_int8 | eng | noise10_amr7.95 | 1947 | 85.36 | -7.60 | 0.243 | 0.046 | 0.936 |
| smart_turn_v3.2_int8 | eng | noise5_amr7.95 | 1947 | 81.66 | -11.30 | 0.316 | 0.045 | 0.909 |
| smart_turn_v3.2_int8 | eng | mixed | 1947 | 84.28 | -8.68 | 0.274 | 0.036 | 0.930 |
| smart_turn_v3.2_int8 | hin | clean | 1284 | 90.26 | +0.00 | 0.177 | 0.024 | 0.982 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1284 | 86.37 | -3.89 | 0.275 | 0.009 | 0.971 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1284 | 82.79 | -7.48 | 0.353 | 0.006 | 0.962 |
| smart_turn_v3.2_int8 | hin | amrnb_4.75 | 1284 | 80.53 | -9.74 | 0.400 | 0.006 | 0.951 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1284 | 73.68 | -16.59 | 0.548 | 0.001 | 0.897 |
| smart_turn_v3.2_int8 | hin | noise5_amr7.95 | 1284 | 69.55 | -20.72 | 0.626 | 0.009 | 0.858 |
| smart_turn_v3.2_int8 | hin | mixed | 1284 | 73.52 | -16.74 | 0.548 | 0.004 | 0.895 |