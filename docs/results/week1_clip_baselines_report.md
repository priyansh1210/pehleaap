# End-of-turn baselines: clean vs phone line

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
| abhishek_tiny_attn | eng | g711_mulaw | 1947 | 60.55 | -0.36 | 0.494 | 0.290 | 0.657 |
| abhishek_tiny_attn | eng | amrnb_7.95 | 1947 | 59.84 | -1.08 | 0.513 | 0.286 | 0.650 |
| abhishek_tiny_attn | hin | clean | 1284 | 68.46 | +0.00 | 0.403 | 0.235 | 0.782 |
| abhishek_tiny_attn | hin | g711_mulaw | 1284 | 68.85 | +0.39 | 0.395 | 0.235 | 0.775 |
| abhishek_tiny_attn | hin | amrnb_7.95 | 1284 | 68.30 | -0.16 | 0.377 | 0.262 | 0.760 |
| chhotu_int8 | eng | clean | 1947 | 88.70 | +0.00 | 0.157 | 0.067 | 0.951 |
| chhotu_int8 | eng | g711_mulaw | 1947 | 65.64 | -23.06 | 0.428 | 0.256 | 0.711 |
| chhotu_int8 | eng | amrnb_7.95 | 1947 | 62.81 | -25.89 | 0.610 | 0.124 | 0.711 |
| chhotu_int8 | hin | clean | 1284 | 88.79 | +0.00 | 0.148 | 0.079 | 0.961 |
| chhotu_int8 | hin | g711_mulaw | 1284 | 62.07 | -26.71 | 0.517 | 0.253 | 0.665 |
| chhotu_int8 | hin | amrnb_7.95 | 1284 | 62.54 | -26.25 | 0.660 | 0.112 | 0.675 |
| codewithmoin_8s | eng | clean | 1947 | 93.07 | +0.00 | 0.089 | 0.049 | 0.978 |
| codewithmoin_8s | eng | g711_mulaw | 1947 | 91.58 | -1.49 | 0.102 | 0.066 | 0.975 |
| codewithmoin_8s | eng | amrnb_7.95 | 1947 | 92.35 | -0.72 | 0.096 | 0.057 | 0.975 |
| codewithmoin_8s | hin | clean | 1284 | 93.93 | +0.00 | 0.067 | 0.055 | 0.985 |
| codewithmoin_8s | hin | g711_mulaw | 1284 | 93.22 | -0.70 | 0.075 | 0.061 | 0.984 |
| codewithmoin_8s | hin | amrnb_7.95 | 1284 | 92.68 | -1.25 | 0.073 | 0.073 | 0.981 |
| deveshu_fp32 | eng | clean | 1947 | 93.32 | +0.00 | 0.079 | 0.055 | 0.980 |
| deveshu_fp32 | eng | g711_mulaw | 1947 | 92.66 | -0.67 | 0.084 | 0.063 | 0.977 |
| deveshu_fp32 | eng | amrnb_7.95 | 1947 | 92.45 | -0.87 | 0.091 | 0.060 | 0.974 |
| deveshu_fp32 | hin | clean | 1284 | 93.22 | +0.00 | 0.059 | 0.076 | 0.984 |
| deveshu_fp32 | hin | g711_mulaw | 1284 | 92.60 | -0.62 | 0.076 | 0.072 | 0.981 |
| deveshu_fp32 | hin | amrnb_7.95 | 1284 | 90.97 | -2.26 | 0.088 | 0.093 | 0.974 |
| smart_turn_v3.2_int8 | eng | clean | 1947 | 92.96 | +0.00 | 0.094 | 0.046 | 0.979 |
| smart_turn_v3.2_int8 | eng | g711_mulaw | 1947 | 92.14 | -0.82 | 0.109 | 0.047 | 0.974 |
| smart_turn_v3.2_int8 | eng | amrnb_7.95 | 1947 | 91.73 | -1.23 | 0.123 | 0.041 | 0.971 |
| smart_turn_v3.2_int8 | hin | clean | 1284 | 90.26 | +0.00 | 0.177 | 0.024 | 0.982 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1284 | 86.37 | -3.89 | 0.275 | 0.009 | 0.971 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1284 | 82.79 | -7.48 | 0.353 | 0.006 | 0.962 |