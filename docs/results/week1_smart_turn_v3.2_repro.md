# Smart Turn v3.2 (cpu) reproduction

| Lang | n (ours/pub) | Acc ours | Acc pub | Δ | FP%tot ours/pub | FN%tot ours/pub |
|---|---|---|---|---|---|---|
| eng | 7820/7820 | 94.17 | 94.26 | -0.09 | 3.87/3.75 | 1.96/1.99 |
| hin | 1284/1284 | 89.95 | 90.11 | -0.16 | 8.72/8.57 | 1.32/1.32 |

**PASS** (accuracy within 1 pt and full sample count)

## Human vs synthetic audio

| Group | n | Acc | FPR (cut-in) | FNR |
|---|---|---|---|---|
| eng/human | 5367 | 94.76 | 0.069 | 0.035 |
| eng/synthetic | 2453 | 92.87 | 0.091 | 0.051 |
| hin/synthetic | 1284 | 89.95 | 0.182 | 0.025 |

## By source dataset

| Group | n | Acc | FPR (cut-in) | FNR |
|---|---|---|---|---|
| eng/chirp3_1 | 1471 | 92.05 | 0.115 | 0.044 |
| eng/chirp3_3_short | 104 | 91.35 | 0.054 | 0.125 |
| eng/human_5 | 402 | 96.52 | 0.031 | 0.039 |
| eng/human_convcollector_1 | 90 | 86.67 | 0.154 | 0.105 |
| eng/liva_1 | 3831 | 93.84 | 0.084 | 0.040 |
| eng/midcentury_1 | 1044 | 98.18 | 0.026 | 0.010 |
| eng/orpheus_endfiller_1 | 181 | 96.13 | 0.000 | 0.076 |
| eng/orpheus_grammar_1 | 163 | 91.41 | 0.141 | 0.035 |
| eng/orpheus_midfiller_1 | 140 | 87.14 | 0.130 | 0.127 |
| eng/rime_2 | 394 | 97.46 | 0.026 | 0.024 |
| hin/chirp3_1 | 1284 | 89.95 | 0.182 | 0.025 |