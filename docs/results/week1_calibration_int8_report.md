# Decision-threshold calibration on the phone line

Predictions from `runs/eot_v0_int8_eval`: phone subset (all Hindi + 25% of English), so clean and
phone rows cover the same clips. Thresholds are **cross-fitted**: the clips are split in two by a hash of their id, and each
half is scored with a threshold tuned (for accuracy) on the other half. Variants: **card** = the model card's threshold;
**channel** = one for clean audio, one for the phone line; **+lang** = per channel and language; **cond+lang** = per exact
condition and language, an upper bound since a deployment rarely knows the codec. Cut-in = share of unfinished turns called finished.

## smart_turn_v3.2_int8 (card threshold 0.5)

Thresholds tuned on all of these clips, per channel/language (reference only: these are test clips; a shipped model takes its thresholds from dev data, as train_eot.py's summary.json does): clean/eng 0.68, clean/hin 0.92, phone/eng 0.79, phone/hin 0.95

| Lang | Condition | n | Acc card | channel | +lang | cond+lang | Cut-in card | Cut-in +lang |
|---|---|---|---|---|---|---|---|---|
| eng | clean | 1947 | 92.96 | 92.71 | 92.96 | 92.96 | 9.4% | 9.1% |
| eng | agc_clip_g711 | 1947 | 89.37 | 88.75 | 89.88 | 89.42 | 16.4% | 12.0% |
| eng | amrnb_4.75 | 1947 | 90.76 | 89.37 | 90.86 | 90.55 | 13.7% | 9.7% |
| eng | amrnb_7.95 | 1947 | 91.73 | 90.24 | 91.53 | 91.22 | 12.3% | 9.3% |
| eng | g711_mulaw | 1947 | 92.14 | 90.19 | 92.14 | 92.04 | 10.9% | 8.3% |
| eng | loss3_g711 | 1947 | 89.52 | 89.37 | 90.09 | 89.57 | 16.1% | 11.2% |
| eng | mixed | 1947 | 84.28 | 85.16 | 85.52 | 84.33 | 27.4% | 21.7% |
| eng | noise10_amr7.95 | 1947 | 85.36 | 85.93 | 87.11 | 87.01 | 24.3% | 17.6% |
| eng | noise20_amr7.95 | 1947 | 88.14 | 88.03 | 89.42 | 89.42 | 19.3% | 13.9% |
| eng | noise5_amr7.95 | 1947 | 81.66 | 83.36 | 83.20 | 83.56 | 31.6% | 25.2% |
| eng | tandem_amr4.75 | 1947 | 89.47 | 88.03 | 89.68 | 89.27 | 16.9% | 12.7% |
| hin | clean | 1284 | 90.26 | 92.06 | 92.68 | 92.68 | 17.7% | 7.8% |
| hin | agc_clip_g711 | 1284 | 82.63 | 88.32 | 88.40 | 88.94 | 35.1% | 12.7% |
| hin | amrnb_4.75 | 1284 | 80.53 | 86.60 | 87.15 | 87.23 | 40.0% | 18.0% |
| hin | amrnb_7.95 | 1284 | 82.79 | 87.62 | 88.79 | 88.40 | 35.3% | 15.8% |
| hin | g711_mulaw | 1284 | 86.37 | 89.95 | 89.95 | 89.25 | 27.5% | 10.9% |
| hin | loss3_g711 | 1284 | 82.09 | 87.38 | 88.32 | 88.24 | 35.8% | 13.7% |
| hin | mixed | 1284 | 73.52 | 80.30 | 81.07 | 80.69 | 54.8% | 28.1% |
| hin | noise10_amr7.95 | 1284 | 73.68 | 78.97 | 80.61 | 81.15 | 54.8% | 32.0% |
| hin | noise20_amr7.95 | 1284 | 77.41 | 83.72 | 84.66 | 84.27 | 46.5% | 23.7% |
| hin | noise5_amr7.95 | 1284 | 69.55 | 76.79 | 77.96 | 77.65 | 62.6% | 38.2% |
| hin | tandem_amr4.75 | 1284 | 76.40 | 83.57 | 84.74 | 84.97 | 49.1% | 24.2% |

## eot_v0 (card threshold 0.5)

Thresholds tuned on all of these clips, per channel/language (reference only: these are test clips; a shipped model takes its thresholds from dev data, as train_eot.py's summary.json does): clean/eng 0.35, clean/hin 0.38, phone/eng 0.40, phone/hin 0.49

Shipped thresholds (from its dev clips): clean/eng 0.48, clean/hin 0.57, phone/eng 0.52, phone/hin 0.44

| Lang | Condition | n | Acc card | channel | +lang | cond+lang | Cut-in card | Cut-in +lang | Acc shipped | Cut-in shipped |
|---|---|---|---|---|---|---|---|---|---|---|
| eng | clean | 1947 | 94.30 | 94.56 | 94.56 | 94.56 | 6.6% | 7.2% | 94.35 | 6.6% |
| eng | agc_clip_g711 | 1947 | 93.43 | 93.68 | 93.68 | 93.58 | 8.2% | 8.6% | 93.27 | 8.1% |
| eng | amrnb_4.75 | 1947 | 92.45 | 93.01 | 93.01 | 92.81 | 8.3% | 8.8% | 92.45 | 8.3% |
| eng | amrnb_7.95 | 1947 | 93.53 | 93.73 | 93.73 | 93.63 | 7.6% | 8.4% | 93.48 | 7.6% |
| eng | g711_mulaw | 1947 | 93.53 | 93.89 | 93.94 | 93.53 | 7.5% | 8.2% | 93.53 | 7.3% |
| eng | loss3_g711 | 1947 | 93.58 | 93.63 | 93.63 | 93.43 | 8.1% | 8.8% | 93.48 | 8.1% |
| eng | mixed | 1947 | 92.50 | 92.66 | 92.71 | 92.30 | 8.8% | 10.4% | 92.55 | 8.6% |
| eng | noise10_amr7.95 | 1947 | 92.14 | 92.04 | 92.04 | 92.14 | 10.2% | 11.1% | 92.24 | 9.9% |
| eng | noise20_amr7.95 | 1947 | 92.24 | 92.45 | 92.60 | 91.94 | 9.6% | 10.1% | 92.24 | 9.4% |
| eng | noise5_amr7.95 | 1947 | 90.60 | 91.06 | 91.17 | 90.60 | 11.2% | 12.5% | 90.55 | 11.0% |
| eng | tandem_amr4.75 | 1947 | 92.76 | 92.66 | 92.66 | 91.83 | 9.4% | 10.3% | 92.66 | 9.4% |
| hin | clean | 1284 | 94.08 | 94.16 | 94.00 | 94.00 | 6.0% | 6.2% | 94.08 | 5.5% |
| hin | agc_clip_g711 | 1284 | 94.70 | 94.31 | 94.63 | 94.78 | 5.9% | 6.0% | 94.31 | 6.8% |
| hin | amrnb_4.75 | 1284 | 93.30 | 93.15 | 93.22 | 93.30 | 7.3% | 7.6% | 93.30 | 8.1% |
| hin | amrnb_7.95 | 1284 | 93.22 | 93.07 | 93.22 | 93.30 | 8.0% | 7.8% | 93.22 | 8.5% |
| hin | g711_mulaw | 1284 | 94.39 | 93.93 | 94.31 | 94.39 | 6.0% | 6.0% | 93.93 | 7.5% |
| hin | loss3_g711 | 1284 | 94.24 | 93.85 | 94.16 | 94.16 | 5.5% | 5.5% | 93.69 | 6.7% |
| hin | mixed | 1284 | 91.90 | 91.82 | 91.82 | 91.51 | 9.3% | 9.1% | 92.21 | 10.2% |
| hin | noise10_amr7.95 | 1284 | 91.20 | 91.36 | 91.04 | 91.12 | 10.4% | 10.7% | 91.36 | 11.4% |
| hin | noise20_amr7.95 | 1284 | 93.61 | 93.61 | 93.61 | 93.69 | 8.3% | 7.8% | 93.46 | 8.9% |
| hin | noise5_amr7.95 | 1284 | 88.63 | 89.02 | 88.47 | 88.08 | 14.6% | 14.5% | 88.79 | 15.6% |
| hin | tandem_amr4.75 | 1284 | 93.15 | 93.38 | 93.22 | 93.38 | 7.5% | 7.5% | 93.15 | 8.3% |

## eot_v0_int8 (card threshold 0.5)

Thresholds tuned on all of these clips, per channel/language (reference only: these are test clips; a shipped model takes its thresholds from dev data, as train_eot.py's summary.json does): clean/eng 0.57, clean/hin 0.77, phone/eng 0.44, phone/hin 0.77

Shipped thresholds (from its dev clips): clean/eng 0.39, clean/hin 0.81, phone/eng 0.37, phone/hin 0.71

| Lang | Condition | n | Acc card | channel | +lang | cond+lang | Cut-in card | Cut-in +lang | Acc shipped | Cut-in shipped |
|---|---|---|---|---|---|---|---|---|---|---|
| eng | clean | 1947 | 94.04 | 93.89 | 93.63 | 93.63 | 7.2% | 5.7% | 93.89 | 8.5% |
| eng | agc_clip_g711 | 1947 | 93.01 | 92.96 | 93.27 | 92.96 | 8.8% | 9.6% | 93.12 | 10.4% |
| eng | amrnb_4.75 | 1947 | 92.14 | 91.58 | 92.35 | 92.35 | 8.6% | 9.5% | 92.55 | 10.2% |
| eng | amrnb_7.95 | 1947 | 93.22 | 92.71 | 93.53 | 93.68 | 7.7% | 8.2% | 93.37 | 9.1% |
| eng | g711_mulaw | 1947 | 93.53 | 93.58 | 93.63 | 93.27 | 8.5% | 8.7% | 93.63 | 9.0% |
| eng | loss3_g711 | 1947 | 93.43 | 93.22 | 93.37 | 93.07 | 8.7% | 9.2% | 93.22 | 10.1% |
| eng | mixed | 1947 | 91.22 | 90.70 | 91.11 | 90.81 | 11.5% | 12.6% | 90.76 | 14.4% |
| eng | noise10_amr7.95 | 1947 | 91.58 | 91.63 | 91.27 | 91.37 | 10.3% | 11.8% | 91.11 | 13.1% |
| eng | noise20_amr7.95 | 1947 | 92.50 | 92.04 | 92.76 | 92.19 | 9.3% | 10.1% | 92.45 | 11.4% |
| eng | noise5_amr7.95 | 1947 | 89.63 | 89.57 | 89.83 | 89.16 | 13.3% | 14.2% | 89.47 | 16.0% |
| eng | tandem_amr4.75 | 1947 | 91.53 | 91.27 | 91.42 | 91.47 | 10.6% | 11.5% | 91.58 | 12.2% |
| hin | clean | 1284 | 92.83 | 93.77 | 93.93 | 93.93 | 11.5% | 7.3% | 93.61 | 7.2% |
| hin | agc_clip_g711 | 1284 | 91.98 | 92.45 | 93.30 | 93.38 | 13.2% | 8.3% | 93.30 | 8.9% |
| hin | amrnb_4.75 | 1284 | 91.04 | 91.51 | 92.29 | 92.06 | 16.4% | 10.2% | 92.13 | 11.5% |
| hin | amrnb_7.95 | 1284 | 91.51 | 92.21 | 92.68 | 92.45 | 14.8% | 9.9% | 92.52 | 10.7% |
| hin | g711_mulaw | 1284 | 92.76 | 93.22 | 93.77 | 93.93 | 11.5% | 7.8% | 93.69 | 8.6% |
| hin | loss3_g711 | 1284 | 91.67 | 91.98 | 92.68 | 92.60 | 13.5% | 9.8% | 92.76 | 10.1% |
| hin | mixed | 1284 | 89.02 | 89.49 | 90.58 | 90.97 | 20.0% | 12.0% | 90.34 | 13.5% |
| hin | noise10_amr7.95 | 1284 | 89.02 | 89.17 | 89.80 | 89.64 | 20.8% | 14.3% | 89.64 | 15.9% |
| hin | noise20_amr7.95 | 1284 | 89.95 | 90.81 | 91.20 | 91.74 | 18.4% | 13.2% | 91.43 | 13.7% |
| hin | noise5_amr7.95 | 1284 | 86.29 | 86.68 | 87.62 | 85.59 | 25.7% | 17.7% | 87.54 | 18.7% |
| hin | tandem_amr4.75 | 1284 | 90.65 | 91.04 | 91.90 | 91.98 | 16.6% | 11.1% | 91.82 | 12.0% |
