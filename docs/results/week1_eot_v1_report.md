# eot_v1: eot_v0's recipe plus real human Hindi voices (GPU box, Oct 25 2026)

Job 5. eot_v0 was the first fine-tune of Smart Turn v3.2 for the phone line. On FLEURS (real human readers) it kept its
phone-line robustness but was only slightly better than Smart Turn on clean audio (`week1_fleurs_eot_report.md`): every
Hindi clip it trained on was TTS. eot_v1 is the same recipe with FLEURS Hindi's **train** split added. FLEURS test and dev
stay out of training: they are the real-voice test set.

- Data: `scripts/make_fleurs_eot.py --splits train --to-store data\external\eot-train --max-pauses 2`. 2,120 recordings
  gave 2,120 finished clips plus 1,641 mid-sentence pauses = 3,761 real-voice clips. The store went from 77,808 to 81,569 clips
  (105 of the new clips landed in dev by md5 and form the `hin_real` dev group).
- Training: `scripts/train_eot.py --config configs\train_eot_v1.yaml --out eot_v1` at commit f66ff32 (clean tree).
  The only config difference from eot_v0 is `real_hin_repeat: 3`. 1,020 steps = 4 epochs of 79,153 clips, bf16 on the RTX 4000 Ada, 25 min.
  Best dev checkpoint: **step 400** (dev score 90.26 = mean phone-line accuracy over eng / hin / hin_real).
  Files: `week1_eot_v1_train_summary.json`, `_train_meta.json`, `_train_dev.csv`.
  Shipped dev thresholds: 0.49-0.74 (`dev_thresholds` in the summary).
- Real-voice test: `eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin --models smart_turn_v3.2_int8
  codewithmoin_8s --onnx eot_v0=... eot_v1=... --conditions clean g711_mulaw amrnb_7.95 noise10_amr7.95 mixed --threads 24`
  (15 min; `week1_eot_v1_fleurs_results.json`; full table below).
- TTS test: `eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx eot_v0=... eot_v1=... --conditions clean g711_mulaw
  amrnb_7.95 amrnb_4.75 noise10_amr7.95 noise5_amr7.95 mixed --threads 24` (48 min; `week1_eot_v1_eval_report.md`, `_results.json`).
  Smart Turn's and eot_v0's rows match Jobs 1-4 exactly.
- Model files (GPU box, not in git): `runs\eot_v1\model.onnx` (sha256 0ccac636…89a52b), `best.pt` (sha256 fa824d31…2f33aa).

All models below are scored at threshold 0.5. Accuracy is in %, with the cut-in rate in brackets. Significance is from a
paired McNemar test on the same clips.

## Q1. Did real-voice Hindi improve? Yes, by a lot.

FLEURS Hindi, n = 1046 (657 finished, 389 mid-sentence pauses; FLEURS test + dev splits, never trained on and not used to pick the checkpoint).

| Condition | eot_v1 | eot_v0 | codewithmoin | Smart Turn int8 |
|---|---|---|---|---|
| clean | **88.8** (18%) | 74.6 (33%) | 81.6 (37%) | 72.8 (60%) |
| G.711 μ-law | **86.7** (23%) | 75.0 (37%) | 80.7 (39%) | 69.4 (73%) |
| AMR-NB 7.95 | **86.9** (25%) | 74.8 (39%) | 80.5 (41%) | 68.4 (77%) |
| noise 10 dB + AMR 7.95 | **83.4** (33%) | 74.2 (42%) | 76.3 (52%) | 65.2 (84%) |
| mixed chain | **82.8** (31%) | 73.5 (40%) | 75.1 (54%) | 65.7 (79%) |
| AUC, clean / mixed | **0.926 / 0.873** | 0.792 / 0.761 | 0.821 / 0.754 | 0.774 / 0.602 |

- **+14.2 points over eot_v0 on clean real voices** (178 vs 29 discordant clips, p = 2e-27). Cut-ins fall from 33% to 18%
  and missed ends from 21% to 7%, so eot_v0's habit of waiting too long on real voices is gone.
- **It beats codewithmoin, the best model so far on real voices, in every condition**, by 7.2-10.4 points
  (clean 123 vs 47 discordant clips, p = 5e-9; noise 10 dB p = 3e-8; mixed p = 1e-8). AUC rises from 0.82 to 0.93 on clean audio.
- **Both FLEURS splits agree**: clean 89.1 on test (n = 672) and 88.2 on dev (n = 374).
- **Phone-line cost on real voices**: −2 points for a codec alone and −5.5 to −6 under noise or the mixed chain, about the same
  drop as codewithmoin. eot_v0 stayed flatter (−1), but at a level 9-14 points lower.

## Q2. Did the phone-line robustness and the TTS results hold? Yes for Hindi. English slipped slightly.

Smart Turn's test set (all Hindi is TTS). Hindi n = 1284 in every condition; English n = 7820 clean, 1947 (25% hash sample) per phone condition.

| Condition | Hindi eot_v1 | Hindi eot_v0 | Hindi Smart Turn | English eot_v1 | English eot_v0 | English Smart Turn |
|---|---|---|---|---|---|---|
| clean | 93.7 (7%) | 94.1 (6%) | 90.3 (18%) | 94.7 | **95.4** | 94.3 |
| G.711 μ-law | 93.8 (8%) | 94.4 (6%) | 86.4 (27%) | 92.7 | 93.5 | 92.1 |
| AMR-NB 7.95 | 93.9 (7%) | 93.2 (8%) | 82.8 (35%) | 93.0 | 93.5 | 91.7 |
| AMR-NB 4.75 | 93.2 (8%) | 93.3 (7%) | 80.5 (40%) | 92.4 | 92.4 | 90.8 |
| noise 10 dB + AMR 7.95 | 91.3 (12%) | 91.2 (10%) | 73.7 (55%) | 91.1 | 92.1 | 85.4 |
| noise 5 dB + AMR 7.95 | 89.3 (15%) | 88.6 (15%) | 69.5 (63%) | 88.9 | **90.6** | 81.7 |
| mixed chain | 91.2 (11%) | 91.9 (9%) | 73.5 (55%) | 91.3 | **92.5** | 84.3 |

- **Hindi TTS: no real change.** eot_v1 is within ±0.7 points of eot_v0 in every condition, and none of the differences is
  significant (clean p = 0.58, 5 dB p = 0.46, mixed p = 0.40). It still beats Smart Turn by 3.4 (clean) to 19.7 (5 dB) points.
- **English: a small, real cost.** Clean, all 7820 clips: 95.40 → 94.69 (−0.7; 86 vs 141 discordant clips, p = 3e-4).
  Under noise: −1.7 at 5 dB (p = 0.003) and −1.2 on the mixed chain (p = 0.02). eot_v1 is still above Smart Turn everywhere
  (clean +0.4, p = 0.06; noise +6 to +7). The likely cause: real-voice Hindi is now 3 × 3,656 ≈ 11k of the ~98k clips seen per
  epoch (255 steps × 384), so English's share falls from about 73% to 65%. A lower `real_hin_repeat` (2) is the obvious knob if English matters for a use case.

## Verdict

**eot_v1 replaces eot_v0 as our best model.** Its gains on real voices (+9 to +14 points, half the cut-ins) are far
larger than its English cost (−0.7 to −1.7). Hindi TTS and phone-line robustness are unchanged.

## Read with care

- FLEURS is **read speech**, not conversation. Readers pause at commas, not before verbs. Our own recordings are the real test.
- FLEURS publishes no speaker IDs, so some readers may appear in both its train split (training) and its test/dev splits (this
  test). If so, part of the gain could be speaker familiarity rather than better Hindi prosody. The sentences differ between splits.
- An 18% cut-in rate on clean real voices is still too high for a product. Threshold tuning and conversational data come next.
- The checkpoint was picked on 105 real-voice dev clips (±3-4 points of noise per group). Later steps did slightly better on TTS and slightly worse on
  `hin_real` dev (step 1020: 88.6 / 81.9 vs 93.3 / 86.7 at step 400).
- Smart Turn int8 scores 72.75 on clean FLEURS here and 72.37 on the Dev box (`week1_fleurs_eot_report.md`). int8 kernels differ
  slightly between CPUs. The fp32 models (eot_v0, eot_v1, codewithmoin) match the Dev box run exactly.

## Next

- Copy `runs\eot_v1\model.onnx` to the Dev box's `weights\eot_v1\` (Chrome Remote Desktop, like eot_v0) and add it to the demo
  (`PEHLEAAP_TURN_MODELS=smart_turn_v3.2_int8,eot_v0,eot_v1`).
- int8 for eot_v1 (`quantize_eot.py --run eot_v1`, about 8 min) if CPU-bound deployment matters.
- The real test: our own recorded conversations (`eval_sessions.py`), once the pilot and the first sessions are labelled.

---

# Full FLEURS table (`runs\eot_v1_fleurs\report.md`)

# End-of-turn baselines: clean vs phone line

Data: `data\external\fleurs-hi-eot`: FLEURS Hindi read speech by human speakers (google/fleurs, CC BY 4.0), cut at the sentence end (finished) or 0.2 s into a mid-sentence pause (not finished).
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Clean audio (model-card accuracy is for Smart Turn's test set, so no match check)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | hin | 1046 | 72.75 | - | n/a | n/a |
| codewithmoin_8s | hin | 1046 | 81.55 | - | n/a | n/a |
| eot_v0 | hin | 1046 | 74.57 | - | n/a | n/a |
| eot_v1 | hin | 1046 | 88.81 | - | n/a | n/a |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `g711_mulaw`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_7.95`: Band-limited to 8 kHz, then this codec; nothing else
- `noise10_amr7.95`: MUSAN noise at 10 dB SNR, AMR-NB 7.95
- `mixed`: One random chain per clip from configs/augment.yaml (no speakerphone echo: the clips have no agent track)
- Noise: the 181 held-out MUSAN noise files (CC BY 4.0; training uses the other 80%), checked against `data/manifests/noise_musan.jsonl`

| Model | Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|---|
| codewithmoin_8s | hin | clean | 1046 | 81.55 | +0.00 | 0.370 | 0.075 | 0.821 |
| codewithmoin_8s | hin | g711_mulaw | 1046 | 80.69 | -0.86 | 0.391 | 0.076 | 0.814 |
| codewithmoin_8s | hin | amrnb_7.95 | 1046 | 80.50 | -1.05 | 0.411 | 0.067 | 0.809 |
| codewithmoin_8s | hin | noise10_amr7.95 | 1046 | 76.29 | -5.26 | 0.524 | 0.067 | 0.754 |
| codewithmoin_8s | hin | mixed | 1046 | 75.05 | -6.50 | 0.540 | 0.078 | 0.754 |
| eot_v0 | hin | clean | 1046 | 74.57 | +0.00 | 0.332 | 0.209 | 0.792 |
| eot_v0 | hin | g711_mulaw | 1046 | 74.95 | +0.38 | 0.368 | 0.181 | 0.781 |
| eot_v0 | hin | amrnb_7.95 | 1046 | 74.76 | +0.19 | 0.388 | 0.172 | 0.765 |
| eot_v0 | hin | noise10_amr7.95 | 1046 | 74.19 | -0.38 | 0.424 | 0.160 | 0.751 |
| eot_v0 | hin | mixed | 1046 | 73.52 | -1.05 | 0.396 | 0.187 | 0.761 |
| eot_v1 | hin | clean | 1046 | 88.81 | +0.00 | 0.183 | 0.070 | 0.926 |
| eot_v1 | hin | g711_mulaw | 1046 | 86.71 | -2.10 | 0.231 | 0.075 | 0.914 |
| eot_v1 | hin | amrnb_7.95 | 1046 | 86.90 | -1.91 | 0.254 | 0.058 | 0.907 |
| eot_v1 | hin | noise10_amr7.95 | 1046 | 83.37 | -5.45 | 0.334 | 0.067 | 0.880 |
| eot_v1 | hin | mixed | 1046 | 82.79 | -6.02 | 0.314 | 0.088 | 0.873 |
| smart_turn_v3.2_int8 | hin | clean | 1046 | 72.75 | +0.00 | 0.604 | 0.076 | 0.774 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1046 | 69.41 | -3.35 | 0.728 | 0.056 | 0.728 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1046 | 68.36 | -4.40 | 0.766 | 0.050 | 0.682 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1046 | 65.20 | -7.55 | 0.843 | 0.055 | 0.589 |
| smart_turn_v3.2_int8 | hin | mixed | 1046 | 65.68 | -7.07 | 0.794 | 0.076 | 0.602 |