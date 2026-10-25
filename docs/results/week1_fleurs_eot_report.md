# End-of-turn models on real human Hindi (FLEURS read speech), Dev box, Oct 25 2026

**Summary.** Every Hindi clip in Smart Turn's test set is TTS. Here 657 human readers' sentences end ("finished")
or stop 0.2 s into a mid-sentence pause ("not finished"): 1,046 clips (`scripts/make_fleurs_eot.py`).
- Real voices are much harder for every model. Smart Turn falls from 90.3% (TTS Hindi) to 72.4% clean, and calls 61%
  of mid-sentence pauses finished.
- eot_v0's phone-line robustness carries over to human voices. It stays at 73.6-75.0% in every condition, while
  Smart Turn falls to 64.6% at 10 dB noise (AUC 0.75 vs 0.59). It also halves Smart Turn's cut-ins everywhere.
- But on clean real voices eot_v0 ranks only slightly better than Smart Turn (AUC 0.79 vs 0.78), and waits too long
  on 21% of finished sentences. Its Hindi training data was all TTS, so it learned the line, not real Hindi prosody.
- codewithmoin (community model) is best on real voices: 81.6%, AUC 0.82 clean. eot_v0 matches its AUC only under
  noise and mixed (0.75-0.76).
- Caveat: read speech. Readers pause at commas with intonation that can sound finished, so some "not finished" labels
  are ambiguous for every model.
- Next: add real-voice clips (FLEURS train split, then our recordings) to the training mix for eot_v1.

Command: `scripts/eval_clip_baselines.py --data-root data/external/fleurs-hi-eot --langs hin --models smart_turn_v3.2_int8
codewithmoin_8s --onnx eot_v0=weights/eot_v0/model.onnx --conditions clean g711_mulaw amrnb_7.95 noise10_amr7.95 mixed
--threads 8 --out fleurs_eot` (Dev box, 4 cores, about 27 min). The full report follows.

---

# End-of-turn baselines: clean vs phone line

Data: `data/external/fleurs-hi-eot`: FLEURS Hindi read speech by human speakers (google/fleurs, CC BY 4.0), cut at the sentence end (finished) or 0.2 s into a mid-sentence pause (not finished).
Each model is scored at its own model-card threshold. Cut-in = P(complete) above threshold on an unfinished turn.

## 1. Clean audio (model-card accuracy is for Smart Turn's test set, so no match check)

| Model | Lang | n | Acc | Card | Δ | Match |
|---|---|---|---|---|---|---|
| smart_turn_v3.2_int8 | hin | 1046 | 72.37 | - | n/a | n/a |
| codewithmoin_8s | hin | 1046 | 81.55 | - | n/a | n/a |
| eot_v0 | hin | 1046 | 74.57 | - | n/a | n/a |

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
| codewithmoin_8s | hin | mixed | 1046 | 75.43 | -6.12 | 0.532 | 0.076 | 0.756 |
| eot_v0 | hin | clean | 1046 | 74.57 | +0.00 | 0.332 | 0.209 | 0.792 |
| eot_v0 | hin | g711_mulaw | 1046 | 74.95 | +0.38 | 0.368 | 0.181 | 0.781 |
| eot_v0 | hin | amrnb_7.95 | 1046 | 74.76 | +0.19 | 0.388 | 0.172 | 0.765 |
| eot_v0 | hin | noise10_amr7.95 | 1046 | 74.19 | -0.38 | 0.424 | 0.160 | 0.751 |
| eot_v0 | hin | mixed | 1046 | 73.61 | -0.96 | 0.396 | 0.186 | 0.761 |
| smart_turn_v3.2_int8 | hin | clean | 1046 | 72.37 | +0.00 | 0.614 | 0.076 | 0.779 |
| smart_turn_v3.2_int8 | hin | g711_mulaw | 1046 | 69.60 | -2.77 | 0.725 | 0.055 | 0.726 |
| smart_turn_v3.2_int8 | hin | amrnb_7.95 | 1046 | 68.16 | -4.21 | 0.769 | 0.052 | 0.679 |
| smart_turn_v3.2_int8 | hin | noise10_amr7.95 | 1046 | 64.63 | -7.74 | 0.846 | 0.062 | 0.591 |
| smart_turn_v3.2_int8 | hin | mixed | 1046 | 65.30 | -7.07 | 0.799 | 0.079 | 0.597 |