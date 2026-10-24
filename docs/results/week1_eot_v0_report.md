# eot_v0: Smart Turn v3.2 fine-tuned for the phone line (GPU box, Oct 25 2026)

Job 4. Smart Turn v3.2 was fine-tuned on its own training set (smart-turn-data-v3.2-train, English + Hindi), with 80% of the
clips passed through a random simulated phone line (`configs/augment.yaml`, MUSAN training-split noise). Then it was
scored on Smart Turn's test set on the same clips and conditions as Jobs 1-3.

- Training: `scripts/train_eot.py --out eot_v0` at commit 148e778 (the only uncommitted change was memory notes). 904 steps =
  4 epochs of 75,497 clips (Hindi repeated 2x), bf16 on the RTX 4000 Ada, 22 min. Best dev checkpoint: step 900.
  Files: `week1_eot_v0_train_summary.json`, `week1_eot_v0_train_meta.json`, `week1_eot_v0_train_dev.csv` (dev check every 100 steps).
- Test set: `scripts/eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx eot_v0=runs\eot_v0\model.onnx`, 11 conditions
  (`week1_eot_v0_eval_report.md`, `_results.json`; 73 min on CPU). Smart Turn's rows match Jobs 1-3 exactly.
- Calibration: `scripts/calibrate_thresholds.py --runs clip_baselines clip_baselines_codecs clip_baselines_chain eot_v0_eval
  --shipped eot_v0=runs\eot_v0\summary.json --out calibration_all` (`week1_calibration_all_report.md`, `_results.json`).
- **Shipped thresholds were recomputed after training.** The run started before PR #14, so its `summary.json` had no
  `dev_thresholds`. We rescored `best.pt` with `train_eot.py`'s own `evaluate()` on the same dev keys and seed. All four dev
  accuracies and AUCs matched `best_dev` exactly. Then we added the accuracy-optimal dev thresholds to `summary.json`: clean/eng 0.483,
  clean/hin 0.574, phone/eng 0.515, phone/hin 0.440. None of them were tuned on test clips.

**Read with care.** All Hindi test clips are synthetic TTS. The chain conditions use the same family of simulations that
eot_v0 was trained on. The noise recordings are different (the 181 held-out MUSAN files), but gains there are expected,
not proof. The real test is real phone calls (plan §6).

## Hindi (n = 1284): accuracy, with the cut-in rate in brackets

eot_v0 is shown at 0.5 and at its shipped dev thresholds. Smart Turn is shown at its card threshold 0.5 and with
**+lang\*** thresholds: one per channel (clean/phone) and language, cross-fitted on these test clips. That is the best
recalibration alone can do without knowing the codec, and it is optimistic, because the thresholds see test data. The
other baselines use their card thresholds, and LiveKit uses its own (hi 0.305). Codec-only rows for the baselines come from
Jobs 1-2, and chain rows from Job 3. All rows use the same clips.

| Condition | eot_v0 @0.5 | eot_v0 shipped | Smart Turn @0.5 | Smart Turn +lang* | codewithmoin | deveshu | chhotu | abhishek | LiveKit v1-mini |
|---|---|---|---|---|---|---|---|---|---|
| noise 5 dB + AMR 7.95 | **88.6** (15%) | 88.8 (16%) | 69.5 (63%) | 77.8 (38%) | 86.3 | 77.9 | 54.4 | 61.1 | 56.2 |
| noise 10 dB + AMR 7.95 | **91.2** (10%) | 91.4 (11%) | 73.7 (55%) | 80.7 (32%) | 89.3 | 82.6 | 55.0 | 63.1 | 57.2 |
| noise 20 dB + AMR 7.95 | **93.6** (8%) | 93.5 (9%) | 77.4 (47%) | 84.7 (24%) | 91.0 | 87.8 | 54.9 | 66.9 | 59.3 |
| mixed chain | **91.9** (9%) | 92.2 (10%) | 73.5 (55%) | 81.3 (28%) | 89.4 | 82.6 | 54.3 | 62.9 | 57.3 |
| tandem AMR 4.75 → G.711 | **93.1** (7%) | 93.1 (8%) | 76.4 (49%) | 85.1 (23%) | 90.5 | 89.3 | 56.5 | 66.3 | 59.9 |
| G.711 + 3% loss | **94.2** (6%) | 93.7 (7%) | 82.1 (36%) | 88.2 (13%) | 92.3 | 90.3 | 55.5 | 66.5 | 61.1 |
| AGC + clip + G.711 | **94.7** (6%) | 94.3 (7%) | 82.6 (35%) | 88.3 (13%) | 92.6 | 91.1 | 54.9 | 66.4 | 60.5 |
| AMR-NB 4.75 | **93.3** (7%) | 93.3 (8%) | 80.5 (40%) | 87.2 (18%) | 91.8 | 90.6 | 63.1 | 67.3 | 59.3 |
| AMR-NB 7.95 | **93.2** (8%) | 93.2 (8%) | 82.8 (35%) | 88.6 (16%) | 92.7 | 91.0 | 62.5 | 68.3 | 59.7 |
| G.711 μ-law | **94.4** (6%) | 93.9 (7%) | 86.4 (27%) | 90.0 (11%) | 93.2 | 92.6 | 62.1 | 68.8 | 58.1 |
| clean | **94.1** (6%) | 94.1 (6%) | 90.3 (18%) | 92.7 (8%) | 93.9 | 93.2 | 88.8 | 68.5 | 55.1 |

## English phone subset (n = 1947, a 25% hash sample)

| Condition | eot_v0 @0.5 | eot_v0 shipped | Smart Turn @0.5 | Smart Turn +lang* | codewithmoin | deveshu | chhotu | abhishek | LiveKit v1-mini |
|---|---|---|---|---|---|---|---|---|---|
| noise 5 dB + AMR 7.95 | **90.6** (11%) | 90.5 (11%) | 81.7 (32%) | 83.3 (26%) | 87.3 | 86.2 | 51.7 | 59.5 | 58.9 |
| noise 10 dB + AMR 7.95 | **92.1** (10%) | 92.2 (10%) | 85.4 (24%) | 87.0 (18%) | 89.5 | 88.5 | 51.6 | 59.8 | 61.0 |
| noise 20 dB + AMR 7.95 | **92.2** (10%) | 92.2 (9%) | 88.1 (19%) | 89.2 (15%) | 91.8 | 90.8 | 52.3 | 59.5 | 63.1 |
| mixed chain | **92.5** (9%) | 92.6 (9%) | 84.3 (27%) | 85.4 (22%) | 88.8 | 85.9 | 52.2 | 59.5 | 57.8 |
| tandem AMR 4.75 → G.711 | **92.8** (9%) | 92.7 (9%) | 89.5 (17%) | 89.6 (14%) | 90.1 | 90.2 | 56.8 | 59.3 | 61.3 |
| G.711 + 3% loss | **93.6** (8%) | 93.5 (8%) | 89.5 (16%) | 90.2 (12%) | 91.3 | 91.0 | 53.4 | 60.5 | 60.9 |
| AGC + clip + G.711 | **93.4** (8%) | 93.3 (8%) | 89.4 (16%) | 89.9 (13%) | 91.5 | 90.8 | 52.0 | 59.7 | 61.0 |
| AMR-NB 4.75 | **92.4** (8%) | 92.4 (8%) | 90.8 (14%) | 90.8 (10%) | 91.2 | 91.1 | 62.0 | 58.4 | 61.2 |
| AMR-NB 7.95 | **93.5** (8%) | 93.5 (8%) | 91.7 (12%) | 91.5 (10%) | 92.3 | 92.4 | 62.8 | 59.8 | 60.5 |
| G.711 μ-law | **93.5** (7%) | 93.5 (7%) | 92.1 (11%) | 92.0 (9%) | 91.6 | 92.7 | 65.6 | 60.6 | 59.6 |
| clean | **94.3** (7%) | 94.4 (7%) | 93.0 (9%) | 93.0 (9%) | 93.1 | 93.3 | 88.7 | 60.9 | 56.2 |

## Findings

- **eot_v0 fixes the phone-line problem on this test set.** On Hindi it scores 88.6-94.7 in every condition, against Smart
  Turn's 69.5-86.4. The biggest gains are in the noisiest conditions: +19.1 points at 5 dB SNR, +18.4 on the mixed chain and +16.7 on
  tandem transcoding. Its cut-in rate falls from 35-63% to 6-15%. Its worst condition (5 dB noise, 88.6) is within
  1.6 points of Smart Turn on *clean* Hindi.
- **It beats what recalibration alone can do.** Even with optimistic, test-tuned thresholds, Smart Turn reaches only 77.8-90.0
  on Hindi phone conditions. eot_v0 is 8-11 points above that under noise and chains (6 on packet loss and AGC). AUC confirms it: at 5 dB, eot_v0 has 0.964
  against Smart Turn's 0.858. The model learned to separate finished from unfinished turns under noise. A threshold shift alone
  doesn't do that.
- **It beats every baseline in every condition.** The strongest community model, codewithmoin, is 0.5-2.6 points behind on
  Hindi phone conditions and 0.4-3.7 points behind on English noise and chains.
- **It needs no recalibration.** Its shipped dev thresholds (0.44-0.57) change accuracy by at most ±0.5 points against 0.5,
  in both directions. Its test-tuned thresholds (0.35-0.49) gain at most 0.6. Smart Turn needed 0.92-0.95 on Hindi.
  Use 0.5 or the shipped thresholds. The difference is within noise.
- **Clean audio did not get worse on the test set. It got better.**
  - Hindi clean: 90.26 → **94.08** (+3.8, n = 1284). eot_v0 is right on 78 clips where Smart Turn is wrong, and wrong on 29 where it
    is right (McNemar p = 2e-6). Cut-in falls from 17.7% to 6.0%. Missed ends rise from 2.4% to 5.8%, so eot_v0 trades a few late
    replies for far fewer interruptions.
  - English clean, all 7820 clips: 94.26 → **95.40** (+1.1; 190 vs 101 discordant clips, p = 2e-7). Cut-in falls from 7.4% to 5.5%,
    and missed ends move from 4.0% to 3.7%.
  - During training, clean dev accuracy slipped 1-2 points (Hindi 97.1 → 95.1). That doesn't show on the test set. The
    published model already scored 97.1 on clean Hindi dev against 90.3 on test, so the dev clips were probably in upstream's
    training data. Judge clean-audio cost on the test set.
- **English gains are smaller but consistent.** eot_v0 scores 90.6-94.3 everywhere, against Smart Turn's 81.7-93.0. The gain is
  +8.9 at 5 dB SNR and +8.2 on the mixed chain.
- **The remaining weak spot is heavy noise.** At 5 dB SNR, Hindi still loses 5.5 points against clean, and cut-in is 15%. At
  10 dB it loses 2.9. The next round should address noise first (more low-SNR training, real babble), and then validate on real calls.

## Next

1. Validate on real 8 kHz phone recordings of consenting speakers (plan §6). Synthetic TTS and simulated lines can overstate
   the gains.
2. Quantise eot_v0 to int8 like Smart Turn's shipped model and check that accuracy holds and CPU latency is acceptable.
3. Wire eot_v0 into the Pipecat demo as the turn analyzer.
