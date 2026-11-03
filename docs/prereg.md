# Preregistration: pehleaap end-of-turn evaluation

**Status: DRAFT (Nov 4 2026).** It freezes when Priyansh merges the PR that removes the word DRAFT, and that must
happen before any confirmatory test below is run. After the freeze this file changes only in **Deviations**, with a
date and a reason.

## 1. What is confirmatory

A test set is confirmatory only if no model was tuned or scored on it before the freeze:

| Test set | Status at the freeze | Use |
|---|---|---|
| Recorded conversations, **test split** (`benchmark/splits/test.txt`) | not recorded yet | **confirmatory** (H1-H4) |
| **IndicVoices Hindi**, hand-labelled (`scripts/make_indicvoices_eot.py`): 300 clips, one per speaker | clips cut, not labelled, never scored | **confirmatory** (H5) |
| FLEURS Hindi test/dev | scored many times; eot_v1 trained on FLEURS train | exploratory |
| Smart Turn's test set (Hindi = TTS) | scored many times | exploratory |
| TurnBench dev (English) | scored Nov 4 2026 | exploratory (English check) |

## 2. Systems

- **Ours: eot_v1**, the released `weights/eot_v1/model.onnx` (models-v1 release). If another model replaces it, it is
  named here in Deviations before its first confirmatory run.
- **Main reference: Smart Turn v3.2 int8** (`smart-turn-v3.2-cpu.onnx`, Pipecat's default), plus Smart Turn v3.2 fp32.
- Community Hinglish models (clip tests only): codewithmoin_8s, deveshu_fp32, abhishek_tiny_attn, chhotu_int8.
- **Silence timeout** (sessions only): no model, the bot answers after a fixed silence.

Each model uses its model-card threshold where a fixed threshold is needed (0.5 for ours and Smart Turn).

## 3. Hypotheses

### H1 (primary): fewer cut-ins on real conversations at the same speed

On the test-split conversations (customer-channel pauses, human-verified labels, **G.711 μ-law**), eot_v1's **cut-in
rate at a matched median response gap of 0.5 s** is lower than Smart Turn v3.2 int8's.

- **Metric:** `scripts/eval_sessions.py`, the "Cut-in @ 0.5 s gap" column. Each model is asked 0.2, 0.3, 0.5, 0.8,
  1.2 and 2.0 s into each pause. The threshold is swept from 0.05 to 0.95, and the hard timeout is 3 s. Cut-in means
  the model answered in a pause where the speaker carried on.
- **Test:** the paired difference (eot_v1 − Smart Turn int8) with a 95% interval from `scripts/paired_bootstrap.py`
  (`pehleaap/eval/bootstrap.py`). The bootstrap resamples the evaluated speakers with replacement, 10,000 draws,
  seed 0. **H1 holds if the interval's upper bound is below 0.**
- **Minimum data:** at least 6 test speakers and 200 HOLD pauses. With less, every result is reported as exploratory.

### H2-H4 (secondary, same data and test)

- **H2:** H1 at a 0.8 s matched gap.
- **H3:** eot_v1 vs the silence timeout at 0.5 s.
- **H4:** eot_v1 vs Smart Turn v3.2 fp32 at 0.5 s.
- The H1 comparison is also run under clean audio, AMR-NB 7.95 and the mixed chain.

Secondary results are reported with their intervals and labelled secondary. No multiplicity correction is claimed for
them.

### H5 (confirmatory, clips): held-out spontaneous Hindi

On the IndicVoices clips labelled finished or not finished (unsure left out), clean audio, each model at its card
threshold: eot_v1's accuracy is higher than codewithmoin_8s's, the best rival on FLEURS.

- **Test:** exact McNemar on the paired decisions, α = 0.05, as in `week1_fleurs_significance.md`. There is one clip
  per speaker, so clips don't share speakers.
- **Secondary:**
  - AUC and cut-in rate for every model.
  - The same test against Smart Turn int8.
  - G.711 μ-law and AMR-NB 7.95.
- **Labels:** one annotator (Priyansh), labelling with `label.html` before any model scores the clips, blind to the cut
  (segment end or pause) and to every model's output. If a second person labels a subset, Cohen's κ is reported.

### English check (exploratory)

TurnBench dev was scored before the freeze, so it supports no confirmatory claim. Its numbers are reported as an
English regression check against Smart Turn v3.2 fp32.

## 4. Decision policy θ(t)

- The hypotheses above use **one threshold per model**: the swept curve for matched-gap metrics, the card threshold for
  fixed-threshold tables.
- θ(t) is a separate, secondary result: one threshold per checkpoint, never rising as the silence grows.
  `eot_events.fire_times` already accepts a threshold per checkpoint.
- **Rule, fixed now:** on the **dev** split only, pick the θ(t) that gives the fewest cut-ins with a median gap of at
  most the profile's target. The grid is 0.1 to 0.9 in steps of 0.1 per checkpoint, non-increasing. Profile targets
  (proposed, fixed at the freeze): `collections_kyc` 0.8 s, `support` 0.5 s, `quick_faq` 0.3 s. The chosen θ(t) is
  then applied unchanged to test.
- θ(t) results can't change H1-H5.

## 5. Data rules

- **Splits:** speaker-disjoint, about 70/10/20 by recorded minutes (`pehleaap/data/splits.py`). Sessions that share a
  speaker go to one split together, and PSTN sessions go to test. The split files are hash-locked in
  `benchmark/splits/lock.json` at the freeze.
- **Labels:** `label_session.py` labels automatically, then a person checks **every test-split SHIFT/HOLD pause** with
  `verify_app.py --all`. Only verified pauses enter H1-H4.
  - On TurnBench dev, 30% of the automatic SHIFTs were HOLDs to the human annotators
    (`docs/results/week2_turnbench_dev.md`). Unverified labels would hide cut-ins.
- **Frozen at the freeze commit:** `configs/labeling.yaml`, `configs/augment.yaml` (the phone-line conditions), the
  scorer (`eval_sessions.py`, `eot_events.py`, `bootstrap.py`, `paired_bootstrap.py`), and `CHECKPOINTS_S`.
- **Excluded:**
  - Pauses shorter than 0.2 s (the labelling config).
  - Speakerphone echo (`noise_echo`) events.
  - Sessions that fail `check_alignment.py`.
  - IndicVoices clips labelled unsure.

## 6. Reporting

- Every hypothesis is reported whatever the outcome, with its interval, alongside failure cases from the error
  analysis.
- Claim template: "On N held-out speakers (M pauses), simulated G.711 phone line, eot_v1 cut in X% of pauses where the
  speaker carried on, against Y% for Smart Turn v3.2 int8, at a 0.5 s median response gap (difference D, 95% CI
  [lo, hi], speaker-clustered bootstrap)."

## Deviations

None yet.
