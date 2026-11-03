# English check on TurnBench dev, and our automatic labels against human ones (GPU box, Nov 4 2026)

TurnBench dev (mundo-ai/turn-benchmark-dev; Mundo AI, Dataset Public License v1.0: NonCommercial, no voice cloning)
has 38 real English two-person conversations, 7.3 h in all. Each person has their own 48 kHz track, and three human
annotators labelled every speaker's turns, backchannels and interruptions. We use it for evaluation only. Its audio
and transcripts stay out of git.

It answers two questions before our own recordings exist:
1. Does eot_v1, tuned for Hinglish phone calls, still handle English conversation (plan §6, "English check")?
2. How good are `label_session.py`'s automatic SHIFT/HOLD labels, which our recorded sessions start from?

## Pipeline: the one our recordings will use

```
python scripts/turnbench_to_sessions.py          # 38 session folders: both FLAC tracks + the human annotations
python scripts/label_session.py "data/external/turnbench-dev/sessions/TB*" --out runs/turnbench_labels   # VAD + duration rules
python scripts/turnbench_verify.py "runs/turnbench_labels/TB*"          # human majority -> events_verified.jsonl
python scripts/eval_sessions.py "runs/turnbench_labels/TB*" --channels agent customer
    --models eot_v1 smart_turn_v3.2_int8 weights/pipecat-ai__smart-turn-v3/smart-turn-v3.2-gpu.onnx
    --conditions clean --threads 20 --out turnbench_dev_eval
python scripts/paired_bootstrap.py runs/turnbench_dev_eval "runs/turnbench_labels/TB*" --a eot_v1 --b <model>
```

Both speakers' pauses are scored. Speaker 1 is the "agent" track and speaker 2 the "customer" track; the names don't
matter here.

## 1. Automatic labels against the human majority

`turnbench_verify.py` gives each automatic pause the verdict of at least two of the three annotators:
- **HOLD:** the pause lies inside the speaker's annotated turn, or they start a new turn before the other person does.
- **SHIFT:** their turn ends at the pause and the other person's turn comes next.

Backchannels, laughter, fillers and channel bleed never count as taking the turn.

| automatic -> human | HOLD | SHIFT | no majority |
|---|---|---|---|
| HOLD (4,489) | 4,292 | 56 | 141 |
| SHIFT (2,088) | **488** | 1,129 | 471 |

- 5,965 of 6,577 pauses (91%) get a majority verdict. The automatic label agrees on 90.9% of them.
- **The errors go one way: 30% of the automatic SHIFTs the humans judged are HOLDs** (488 of 1,617). Of six sampled cases,
  two were long backchannels ("Aww", "Mm-hmm", over the 0.6 s no-transcript limit), three were overlaps where the
  speaker kept the floor, and one was ambiguous. Automatic HOLDs are almost always right (1.3% are SHIFTs).
- **This changes results.** Scored on the same pauses with the automatic labels, every model's cut-in rate at a 0.5 s
  matched gap reads 2-3 points lower, and the order flips:

  | cut-in @ 0.5 s gap | human labels | automatic labels |
  |---|---|---|
  | eot_v1 | 13.5% | 11.5% |
  | Smart Turn fp32 | 13.9% | 11.0% |
  | Smart Turn int8 | 14.7% | 11.8% |

  So recorded sessions need `verify_app.py` before any claim (`docs/prereg.md` §5). Duration rules alone aren't enough.
  These sessions ran without a transcript; `--asr` adds the backchannel word list, which may fix part of it (not tried).

The no-majority rows are mostly pauses after speech that wasn't a turn, such as a backchannel or laughter. They are
left out.

## 2. English end-of-turn on real conversation (clean 16 kHz, human-verified pauses)

5,965 pauses from 26 speakers (1,185 turn ends, 4,780 holds). Each model is asked 0.2, 0.3, 0.5, 0.8, 1.2 and 2 s into
each pause, with a 3 s hard timeout (`runs/turnbench_dev_eval`, `week2_turnbench_dev_results.json`).

| Model | Cut-in @ 0.5 s gap | Cut-in @ 0.8 s gap | At its 0.5 threshold: cut-in / median gap / missed |
|---|---|---|---|
| **eot_v1** | **13.5%** | **10.8%** | 12.7% / 0.80 s / 18.0% |
| Smart Turn v3.2 fp32 | 13.9% | 12.4% | 15.0% / 0.50 s / 15.9% |
| Smart Turn v3.2 int8 (Pipecat default) | 14.7% | 13.0% | 16.4% / 0.30 s / 16.0% |
| Silence timeout | 41.6% | 19.5% | - |

Paired differences, eot_v1 minus each rival (negative = eot_v1 cuts in less). The 95% intervals resample the 26
speakers (10,000 draws, seed 0):

| | @ 0.5 s gap | @ 0.8 s gap |
|---|---|---|
| eot_v1 − Smart Turn fp32 | −0.4 points [−1.7, +0.6] | −1.5 points [−2.3, +0.4] |
| eot_v1 − Smart Turn int8 | −1.2 points [−2.6, +0.6] | **−2.2 points [−3.8, −0.2]** |

(`week2_turnbench_dev_bootstrap.json`.)

**Reading.**
- **English hasn't regressed.** At matched speed eot_v1 cuts in no more than Smart Turn, which trained on English.
  Only one gap clears zero: against Smart Turn int8 at 0.8 s. The other three intervals include zero, so on English
  conversation eot_v1 is level with Smart Turn, not ahead of it.
- Every model beats the silence timeout by far at 0.5 s (14% against 42%).
- At its 0.5 threshold eot_v1 waits longer than Smart Turn (median 0.8 s against 0.3-0.5 s). Its fine-tune leans
  towards "not finished" on English conversation, so a deployment wanting fast English replies would lower its threshold.
- This is clean studio audio, not a phone line. G.711 and the other conditions weren't run, to keep the run to about
  20 min. Each model recomputes the same Whisper features per window; sharing them would cut the cost (not done).

**Caveats.**
- TurnBench dev was scored before `docs/prereg.md` froze, so this is exploratory, not a confirmatory test.
- This isn't the official TurnBench metric (recall at FPR ≤ 0.15 with their causal scorer). It is our pause-level
  protocol on their audio and labels.
