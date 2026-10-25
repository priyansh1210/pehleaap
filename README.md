# pehleaap

*Pehle aap* ("after you"): turn-taking for **Hinglish voice agents on 8 kHz phone lines**.

At every pause, a voice agent has to decide whether the caller has finished. Hindi callers often pause mid-sentence, before
the verb or to look something up ("mera account number hai... ek minute..."), and a phone codec throws away half the audio
bandwidth the model listens to. Smart Turn v3.2, the open end-of-turn model most voice stacks use, learned from wideband audio
and synthetic (TTS) Hindi. On real Hindi voices over a G.711 phone line, the file Pipecat ships calls **73% of mid-sentence
pauses finished**, so the bot talks over the caller.

This repo has three parts:

- **eot_v1**, the model: Smart Turn v3.2 fine-tuned for phone lines and real Hindi voices. It's the same size and speed as
  Smart Turn (a 32 MB ONNX file, about 17 ms per decision on 4 threads of a workstation CPU), and the file drops straight
  into Pipecat's own analyzer.
- **A benchmark** that scores any end-of-turn model on clean audio and on simulated phone lines (G.711, AMR-NB, noise, echo,
  packet loss), on single clips and on recorded two-person conversations.
- **The recipe** to rerun all of it on your own call data: phone-line augmentation, the training script, labelling tools and a
  recording protocol.

## Results (v1, October 2026)

### Real human Hindi voices

FLEURS Hindi read speech: 1,046 clips (657 finished sentences, and 389 cut 0.2 s into a pause the reader made mid-sentence).
None of these clips were trained on. Each cell is accuracy (%) at the model's own threshold, with the **cut-in rate** in
brackets: the share of mid-sentence pauses the model called finished, where the bot would talk over the caller.

| Audio | **eot_v1 (ours)** | codewithmoin | deveshu | abhishek | Smart Turn fp32 | Smart Turn int8 | chhotu |
|---|---|---|---|---|---|---|---|
| clean 16 kHz | **88.8** (18%) | 81.6 (37%) | 80.9 (36%) | 79.5 (49%) | 75.0 (36%) | 72.8 (60%) | 67.2 (55%) |
| G.711 μ-law (landline) | **86.7** (23%) | 80.7 (39%) | 79.9 (39%) | 78.3 (51%) | 71.8 (49%) | 69.4 (73%) | 54.4 (40%) |
| AMR-NB 7.95 kbit/s (mobile) | **86.9** (25%) | 80.5 (41%) | 77.5 (43%) | 78.4 (49%) | 70.6 (57%) | 68.4 (77%) | 60.2 (62%) |
| 10 dB noise + AMR-NB | **83.4** (33%) | 76.3 (52%) | 71.1 (47%) | 76.8 (45%) | 67.7 (66%) | 65.2 (84%) | 64.1 (92%) |
| mixed phone chain | **82.8** (31%) | 75.1 (54%) | 70.6 (41%) | 76.8 (47%) | 68.6 (59%) | 65.7 (79%) | 62.9 (90%) |
| AUC, clean / mixed | **0.93 / 0.87** | 0.82 / 0.75 | 0.81 / 0.73 | 0.86 / 0.80 | 0.79 / 0.67 | 0.77 / 0.60 | 0.67 / 0.52 |

- eot_v1 leads every model in every condition, by 6.0 to 7.2 points over the next best (paired McNemar p ≤ 3e-8 against
  codewithmoin).
- codewithmoin, deveshu, abhishek and chhotu are community Hinglish end-of-turn models published on Hugging Face.
- Smart Turn int8 is the file Pipecat ships by default. On real voices it cuts in far more often than Smart Turn's own fp32 file.
- The "mixed phone chain" draws one random chain per clip: room reverb, noise, AGC, clipping, codecs, packet loss and jitter.

### Smart Turn's own test set (all its Hindi clips are TTS)

| Audio | Hindi: **eot_v1** | Hindi: codewithmoin | Hindi: deveshu | Hindi: Smart Turn int8 | English: **eot_v1** | English: Smart Turn int8 |
|---|---|---|---|---|---|---|
| clean 16 kHz | 93.7 (7%) | **93.9** (7%) | 93.2 (6%) | 90.3 (18%) | **94.7** | 94.3 |
| G.711 μ-law | **93.8** (8%) | 93.2 (8%) | 92.6 (8%) | 86.4 (27%) | **92.7** | 92.1 |
| AMR-NB 4.75 kbit/s | **93.2** (8%) | 91.8 (8%) | 90.6 (8%) | 80.5 (40%) | **92.4** | 90.8 |
| 5 dB noise + AMR-NB | **89.3** (15%) | 86.3 (21%) | 77.9 (12%) | 69.5 (63%) | **88.9** | 81.7 |
| mixed phone chain | **91.2** (11%) | 89.4 (13%) | 82.6 (9%) | 73.5 (55%) | **91.3** | 84.3 |

The community models trained on this test set's own training split, so it's in-distribution for them. On clean TTS Hindi,
codewithmoin edges eot_v1 by 0.2 points. Once phone-line damage is added, eot_v1 leads, and on real voices (above) the gap is
6 to 7 points.

### A real two-person conversation

Coming next: the first recorded Hinglish conversation (two consenting adults, scripted role-play with fake details), scored
by `scripts/eval_sessions.py` on its cut-in rate and response gap.

### Read with care

- **FLEURS is read speech, not conversation.** Readers pause at commas, not before a verb. Recorded conversations are the real test.
- FLEURS publishes no speaker IDs. eot_v1 trained on FLEURS's train split, so some test readers may also appear in training.
  The sentences never overlap.
- An 18% cut-in rate on clean real voices is still too high for a product. Threshold tuning and conversational training data come next.
- English slips slightly against our first fine-tune, eot_v0 (−0.7 points clean, p = 3e-4), but stays above Smart Turn everywhere.
- eot_v1 is fp32, the same size and speed as Smart Turn's fp32 file (`smart-turn-v3.2-gpu.onnx`). The test-set table compares
  it with Smart Turn int8 (`smart-turn-v3.2-cpu.onnx`), the file Pipecat ships.
- Full reports, commands and significance tests: [`docs/results/week1_eot_v1_report.md`](docs/results/week1_eot_v1_report.md),
  [`docs/results/week1_fleurs_more_baselines_report.md`](docs/results/week1_fleurs_more_baselines_report.md), and the rest of
  [`docs/results/`](docs/results/).

## Use the model

```powershell
git clone https://github.com/priyansh1210/pehleaap
cd pehleaap
python -m venv .venv-demo                                   # the light install: Pipecat, onnxruntime, no PyTorch
.venv-demo\Scripts\python -m pip install -r requirements-demo.txt
.venv-demo\Scripts\python -m pip install -e . --no-deps
.venv-demo\Scripts\python scripts\get_weights.py            # weights\eot_v1\model.onnx, checksum-verified
```

On Linux or macOS, use `.venv-demo/bin/python` instead.

**In Pipecat**, use it in place of Smart Turn's own file:

```python
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

turn_analyzer = LocalSmartTurnAnalyzerV3(smart_turn_model_path="weights/eot_v1/model.onnx")
```

Then pass it wherever your pipeline takes Smart Turn, for example `TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)` as in
[`demo/bot.py`](demo/bot.py). `tests/test_pipecat_ext.py` checks that Pipecat's analyzer and ours give the same probabilities.

**In Python**, ask it at every pause your VAD reports:

```python
from pehleaap.audio.io import load_audio
from pehleaap.baselines.turn_models import load_any

model, threshold = load_any("eot_v1")                    # weights/eot_v1/model.onnx, threshold 0.5
audio, _ = load_audio("caller_so_far.wav", sr=16000)     # the caller's audio up to the pause
p = model.predict(audio)                                  # P(turn finished), from the last 8 s
print("reply" if p >= threshold else "keep listening")
```

## Demo: hear the difference

Run a browser call through a simulated 8 kHz phone line. The bot dings when the turn model decides you've finished, so a ding
mid-sentence is a cut-in.

```powershell
.venv-demo\Scripts\python demo\bot.py        # open http://localhost:7860, allow the mic, talk
```

Open http://localhost:7860/switch in a second tab to flip between Smart Turn v3.2 and eot_v1 mid-call. The terminal shows both
models' verdicts at every pause, and each turn's gap from the end of speech to the ding.

To change what's loaded:
- `$env:PEHLEAAP_TURN_MODELS`: other models (registry names, or folders under `weights\`).
- `$env:PEHLEAAP_TURN_MODEL`: which one decides first.
- `$env:PEHLEAAP_PHONE_LINE = "0"`: wideband audio, no phone line.

## Score your own model

The benchmark scores any Smart-Turn-format ONNX file (`--onnx name=path`) or any model in
`pehleaap/baselines/turn_models.py`'s `REGISTRY`. This needs the full research install (below).

```powershell
# Smart Turn's test set (downloaded on first use, 4.8 GB), clean and phone conditions
.venv\Scripts\python scripts\eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx mine=path\to\model.onnx `
    --conditions clean g711_mulaw amrnb_7.95 mixed
# Real Hindi voices: build the FLEURS clips once, then score them
.venv\Scripts\python scripts\make_fleurs_eot.py
.venv\Scripts\python scripts\eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin `
    --onnx mine=path\to\model.onnx --conditions clean g711_mulaw amrnb_7.95 noise10_amr7.95 mixed
```

The noise and `mixed` conditions use MUSAN noise. Fetch it once with `scripts\get_noise.py` (0.7 GB kept, about 75 min).

To score **recorded conversations** instead of clips:
1. Record two tracks, one per speaker, following [`docs/recording_protocol.md`](docs/recording_protocol.md) (consent forms included).
2. Run `scripts\process_session.py`: consent check, audio and clock checks, then turn-taking labels.
3. Check the labels by ear with `scripts\verify_app.py`.
4. Run `scripts\eval_sessions.py`. It reports the cut-in rate against the response gap, next to a plain silence-timeout bot.

## Retrain on your own data

```powershell
.venv\Scripts\python scripts\prepare_eot_train.py                    # Smart Turn's English + Hindi training clips (41 GB download, 16 GB kept)
.venv\Scripts\python scripts\make_fleurs_eot.py --splits train --to-store data\external\eot-train --max-pauses 2
.venv\Scripts\python scripts\train_eot.py --config configs\train_eot_v1.yaml --out eot_v1
```

Training takes about 25 minutes on a 20 GB GPU (RTX 4000 Ada, bf16). Every clip passes through the phone-line augmentation
chain on the fly (`configs/augment.yaml`). Checkpoints are resumable, and the best dev step is exported to ONNX.

Your own recordings go into training with `scripts\sessions_to_store.py`, after `scripts\make_splits.py --freeze` has made
speaker-disjoint train, dev and test splits. Real phone-line sessions only ever go in the test split.

## Install (research stack, Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # CUDA 12.6 PyTorch + everything else
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\env_check.py                 # specs, dedicated VRAM, codecs
```

You also need an **ffmpeg full build** that includes `libopencore_amrnb`, for the AMR-NB codec (for example the gyan.dev
`full_build-shared` build). Put it on `PATH`, or set `PEHLEAAP_FFMPEG` to `ffmpeg.exe`. The demo's tests run in their own
venv: `.venv-demo\Scripts\python -m pytest tests\test_pipecat_ext.py`.

## What's in the repo

| Piece | Where |
|---|---|
| Smart Turn v3.2 baseline (ONNX), plus a reproduction of its published accuracy | `scripts/repro_smart_turn.py` |
| Community Hinglish end-of-turn models (deveshu, abhishek, codewithmoin, Chhotu) behind one interface | `pehleaap/baselines/turn_models.py` |
| Clip metrics (Smart Turn's definitions) and causal event-level metrics (cut-in vs response gap) | `pehleaap/eval/` |
| Baselines on clean vs 8 kHz G.711 / AMR-NB audio, a codec sweep, and the full phone chain | `scripts/eval_clip_baselines.py` |
| Phone-line augmentation: room reverb, noise at an SNR bucket, speakerphone echo, AGC, clipping, band-pass, codecs (with G.711 tandem), packet loss + jitter with concealment. Timing stays within 1 ms | `pehleaap/telephony/pipeline.py`, `configs/augment.yaml` |
| Real-time phone line (G.711 bit-identical to ffmpeg, ~6 ms delay) | `pehleaap/telephony/realtime.py` |
| Smart Turn v3.2 in PyTorch, loaded from the published fp32 ONNX (matches it within 4e-6), as the starting point for fine-tuning | `pehleaap/models/smart_turn.py` |
| Fine-tuning: English + Hindi clips through the augmentation chain, resumable checkpoints, ONNX export, int8 copies | `scripts/train_eot.py`, `scripts/quantize_eot.py`, `pehleaap/train/`, `configs/train_eot*.yaml` |
| Cross-fitted threshold calibration: how much of a model's phone-line drop a re-picked threshold wins back | `scripts/calibrate_thresholds.py` |
| Real human Hindi end-of-turn test from FLEURS read speech | `scripts/make_fleurs_eot.py` |
| Labelling engine: VAD → speech units → SHIFT / HOLD / BC / INTERRUPT / HOLD_REQ / NOISE_ECHO events. With `--asr`, Whisper large-v3 transcripts drive the backchannel, hold-request and digit rules; speakerphone echo is cut out frame by frame | `pehleaap/labeling/`, `scripts/label_session.py` |
| Label checking by ear: a local page that plays each event with both speakers' loudness, and 10% double-annotated for Cohen's kappa | `scripts/verify_app.py` |
| One command per recorded session: consent check (stops without it), audio health, clock drift, labels, checksums | `scripts/process_session.py` |
| Turn models on recorded sessions: each pause replayed to every model through each phone condition | `scripts/eval_sessions.py` |
| Speaker-disjoint splits (real phone sessions test-only), frozen with a checksum lock | `scripts/make_splits.py`, `scripts/sessions_to_store.py` |
| Recording kit: consent forms (EN/HI), protocol, scenario cards, fake-PII cards, outreach drafts | `docs/` |
| Pipecat demo with a live model switch | `demo/bot.py`, `pehleaap/pipecat_ext/` |

## Roadmap

v1 is the first working cut. Next:
- More recorded conversations, then conversational fine-tuning on speaker-disjoint splits.
- Telling a barge-in from a backchannel ("haan", "achha") while the agent is talking.
- Hold intent ("ek minute"): wait quietly instead of replying.
- An int8 eot_v1 for CPU-bound deployments.

The full plan, including the fact-check of prior work, is in [`docs/PLAN.md`](docs/PLAN.md).

## Development

The code was developed on a laptop, and the main runs used a GPU box (runbooks: [`docs/machines.md`](docs/machines.md));
Kaggle is the fallback. Audio never goes in git. `data/manifests/*.jsonl` holds a sha256 per file, so
`scripts\env_check.py --verify-data` proves two machines hold the same data. Claude Code's project memory lives in git
(`CLAUDE.md` and `memory/`), so both machines share it.

## Licence and credits

BSD-2-Clause ([`LICENSE`](LICENSE)). eot_v1 is a fine-tune of [Smart Turn](https://github.com/pipecat-ai/smart-turn)
v3.2 (BSD-2-Clause, Daily). It was trained on Smart Turn's training data, FLEURS and MUSAN (all CC BY 4.0). Notices and
credits are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Community models are compared, not redistributed, and each
keeps its own licence.
