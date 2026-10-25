# pehleaap

*Pehle aap* ("after you"): turn-taking for **Hinglish voice agents on 8 kHz phone lines**.
It covers end-of-turn detection, barge-in vs backchannel classification, and hold intent ("ek minute").

This is a **training recipe plus a benchmark** that a voice platform can rerun on its own call data.
It is not a production model. The full plan, including the fact-check of prior work, the architecture
and the week-by-week schedule, is in [`docs/PLAN.md`](docs/PLAN.md).

## Status (week 1)

| Piece | State |
|---|---|
| Smart Turn v3.2 baseline (ONNX) + reproduction of its published accuracy | `scripts/repro_smart_turn.py` |
| Clip metrics (Smart Turn definitions) and causal event-level EOT metrics (cut-in vs response gap) | `pehleaap/eval/` |
| Two-track clock alignment and drift checks | `pehleaap/audio/align.py`, `scripts/check_alignment.py` |
| Consent forms (EN/HI), recording protocol, scenario cards, fake-PII cards | `docs/` |
| Community Hinglish end-of-turn models (deveshu, abhishek, codewithmoin, Chhotu) behind one interface | `pehleaap/baselines/turn_models.py` |
| Baselines on clean vs 8 kHz G.711 / AMR-NB audio (heavy: runs on the GPU box) | `scripts/eval_clip_baselines.py`, runbook in `docs/machines.md` |
| Real-time phone line (G.711 bit-identical to ffmpeg, ~6 ms delay) | `pehleaap/telephony/realtime.py` |
| Smart Turn v3.2 in PyTorch, loaded from the published fp32 ONNX weights (matches it within 4e-6), as the starting point for fine-tuning | `pehleaap/models/smart_turn.py` |
| Cross-fitted decision-threshold calibration: how much of a model's phone-line drop a re-picked threshold recovers, from saved predictions | `pehleaap/eval/calibration.py`, `scripts/calibrate_thresholds.py` |
| Phone-line fine-tuning of Smart Turn: English + Hindi training clips through the augmentation chain on the fly, resumable checkpoints, ONNX export scored by the baseline eval (`--onnx`). Smoke-tested on the Dev box; the real run is GPU box Job 4 | `scripts/prepare_eot_train.py`, `scripts/train_eot.py`, `pehleaap/train/`, `configs/train_eot.yaml` |
| Phone-line augmentation chain: room reverb, noise at an SNR bucket, speakerphone echo, AGC, clipping, band-pass, codecs (with G.711 tandem), packet loss + jitter with concealment. Keeps timing within 1 ms; synthetic noise until noise datasets are added | `pehleaap/telephony/pipeline.py`, `configs/augment.yaml`, `scripts/augment_examples.py` |
| Pipecat demo: browser call → phone line → Silero VAD → turn model → ding, with a live switch between Smart Turn v3.2 and eot_v0 (`/switch`); the other model's verdict is logged at every pause | `demo/bot.py`, `pehleaap/pipecat_ext/` |
| Labeling engine v0: Silero VAD → IPUs → SHIFT / HOLD / BC / INTERRUPT / HOLD_REQ / NOISE_ECHO events (JSONL). With `--asr`, Whisper large-v3 transcribes each speech segment so the backchannel, hold-request and digit rules apply; speakerphone echo of the other speaker is cut out frame by frame and becomes NOISE_ECHO | `pehleaap/labeling/`, `configs/labeling.yaml`, `scripts/label_session.py`, `scripts/make_tts_session.py` |
| Label checking by ear: a local page that plays each event (agent left, customer right) with both speakers' loudness drawn; one key per verdict, 10% double-annotated for Cohen's kappa, verified events written per session | `scripts/verify_app.py`, `pehleaap/labeling/verify.py` |
| Real human Hindi end-of-turn test from FLEURS read speech (finished sentence vs mid-sentence pause), scored like Smart Turn's test set | `scripts/make_fleurs_eot.py` |
| Outreach drafts: WhatsApp invite (English and Hinglish), booking sheet, HingCoS access email; pilot-day checklist in the recording protocol | `docs/recruiting.md`, `docs/recording_protocol.md` |
| After a recording day, one command per session: consent check (stops without it), audio health, clock drift between the sync beeps, labels, checksums, next steps | `scripts/process_session.py` |
| Turn models on recorded sessions: each labelled pause replayed to every model at fixed moments after it starts, through each phone condition; cut-in rate vs response gap, against the silence-timeout bot | `scripts/eval_sessions.py`, `pehleaap/eval/sessions.py`, `pehleaap/telephony/conditions.py` |

## Setup (both machines, Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # CUDA 12.6 PyTorch + everything else
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\env_check.py                 # specs, dedicated VRAM, codecs
```

You also need an **ffmpeg full build** that includes `libopencore_amrnb`, for the AMR-NB telephony codec,
e.g. the gyan.dev `full_build-shared` build. Put it on `PATH` or set `PEHLEAAP_FFMPEG` to `ffmpeg.exe`.

## Compute

| | Dev box (laptop) | Kaggle (from Oct 29, 2026) |
|---|---|---|
| Does | code, tests, labeling UI, eval on small slices, Pipecat demo, recording checks | all training, batch ASR, augmentation precompute, full eval sweeps |
| Code | edit, commit, push | jobs clone the pushed commit (`scripts/kaggle_job.py`) |
| Data | masters `data/raw` (recordings) | private Kaggle datasets, checked against the manifests |

The GPU box (RTX 4000 SFF Ada, 20 GB) runs the main evaluations and fine-tuning for as long as it is available (end date unknown); runbooks in `docs/machines.md`. Kaggle is the fallback.
Audio never goes in git. `data/manifests/*.jsonl` (sha256 per file) does, and
`python scripts\env_check.py --verify-data` proves a machine or job holds the same data.

**Shared Claude Code memory.** Claude Code's memory is normally per machine. Here it lives in git instead:
`CLAUDE.md` plus `memory/`. Run `powershell -ExecutionPolicy Bypass -File scripts\setup_claude_memory.ps1` once
per machine; a SessionStart hook (`.claude/settings.json`) pulls the latest commit when a session starts.
Commit and push before switching machines.

## Week-1 commands

```powershell
.venv\Scripts\python scripts\repro_smart_turn.py               # must PASS before trusting the harness
.venv\Scripts\python scripts\make_fake_pii_cards.py --sessions 60
.venv\Scripts\python scripts\check_alignment.py trackA.flac trackB.flac
.venv\Scripts\python scripts\label_session.py data\raw\sessions\S017   # -> data\labels\S017\events.jsonl
```

## Demo (Dev box)

Pipecat pins older onnxruntime/soundfile/soxr than the research stack, so the demo has its own venv:

```powershell
python -m venv .venv-demo
.venv-demo\Scripts\python -m pip install -r requirements-demo.txt
.venv-demo\Scripts\python -m pip install -e . --no-deps
.venv-demo\Scripts\python -m pytest tests\test_pipecat_ext.py
.venv-demo\Scripts\python demo\bot.py        # open http://localhost:7860, allow the mic, talk
```

Speak a sentence and stop: the bot dings when the turn model decides you've finished. A ding mid-sentence
is a cut-in. Open http://localhost:7860/switch in a second tab to switch between Smart Turn v3.2 (the
baseline) and eot_v0 (ours; needs `weights\eot_v0\model.onnx`) mid-call; the terminal shows both models'
verdicts at every pause. Load other models with `$env:PEHLEAAP_TURN_MODELS` (registry names or folders
under `weights\`), pick the first decider with `$env:PEHLEAAP_TURN_MODEL`, and turn the simulated phone
line off with `$env:PEHLEAAP_PHONE_LINE = "0"`.

## Licences and credits

Smart Turn (pipecat-ai/smart-turn) is BSD-2-Clause, and its test data is CC-BY-4.0. Prior work this project
builds on or compares against is listed in `docs/PLAN.md` §1.
