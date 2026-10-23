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
| Two-machine helpers (sync, remote jobs, checkpoint fetch) | `scripts/*.ps1`. **Untested until the GPU box is set up.** |

## Setup (both machines, Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # CUDA 12.6 PyTorch + everything else
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\env_check.py                 # specs, dedicated VRAM, codecs
```

You also need an **ffmpeg full build** that includes `libopencore_amrnb`, for the AMR-NB telephony codec,
e.g. the gyan.dev `full_build-shared` build. Put it on `PATH` or set `PEHLEAAP_FFMPEG` to `ffmpeg.exe`.

## Two machines

| | Dev box (laptop) | GPU box (RTX 4060) |
|---|---|---|
| Does | code, tests, labeling UI, eval on small slices, Pipecat demo, recording checks | all training, batch ASR, augmentation precompute, full eval sweeps |
| Code | edit, commit, push | `git pull` only (`scripts/run_job.cmd` pulls automatically) |
| Data | masters `data/raw` (recordings) | receives `data/raw`, produces derived data |

```powershell
$env:PEHLEAAP_GPU_SHARE = "\\GPUBOX\pehleaap-data"   # or a portable SSD path
.\scripts\sync_data.ps1 raw                          # push recordings
.\scripts\remote.ps1 -Detach train.train_eot --config configs\train_eot.yaml
.\scripts\fetch_ckpt.ps1 <run_id>                    # bring back ONNX + metrics
```

Audio never goes in git. `data/manifests/*.jsonl` (sha256 per file) does, and
`python scripts\env_check.py --verify-data` proves both machines hold the same data.

## Week-1 commands

```powershell
.venv\Scripts\python scripts\repro_smart_turn.py               # must PASS before trusting the harness
.venv\Scripts\python scripts\make_fake_pii_cards.py --sessions 60
.venv\Scripts\python scripts\check_alignment.py trackA.flac trackB.flac
```

## Licences and credits

Smart Turn (pipecat-ai/smart-turn) is BSD-2-Clause, and its test data is CC-BY-4.0. Prior work this project
builds on or compares against is listed in `docs/PLAN.md` §1.
