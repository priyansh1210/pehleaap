---
name: dev-environment
description: "Per-machine toolchain locations, venvs, and known Windows/network gotchas (cp1252 output, PowerShell 5.1 stderr, flaky downloads, 8 GB RAM)"
metadata:
  type: reference
---

**Dev box** (repo `D:\pehleaap`):
- Python 3.11.9 at `D:\python`; research venv `.venv` (torch 2.14.1+cu126, pins in requirements.txt).
- Demo venv `.venv-demo` (Pipecat 1.13, onnxruntime 1.24, no torch); `demo\bot.py` serves http://localhost:7860.
- ffmpeg 9.0.2 gyan.dev full_build-shared at `D:\tools\ffmpeg-9.0.2-full_build-shared\bin` (has libopencore_amrnb, libgsm, libopus).
  Not on PATH: `pehleaap/paths.py::ffmpeg_bin()` falls back to it.
- Smart Turn v3.2 test set (4.8 GB, CC-BY-4.0) in `data/external/smart-turn-data-v3.2-test`; weights in `weights/` (both gitignored).
- No winget, no 7-Zip, no gh. Use `curl.exe` and `Expand-Archive` (zip builds).

**GPU box** (maybe until 2026-10-28; repo `C:\Users\FITT\Desktop\PRIYANSH!!\pehleaap`): evals, fine-tuning, quantisation; 24 cores, 64 GB RAM. Python 3.12.10 via `py`
(venv `.venv`, pins install fine, 64 tests pass). ffmpeg 9.0.1 gyan.dev full build from winget is on PATH (amrnb/gsm/opus present),
so no `~\tools` install. Has winget. The auto-mode classifier blocked running `setup_claude_memory.ps1` for me: the user runs it.

**Kaggle** (from 2026-10-29): CLI `kaggle==2.2.4` in `.venv` (pinned in requirements.txt; 2.2.5 skipped as only 2 days old).
Auth = API token from kaggle.com/settings/api saved as `%USERPROFILE%\.kaggle\access_token` (or `.txt`, or env KAGGLE_API_TOKEN),
never in git. Importing `kaggle` auto-authenticates and prints login help on failure; kaggle_job.py silences that import.

Gotchas:
- Windows PowerShell 5.1 wraps any stderr line (e.g. the HF "unauthenticated requests" warning) as `NativeCommandError`. This is not a failure; check the exit code.
- Windows pipes and log files default to cp1252: any script printing non-ASCII (the Delta sign, box-drawing in Pipecat's runner banner)
  crashes when redirected. Scripts call `sys.stdout.reconfigure(encoding="utf-8")`; do the same in new entry points.
- The network is flaky: github.com DNS failed once, a gyan.dev download aborted, and pip hung for 15+ min. Use `curl -C - --retry`,
  and run pip with `--timeout 30 --retries 10 --progress-bar off`, logging to a file in the background.
- A curl-downloaded parquet shard was silently bad; the sha256 check against the HF tree API caught it. Keep checksum checks.
- Dev box has 8 GB RAM: Claude Code killed a ~2 GB eval run plus a pip install for low memory (with Chrome open). Run heavy jobs one at
  a time, and full evals on Kaggle (the GPU box until Oct 28).

- GPU box: a long job started in the user's own PowerShell window died silently mid-run (Job 3, 2026-10-24, no traceback, no
  low-memory event); run long jobs from Claude's session in the background, resumable (eval scripts skip saved shards).
- Logs written with PowerShell `*>` are UTF-16: read them with `grep -a` or iconv.
- Windows DataLoader workers (spawn) need an `if __name__ == "__main__":` guard even in scratch scripts, or the run hangs with no
  workers and no error (2026-10-25: a hung rescoring process, PID 1808 on the GPU box; auto mode blocked Stop-Process, so the user
  kills such leftovers).
- onnxruntime's histogram (entropy/percentile) calibrator keeps every activation of every clip in RAM (~150 MB per Smart Turn clip):
  1024 clips ran out of memory even with 64 GB; `CalibMaxIntermediateOutputs` only applies to MinMax. `pehleaap/models/quantize.py`
  feeds it 32 clips at a time.

Related: [[two-machine-setup]], [[pehleaap-status]].
