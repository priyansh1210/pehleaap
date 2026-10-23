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

**GPU box** (repo `C:\Users\FITT\pehleaap`, may differ): not set up yet. Runbook in docs/machines.md puts ffmpeg under `~\tools`,
which `ffmpeg_bin()` also checks. Fill in real paths here once it is set up.

Gotchas:
- Windows PowerShell 5.1 wraps any stderr line (e.g. the HF "unauthenticated requests" warning) as `NativeCommandError`. This is not a failure; check the exit code.
- Windows pipes and log files default to cp1252: any script printing non-ASCII (the Delta sign, box-drawing in Pipecat's runner banner)
  crashes when redirected. Scripts call `sys.stdout.reconfigure(encoding="utf-8")`; do the same in new entry points.
- The network is flaky: github.com DNS failed once, a gyan.dev download aborted, and pip hung for 15+ min. Use `curl -C - --retry`,
  and run pip with `--timeout 30 --retries 10 --progress-bar off`, logging to a file in the background.
- A curl-downloaded parquet shard was silently bad; the sha256 check against the HF tree API caught it. Keep checksum checks.
- Dev box has 8 GB RAM: Claude Code killed a ~2 GB eval run plus a pip install for low memory (with Chrome open). Run heavy jobs one at
  a time, and full evals on the GPU box.

Related: [[two-machine-setup]], [[pehleaap-status]].
