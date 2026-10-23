# Machines

Output of `python scripts/env_check.py --markdown` on each machine. Re-run after driver or package changes.

## Dev box (laptop)

| Item | Value |
|---|---|
| host | PRIYANSH |
| os | Windows 10 (10.0.22621) |
| python | 3.11.9 |
| cpu | 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz |
| cpu_cores | 4 physical / 8 logical |
| ram_gb | 7.8 |
| disk_free_gb | 452 |
| gpu (nvidia-smi) | NVIDIA GeForce GTX 1650, 4096 MiB, 592.82 |
| torch | 2.14.1+cu126 |
| cuda (torch build) | 12.6 |
| gpu dedicated VRAM (GB) | 4.0 |
| gpu compute capability | 7.5 |
| bf16 native | False |
| numpy | 2.4.6 |
| onnxruntime | 1.31.0 |
| transformers | 5.20.2 |
| datasets | 5.1.0 |
| soxr | 1.1.0 |
| soundfile | 0.14.0 |
| pipecat | not installed |
| ffmpeg | ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers |
| ffmpeg codecs | libopencore_amrnb, libgsm, libopus, pcm_mulaw, pcm_alaw |
| git commit | no commits |

## GPU box (RTX 4060)

Pending: on that machine, clone the repo, create the venv, then run `python scripts\env_check.py --markdown` and paste the table here. The `gpu dedicated VRAM (GB)` row is the number training batch sizes are based on.

### First-time setup and the first job (PowerShell on the GPU box)

Needs git (with Git Bash, which Claude Code also uses) and Python 3.11.9 (same as the Dev box). Everything else
comes from the repo. The repo goes in your user folder, `C:\Users\FITT\pehleaap` (`$HOME\pehleaap`); if you put it
elsewhere, set `PEHLEAAP_GPU_REPO` to that path on the Dev box for `remote.ps1` / `fetch_ckpt.ps1`.

```powershell
git clone https://github.com/priyansh1210/pehleaap $HOME\pehleaap
cd $HOME\pehleaap
python -m venv .venv
.venv\Scripts\python -m pip install --timeout 30 --retries 10 -r requirements.txt

# ffmpeg full build with AMR-NB. Unzipped to ~\tools, pehleaap finds it without PEHLEAAP_FFMPEG.
curl.exe -L -C - --retry 5 -o $env:TEMP\ffmpeg.zip https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip
Expand-Archive $env:TEMP\ffmpeg.zip $HOME\tools

# Shared Claude Code memory: point this machine's auto memory at the repo's memory\ folder (once).
powershell -ExecutionPolicy Bypass -File scripts\setup_claude_memory.ps1

.venv\Scripts\python -m pytest                              # all pass (the pipecat tests skip here)
.venv\Scripts\python scripts\env_check.py --markdown        # paste the table above
```

Then start Claude Code in `$HOME\pehleaap`. It loads `CLAUDE.md` and the shared `memory\`, and its SessionStart hook
pulls the latest commit each time. Approve the project hook when Claude Code asks.

**Job 1: end-of-turn baselines, clean vs phone line** (Smart Turn v3.2 + 4 community Hinglish models
on all 31.5k test clips; 2.5 h on the laptop, too heavy for its 8 GB RAM). The first run downloads the
4.8 GB test set, checking each shard's sha256.

```powershell
.venv\Scripts\python scripts\eval_clip_baselines.py --models chhotu_int8 --limit-shards 1 --out smoke   # smoke test, a few minutes
.venv\Scripts\python scripts\eval_clip_baselines.py --threads 8 *> runs\clip_baselines.log              # --threads = physical cores
```

Each shard's predictions are saved as it finishes (`runs\clip_baselines\shards\`). If the run stops
(sleep, reboot, Ctrl+C), rerun the same command and it resumes. When it is done, bring back
`runs\clip_baselines\report.md` and `results.json` (small) to the Dev box, e.g. into `docs\results\`.
