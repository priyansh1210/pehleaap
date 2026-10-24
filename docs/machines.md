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

## GPU box (RTX 4000 SFF Ada, 20 GB): until Oct 28, 2026

This box is only available until Oct 28, so it runs Job 1 below and nothing else. From Oct 29, heavy jobs run on
Kaggle (next section). Repo at `C:\Users\FITT\Desktop\PRIYANSH!!\pehleaap`. Python 3.12.10 (the pins install and all
tests pass on it). ffmpeg is a winget 9.0.1 full build on PATH, with the same codecs as the Dev box's 9.0.2.

| Item | Value |
|---|---|
| host | DESKTOP-RTIKELL |
| os | Windows 11 (10.0.26100) |
| python | 3.12.10 |
| cpu | Intel(R) Core(TM) Ultra 9 285 |
| cpu_cores | 24 physical / 24 logical |
| ram_gb | 63.4 |
| disk_free_gb | 102 |
| gpu (nvidia-smi) | NVIDIA RTX 4000 SFF Ada Generation, 20475 MiB, 581.42 |
| torch | 2.14.1+cu126 |
| cuda (torch build) | 12.6 |
| gpu dedicated VRAM (GB) | 20.0 |
| gpu compute capability | 8.9 |
| bf16 native | True |
| numpy | 2.4.6 |
| onnxruntime | 1.31.0 |
| transformers | 5.20.2 |
| datasets | 5.1.0 |
| soxr | 1.1.0 |
| soundfile | 0.14.0 |
| pipecat | not installed |
| ffmpeg | ffmpeg version 9.0.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers |
| ffmpeg codecs | libopencore_amrnb, libgsm, libopus, pcm_mulaw, pcm_alaw |
| git commit | 5f5c935 |

### First-time setup and the first job (PowerShell on the GPU box)

Needs git (with Git Bash, which Claude Code also uses) and Python 3.11.9 (same as the Dev box). Everything else
comes from the repo. The repo goes in your user folder (`$HOME\pehleaap`).

```powershell
git clone https://github.com/priyansh1210/pehleaap $HOME\pehleaap
cd $HOME\pehleaap
python -m venv .venv
.venv\Scripts\python -m pip install --timeout 30 --retries 10 -r requirements.txt

# ffmpeg full build with AMR-NB. Unzipped to ~\tools, pehleaap finds it without PEHLEAAP_FFMPEG.
curl.exe -L -C - --retry 5 -o $env:TEMP\ffmpeg.zip https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip
Expand-Archive $env:TEMP\ffmpeg.zip $HOME\tools

# Shared Claude Code memory (only if you'll run Claude Code here): point auto memory at the repo's memory\ folder.
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

## Kaggle (from Oct 29, 2026)

All training, batch ASR, augmentation precompute and full eval sweeps (plan §3a). Measured by the smoke job
on 2026-10-24 (`--gpu t4`):

| Item | Value |
|---|---|
| gpu | 2× Tesla T4, 14.6 GB dedicated each, compute capability 7.5, no native bf16 (use fp16 AMP) |
| torch | 2.14.1+cu126, CUDA available |
| ram / cpu / disk | 31.3 GB / 2 cores, 4 threads / about 1 TB free |
| python | 3.11.9 (uv-managed venv) |
| setup time | clone + venv + full `requirements.txt` install in about 1 min (the install itself 40-55 s) |
| ffmpeg | Ubuntu 6.1.1 (G.711, GSM, Opus; no AMR-NB) plus the pinned johnvansickle 7.0.2 static build for AMR-NB (no libgsm), found through `PEHLEAAP_FFMPEG_EXTRA` |
| quota | 30 GPU-h/week, 12 h per session; CPU sessions are free |

### One-time setup (you, in the browser)

1. Create a Kaggle account, or use an existing one, and **verify your phone number** in its settings.
   Without phone verification, notebooks get neither a GPU nor internet access.
2. https://www.kaggle.com/settings/api → **Generate New Token**, and copy it. On the Dev box, save it as the only
   content of `%USERPROFILE%\.kaggle\access_token` (no extension; `access_token.txt` also works). It is a secret:
   never put it in the repo.
3. Check it: `.venv\Scripts\python scripts\kaggle_job.py quota` prints the weekly GPU hours. On 2026-10-24: **30 GPU-h/week** (and 20 TPU-h).

### Jobs

`scripts/kaggle_job.py` pushes a job from the Dev box as a private script kernel (`pehleaap-<name>`). On Kaggle,
`scripts/kaggle_job_template.py` clones this repo at the current (pushed) commit, installs Python 3.11.9 and
`requirements.txt` into a venv with uv, and runs the command from the repo root. The runner waits, then downloads
`job.log`, `env_check.md`, `job.json` and the job's `runs/` folder into `runs\kaggle\<name>\output\`.

```powershell
.venv\Scripts\python scripts\kaggle_job.py run smoke --gpu t4 -- python -m pytest     # first check: env + tests on a T4
.venv\Scripts\python scripts\kaggle_job.py status smoke
.venv\Scripts\python scripts\kaggle_job.py fetch smoke
```

GPU sessions (`--gpu t4`) use the weekly quota; CPU sessions (the default) don't. Recordings go up as the **private** dataset
`pehleaap-raw` (a new version after each recording day) and are checked against `data/manifests/raw.jsonl`.
