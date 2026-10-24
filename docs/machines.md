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

**Job 3: the same baselines over the full phone-line chain** (after Job 2 has finished; about 2.5-3 h with
`--threads 24`). It adds gateway transcoding, packet loss, AGC/clipping and MUSAN noise at 20/10/5 dB (from the 20% of MUSAN's noise files held out from training)
(`CHAIN` in `scripts/eval_clip_baselines.py`), plus one random chain per clip from `configs/augment.yaml`.
The codec-only conditions are unchanged, so Job 1 and Job 2 numbers stay comparable. `get_noise.py` streams
openslr's MUSAN archive, keeps only its noise part (about 0.7 GB) and checks every file against the committed
`data/manifests/noise_musan.jsonl`. A rerun skips the download if the copy already checks out.

```powershell
git pull
.venv\Scripts\python scripts\get_noise.py
.venv\Scripts\python scripts\eval_clip_baselines.py --models chhotu_int8 --limit-shards 1 --conditions clean noise5_amr7.95 --out smoke_chain
.venv\Scripts\python scripts\eval_clip_baselines.py --conditions clean tandem_amr4.75 loss3_g711 agc_clip_g711 noise20_amr7.95 noise10_amr7.95 noise5_amr7.95 mixed --threads 24 --out clip_baselines_chain *> runs\clip_baselines_chain.log
```

Bring back `runs\clip_baselines_chain\report.md` and `results.json`, as for Job 1.

**LiveKit turn-detector v1-mini** (clean + the 7 codecs on all clips; about 10 min with `--workers 12`). It
runs from its own venv, so livekit-agents stays out of `.venv`. Its license allows use only with LiveKit
Agents and forbids using its outputs to improve other models: the script drives it through livekit-agents'
`TurnDetector` API only, and its predictions are comparison numbers, never training data or labels.
Add the chain conditions above with `--conditions` once MUSAN is downloaded.

```powershell
py -3.12 -m venv .venv-livekit
.venv-livekit\Scripts\python -m pip install -r requirements-livekit.txt
.venv-livekit\Scripts\python -m pip install -e . --no-deps
.venv-livekit\Scripts\python scripts\eval_livekit_v1mini.py --limit-shards 1 --conditions clean amrnb_4.75 --out livekit_smoke
.venv-livekit\Scripts\python scripts\eval_livekit_v1mini.py --workers 12 *> runs\livekit_v1mini.log
```

**Job 4: fine-tune Smart Turn v3.2 for the phone line** (`scripts/train_eot.py`; after Job 3, whose CPU work it
would compete with). It starts from the published v3.2 weights and trains on Smart Turn's own English and Hindi
training clips (about 80k; every Hindi clip is TTS), 80% of them sent through the phone-line chain on the fly by
20 worker processes (training noise only: the 80% of MUSAN that Job 3 never uses). bf16 on this card, 4 epochs,
upstream's recipe otherwise. Expect about 1 h; it checkpoints every 30 min, and rerunning the same command resumes.

1. **Training clips** (`prepare_eot_train.py`): downloads the 83 shards (41 GB, sha256-checked), keeps English and
   Hindi (about 16 GB in `data\external\eot-train\`) and deletes each shard once stored. Mostly downloading, so it can
   run while Job 3 is still going; a rerun resumes.
2. **Smoke run**, a few minutes: checks the GPU, bf16 and the workers.
3. **Training**: progress in the log, numbers in `runs\eot_v0\` (`dev.csv`: dev accuracy, AUC and cut-in rate,
   clean and phone line, per language; step 0 is the published model).
4. **Test-set evaluation**: the fine-tuned model next to the published one under every phone condition so far.

```powershell
git pull
.venv\Scripts\python scripts\prepare_eot_train.py --delete-shards *> runs\prepare_eot_train.log
.venv\Scripts\python scripts\train_eot.py --out eot_smoke --max-steps 5 --set num_workers=4 dev_limit=64
.venv\Scripts\python scripts\train_eot.py --out eot_v0 *> runs\eot_v0.log
.venv\Scripts\python scripts\eval_clip_baselines.py --models smart_turn_v3.2_int8 --onnx eot_v0=runs\eot_v0\model.onnx --conditions clean g711_mulaw amrnb_7.95 amrnb_4.75 tandem_amr4.75 loss3_g711 agc_clip_g711 noise20_amr7.95 noise10_amr7.95 noise5_amr7.95 mixed --threads 24 --out eot_v0_eval *> runs\eot_v0_eval.log
```

`summary.json` also holds the model's decision thresholds per channel (clean/phone) and language, tuned on dev clips.
Bring back `runs\eot_v0\dev.csv`, `summary.json`, `meta.json` and `runs\eot_v0_eval\report.md` + `results.json`. Keep
`runs\eot_v0\model.onnx` (32 MB, not in git) on the GPU box. Read the chain conditions with care: the model trained on
the same family of augmentations (with other noise recordings), so gains there are expected; the real test is real
phone calls (plan §6).

**Threshold calibration** (`calibrate_thresholds.py`, seconds, one core: fine to run while another job is going). How much of
each model's phone-line drop a re-picked decision threshold recovers, cross-fitted so no clip's threshold is tuned on that clip.
It only reads saved `predictions.csv` files, so run it on Jobs 1-2 now, and again once Jobs 3 and 4 have finished:

```powershell
.venv\Scripts\python scripts\calibrate_thresholds.py --runs clip_baselines clip_baselines_codecs --out calibration_codecs
.venv\Scripts\python scripts\calibrate_thresholds.py --runs clip_baselines clip_baselines_codecs clip_baselines_chain eot_v0_eval --out calibration_all
```

Bring back `runs\calibration_*\report.md` and `results.json` into `docs\results\`.

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
