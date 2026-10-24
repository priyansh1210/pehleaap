---
name: two-machine-setup
description: "Compute: Dev laptop for code/demo; GPU box (RTX 4000 Ada 20 GB) for main runs + fine-tuning while it lasts (end date unknown); Kaggle T4 = ready fallback"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-24T13:52:58.786Z
---

Compute (revised twice on 2026-10-24). The GitHub repo is the only thing shared between machines ([[shared-memory]]).

- **Dev box** = laptop, repo at `D:\pehleaap`: i5-11300H 4C/8T, GTX 1650 4 GB (Turing, no native bf16 → fp16), 7.8 GB RAM,
  ~450 GB free on D:. Code, tests, labeling + verify UI, eval on slices, Pipecat demo, smoke runs, CPU benchmark #1.
- **GPU box** (checked 2026-10-24 by env_check: **RTX 4000 SFF Ada, 20 GB dedicated**, bf16; Core Ultra 9 285 24 cores;
  63 GB RAM; ~100 GB free on C:; Python 3.12.10; repo at `C:\Users\FITT\Desktop\PRIYANSH!!\pehleaap`): **all main runs and
  fine-tuning, for as long as it is available.** The user said (2026-10-24) they're not sure when they'll lose it (it was first
  said to be until Oct 28) and will use it as much as they can. Runbooks: docs/machines.md (Job 1 done, Job 2 codec sweep running,
  Job 3 full-chain eval next). Earlier notes called it an RTX 4060 with 8 GB; env_check proved that wrong. Job 1 was CPU-only
  (onnxruntime CPUExecutionProvider), so its 24 cores set the speed (`--threads 24`); see [[compute-routing]].
- **Kaggle** = the fallback, ready now: T4 16 GB (two per session) or P100; no bf16 → fp16 AMP + GradScaler; measured 2026-10-24:
  2× T4 14.6 GB usable, 31 GB RAM, 2 cores / 4 threads; 30 GPU-h/week (+20 TPU-h), 12 h sessions. Jobs clone a pushed commit via
  `scripts/kaggle_job.py` (Dev box only, where the API token is); smoke test passes. Details: docs/PLAN.md §3a, docs/machines.md.

**Access (2026-10-24):** the user set up Chrome Remote Desktop, so they can reach the GPU box's screen from the Dev box at any
time (to run runbook commands or talk to its own Claude Code session). Claude on the Dev box still can't run commands there.

**Why:** the laptop can't train in reasonable time; the GPU box is much faster than Kaggle's T4 but may go any day.
**How to apply:** front-load GPU-heavy work (evals, fine-tuning) onto the GPU box while it exists. Write training code that runs
unchanged on both (bf16 on Ada, fp16 on T4; checkpoint every 30 min and resume), so a job can move to Kaggle the day the box goes.
Never disturb a running GPU-box job from here ([[gpu-box-runs-untouched]]). Never make a Kaggle dataset with audio public.

Related: [[pehleaap-status]], [[dev-environment]].
