---
name: two-machine-setup
description: "Compute: Dev laptop for code/demo, Kaggle T4 for all heavy jobs from 2026-10-29, RTX 4000 Ada GPU box only until 2026-10-28 (baseline eval)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-23T22:42:10.287Z
---

Compute (revised 2026-10-24). The GitHub repo is the only thing shared between machines ([[shared-memory]]).

- **Dev box** = laptop, repo at `D:\pehleaap`: i5-11300H 4C/8T, GTX 1650 4 GB (Turing, no native bf16 → fp16), 7.8 GB RAM,
  ~450 GB free on D:. Code, tests, labeling + verify UI, eval on slices, Pipecat demo, smoke runs, CPU benchmark #1.
- **Kaggle, from 2026-10-29** (user's choice, 2026-10-24): all training, batch ASR, augmentation precompute (CPU sessions, no GPU
  quota), full eval sweeps. T4 16 GB (two per session; use T4, Turing like the Dev box) or P100 16 GB; no bf16 → fp16 AMP + GradScaler;
  ~29 GB RAM, 4 vCPUs; ~30 GPU-h/week (varies; user to check in account settings), 12 h sessions. Recordings go up as a private
  dataset; jobs clone a pinned commit. Runner `scripts/kaggle_job.py` not written yet. Details: docs/PLAN.md §3a, docs/machines.md.
- **GPU box** (checked 2026-10-24 by env_check: **RTX 4000 SFF Ada, 20 GB dedicated**, bf16; Core Ultra 9 285 24 cores;
  63 GB RAM; ~100 GB free on C:; Python 3.12.10): **only available until 2026-10-28**. It runs the baseline eval
  (docs/machines.md Job 1) and nothing else. Repo at `C:\Users\FITT\Desktop\PRIYANSH!!\pehleaap`.
  Earlier notes called it an RTX 4060 with 8 GB dedicated; env_check proved that wrong. Job 1 is CPU-only (onnxruntime
  CPUExecutionProvider), so the 24 cores set its speed (`--threads 24`) and the GPU sits idle during it.

**Why:** the laptop can't train in reasonable time, and the GPU box leaves on Oct 28; Kaggle was the plan's fallback and the
consent forms already name private Kaggle datasets.
**How to apply:** don't plan jobs on the GPU box after Oct 28. Size batches for 16 GB fp16, checkpoint every 30 min and resume across
sessions, keep within the weekly GPU-hour budget in PLAN §3a. Never make a Kaggle dataset with audio public.

Related: [[pehleaap-status]], [[dev-environment]].
