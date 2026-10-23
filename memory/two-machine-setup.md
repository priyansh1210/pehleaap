---
name: two-machine-setup
description: "Dev laptop vs RTX 4060 GPU box: specs, repo paths, roles; GPU box VRAM (\"20 GB\") unconfirmed, plan assumes 8 GB dedicated"
metadata:
  type: project
---

Two machines (decided 2026-10-23). The GitHub repo is the only thing they share ([[shared-memory]]).

- **Dev box** = laptop, repo at `D:\pehleaap`: i5-11300H 4C/8T, GTX 1650 4 GB (Turing, no native bf16 → fp16), 7.8 GB RAM,
  ~450 GB free on D:. Used for code, tests, labeling UI, eval on slices, Pipecat demo, CPU benchmark #1.
- **GPU box**, repo at `C:\Users\FITT\pehleaap` (the user said the exact path may differ; check before relying on it): RTX 4060.
  The user says it has "20 GB VRAM". Every 4060 has 8 GB dedicated (the 4060 Ti has 8 or 16 GB), so 20 GB is almost certainly
  Task Manager's dedicated+shared figure. **Not yet confirmed**: run `python scripts\env_check.py --markdown` there and paste it into
  docs/machines.md. Used for all training, batch ASR, augmentation precompute, full eval sweeps. Kaggle/Colab is only a fallback.

**Why:** the laptop can't train in reasonable time. Batch sizes in docs/PLAN.md §3a are sized for 8 GB.
**How to apply:** if env_check shows 16 GB, double the batches; don't plan around 20 GB. Develop on the Dev box; on the GPU box, run
and monitor jobs (runbook: docs/machines.md "GPU box"). If a fix is needed on the GPU box, make it on a branch and push before switching
back, so the machines never diverge. remote.ps1 / fetch_ckpt.ps1 default to the GPU box path above (override with PEHLEAAP_GPU_REPO);
they and sync_data.ps1 are untested until OpenSSH/Tailscale and PEHLEAAP_GPU_SHARE are set up.

Related: [[pehleaap-status]], [[dev-environment]].
