---
name: gpu-box-runs-untouched
description: "User rule: nothing done on the Dev box may affect GPU-box runs; keep the scripts its jobs use backward compatible and prove outputs are unchanged"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-24T12:30:04.329Z
---

Work on the Dev box must never change, break or slow down a job on the GPU box.

**Why:** the user (2026-10-24): "all the main runs will be done on the gpu box machine and whatever we will do here wont affect
there and fine tuning also we will do there". The GPU box pulls main at every Claude Code session start (SessionStart hook), so a
merged change reaches it the next time Claude starts there, and an interrupted job resumes with whatever code it has then.

**How to apply:**
- Before changing a script a GPU-box job uses (e.g. `scripts/eval_clip_baselines.py`, `pehleaap/telephony/codecs.py`), keep its
  existing options and outputs exactly as they are; put new behaviour behind new options, conditions or `--out` folders.
- Prove it: compare old vs new outputs on real data (done 2026-10-24: codec-only audio bit-identical to commit 671a151 for all
  7 codecs) and say so when handing the change over.
- Never put a change that alters results into the middle of a GPU-box job; new jobs go after the running one (they compete for its
  24 cores otherwise).
- The Dev box downloads and data are separate from the GPU box's; the GPU box fetches its own copies (checked against
  committed manifests).

Related: [[two-machine-setup]], [[compute-routing]].
