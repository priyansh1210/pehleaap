---
name: hf-publish-plan
description: "Hugging Face upload of eot_v1: not dated in the plan; Laptop recommended Oct 28 with the v1.0 tag, as a mirror of the GitHub release (same bytes); Priyansh hasn't decided"
metadata:
  node_type: memory
  type: project
  originSessionId: 7e914bc0-a68b-4459-8501-be82eac6c07c
  modified: 2026-10-25T21:06:42.819Z
---

The plan never dates a Hugging Face upload. The Oct 28 v1 checklist said "weights published (GitHub release or Hugging Face)", and
the GitHub release `models-v1` (live since 2026-10-25) satisfied it. PLAN week 8 (Dec 14-20) has "model card and datasheet".

**Laptop's recommendation (2026-10-26, Priyansh asked "when can we publish this on hugging face"):** publish on Oct 28 alongside the
v1.0 tag (or Oct 27 if there's time); nothing blocks it, since eot_v1 is final and the pilot only changes the README. Why HF: the
community models eot_v1 beats (codewithmoin, deveshu, abhishek) live there, and HF shows public download counts (a resume number).

How: Priyansh's account and click, repo e.g. `priyansh1210/eot_v1`; upload the SAME `model.onnx` as the GitHub release (same
sha256); model card = FLEURS table + p-values, 8 kHz focus, Pipecat snippet + threshold, BSD-2 + Smart Turn's Daily notice +
CC BY 4.0 data credits, limits (read speech, FLEURS in-distribution [[fleurs-in-distribution-caveat]], English slip, int8 not shipped).
Week 8 expands this card instead of starting a new one.

**Keep `scripts/get_weights.py` on the GitHub release;** add HF to the README as a second link ([[gpu-box-runs-untouched]]).

**Status:** Priyansh hasn't said yes yet. Laptop offered to draft the model card + copy-paste upload commands; ask tomorrow.
