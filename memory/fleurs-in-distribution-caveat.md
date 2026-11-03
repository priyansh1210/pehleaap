---
name: fleurs-in-distribution-caveat
description: "eot_v1's 6-7 pt FLEURS lead is partly home field (trained on FLEURS train; eot_v0 without it scored 74.6 < codewithmoin 81.6); README says so (PR #44); IndicVoices hand-labelled test in progress"
metadata:
  node_type: memory
  type: project
  originSessionId: 7e914bc0-a68b-4459-8501-be82eac6c07c
  modified: 2026-10-25T21:06:48.643Z
---

Laptop's honest review of the project (Priyansh asked "how good is this project", 2026-10-26): engineering and rigour strong
(resume-grade); research not publishable yet; product = credible upgrade over Pipecat's default but unproven on real calls.

**Biggest weakness:** eot_v1 trained on FLEURS Hindi train, so the FLEURS test/dev benchmark is in-distribution for it (same read
speech, recording setup, clip construction "cut 0.2 s into a pause", maybe same readers; FLEURS has no speaker IDs). The rivals never
saw this data. Evidence: eot_v0 (same recipe, no FLEURS) scored 74.6 clean, below codewithmoin 81.6, so most of the lead comes from
the FLEURS data. The README makes this point about the rivals on Smart Turn's test set but not about eot_v1 on FLEURS.
Other weaknesses: the "pause before the verb" claim is untested (FLEURS pauses at commas); no new modelling (fine-tune of Smart Turn);
18% clean cut-in still too high.

**Fixes (status 2026-11-04):**
1. Pilot P01 conversation scored with `eval_sessions.py` (first test with no home-field advantage): pilot status unknown, see
   [[pehleaap-status]].
2. Held-out spontaneous Hindi: IndicVoices (CC BY 4.0). Its segment ends aren't turn ends, so Priyansh hand-labels 300 clips
   (`make_indicvoices_eot.py`, PR #49; clips on the GPU box). Scored after `docs/prereg.md` freezes: its H5.
3. README "Read with care" line: DONE (PR #44, Nov 3).
