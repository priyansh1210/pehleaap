---
name: fleurs-in-distribution-caveat
description: "eot_v1's 6-7 pt FLEURS lead is partly home field (trained on FLEURS train; eot_v0 without it scored 74.6 < codewithmoin 81.6); README doesn't say so yet; fixes proposed"
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

**Proposed fixes, in order (none done yet; Priyansh hasn't decided):**
1. Pilot P01 conversation scored with `eval_sessions.py` (first test with no home-field advantage).
2. A held-out real-voice corpus no model trained on, ideally spontaneous Hindi (IndicVoices extempore; Kathbath = read). Check
   licences first.
3. One line in README "Read with care": FLEURS is in-distribution for eot_v1. Laptop offered to draft it before the v1.0 tag (Oct 28).
