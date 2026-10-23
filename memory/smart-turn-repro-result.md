---
name: smart-turn-repro-result
description: Smart Turn v3.2 reproduction PASS numbers and the headline finding that its Hindi test set is 100% synthetic TTS
metadata:
  type: project
---

Run on 2026-10-24 with `scripts/repro_smart_turn.py` (int8 CPU model, threshold 0.5, upstream preprocessing that left-pads to 8 s).
Report in the repo at `docs/results/week1_smart_turn_v3.2_repro.md`.

- English: 94.17% vs published 94.26% (n=7,820). Hindi: 89.95% vs 90.11% (n=1,284). PASS, so the harness is trusted.
- Smart Turn's "FPR/FNR" are % of *all* samples (FP/total), not conventional rates. Our clip_metrics keeps both.
- **All 1,284 Hindi test clips are synthetic TTS (dataset `chirp3_1`).** Conventional cut-in rate (FP on unfinished turns):
  Hindi synthetic 18.2% vs English human 6.9% vs English synthetic 9.1%.

**Why it matters:** this is the strongest early motivation number for the write-up. There is no real Hindi speech in Smart Turn's evaluation.
**How to apply:** cite it with that scope (synthetic Hindi, Smart Turn's own test set), never as "Smart Turn fails on Hindi calls".
Laptop latency in the same session (168 ms/inference int8, 1 thread, while downloading) is NOT a valid benchmark; the formal compute benchmark is in week 5.
