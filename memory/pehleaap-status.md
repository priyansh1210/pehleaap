---
name: pehleaap-status
description: "Locked decisions (compute = Dev box + Kaggle from Oct 29), 8-week schedule with dates, done/pending (PR #1-#4 merged; Kaggle switch on branch kaggle-compute)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-23T22:53:34.850Z
---

The full plan is in the repo at `docs/PLAN.md` (approved 2026-10-23). Don't duplicate it here; read it.

**Locked decisions (user-chosen):** compute = Dev box + **Kaggle from 2026-10-29** (changed 2026-10-24: the RTX 4060 GPU box is
only available until Oct 28 and runs the baseline eval only; [[two-machine-setup]]);
recording = bulk browser/app two-track recording (Craig on Discord or Zoom per-participant files) with telephony simulated afterwards,
plus a 1–2 h real-PSTN set used only for testing; 40+ speakers (~17–20 h); demo = web call with 8 kHz telephony simulation + baseline/ours toggle,
plus recorded PSTN clips (no Indian phone number/KYC).

**Schedule:** week 1 = Oct 26–Nov 1 2026 … week 8 = Dec 14–20 2026. Splits and `docs/prereg.md` freeze at end of week 3 (Nov 15).
The Tier B (VAP) go/no-go gate is at end of week 6 (Dec 6).

**Done by 2026-10-24 (PR #1, merge e6ec6df):** repo scaffold; venv with pinned requirements; Smart Turn v3.2 ONNX baseline
and reproduction PASS ([[smart-turn-repro-result]]); clip + causal event-level EOT metrics; two-track drift check;
delay-compensated telephony codecs; consent forms EN/HI; recording protocol; scenario cards; 60 fake-PII cards; env_check; data manifests.

**Done 2026-10-24 (PR #2, merge dcdd96e):** community Hinglish models behind one interface (`pehleaap/baselines/turn_models.py`:
deveshu, abhishek, codewithmoin, Chhotu + Smart Turn); resumable clean-vs-phone-line baseline eval (`scripts/eval_clip_baselines.py`,
saves per shard); real-time phone line (`pehleaap/telephony/realtime.py`, G.711 bit-identical to ffmpeg, ~6 ms); `.venv-demo` with
Pipecat 1.13 (`requirements-demo.txt`); `pehleaap/pipecat_ext/` (ClipModelTurnAnalyzer, PhoneLineProcessor); `demo/bot.py` ding bot
(serves :7860; offline end-to-end test passes; Smart Turn parity with Pipecat's own analyzer < 1e-3). 26 tests in .venv, 4 more in .venv-demo.

**Done 2026-10-24 (PR #3, merge c8f4606):** project memory moved into the repo so both machines share it ([[shared-memory]]).
Dev box is set up (autoMemoryDirectory -> D:/pehleaap/memory); the GPU box still needs `scripts\setup_claude_memory.ps1`.

**Decision (user, 2026-10-24):** the full baseline eval (~2.5 h, too heavy for the laptop's 8 GB RAM) runs on the GPU box; runbook in
`docs/machines.md`. The laptop runs the demo and small smoke tests.

**Decision (user, 2026-10-24): Kaggle replaces the GPU box from Oct 29.** docs/PLAN.md §3a rewritten (T4 16 GB, fp16 AMP + GradScaler
since no bf16, batches 192×2 EOT / 128×3 multi-task, ~30 GPU-h/week budget per week with week 6 tightest, 12 h sessions → checkpoint
every 30 min and resume, recordings as a private Kaggle dataset `pehleaap-raw` checked against manifests, CPU benchmark #2 = Kaggle vCPU
or another machine). docs/machines.md has a Kaggle section; the untested GPU-box scripts (remote.ps1, fetch_ckpt.ps1, sync_data.ps1,
run_job.cmd) were deleted. torch 2.14.1+cu126 includes sm_60 (P100) and sm_75 (T4) kernels (checked).

**Done 2026-10-24 (PR #4, merge 7d4186d):** labeling engine v0. `pehleaap/audio/vad.py` (Silero ONNX via onnxruntime, no torch,
batched tracks, bit-identical to silero's wrapper, 25 min × 2 tracks in ~12 s), `pehleaap/labeling/` (config, lexicons, ipu,
events, schema), `configs/labeling.yaml`, `scripts/label_session.py`; 38 new tests (64 pass in .venv). Check on a synthetic
session built from real Smart Turn Hindi clips: every SHIFT/HOLD/INTERRUPT right; VAD offsets ±130 ms, onsets 0–150 ms late
(up to 550 ms on soft onsets), so short early starts (gap −0.1 s) read as small positive gaps. Calibrate boundaries on the pilot
recording, not on TTS. Not built yet: NOISE_ECHO (speakerphone sessions will show spurious BC/INTERRUPT), ASR hookup
(text rules wait for it), Gradio verify app.

**Pending, week 1:** user to run the baseline eval on the GPU box before Oct 28 (runbook in docs/machines.md; setup_claude_memory only
if running Claude Code there) and bring back report.md; user to create + phone-verify a Kaggle account, put the API token in
`%USERPROFILE%\.kaggle\kaggle.json` on the Dev box and note the weekly GPU quota; me: `scripts/kaggle_job.py` + a smoke job on a T4; user to review the consent forms (contact details, deletion date, college ethics
approval?); Craig vs Zoom pilot with one pair (`scripts/check_alignment.py`); start recruiting and booking >=15 sessions; HingCoS access
email; PSTN provider signup (Twilio dual-channel first); LiveKit v1-mini baseline (time-boxed to 1 day); user to try the ding-bot demo by voice.
Note: Pipecat's LocalSmartTurnAnalyzerV3 only fits Smart Turn's exact interface; abhishek's model would load there but give wrong
probabilities (logits, no normalisation), so the demo uses ClipModelTurnAnalyzer.
