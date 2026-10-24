---
name: pehleaap-status
description: "Locked decisions (compute = Dev box + Kaggle from Oct 29), 8-week schedule with dates, done/pending (PR #1-#6 merged; GPU-box baseline eval done + merged 2026-10-24)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-23T22:53:34.850Z
---

The full plan is in the repo at `docs/PLAN.md` (approved 2026-10-23). Don't duplicate it here; read it.

**Locked decisions (user-chosen):** compute = Dev box + **Kaggle from 2026-10-29** (changed 2026-10-24: the GPU box (RTX 4000 Ada 20 GB) is
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
Dev box is set up (autoMemoryDirectory -> D:/pehleaap/memory); GPU box (2026-10-24): repo cloned, `.venv` built, 64 tests pass,
smoke eval passed (chhotu, 1 shard, 39 s), full Job 1 started with `--threads 24`; `setup_claude_memory.ps1` done there too (autoMemoryDirectory ->
C:/Users/FITT/Desktop/PRIYANSH!!/pehleaap/memory). From Bash `!`, use forward slashes: `scripts/setup_claude_memory.ps1`.

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

**Done 2026-10-24 (GPU box, Job 1, 73 min CPU-only, `--threads 24`; PR #6, merge c397b8e):** full clean vs phone-line baseline eval, all 31.5k clips.
Saved in `docs/results/week1_clip_baselines_report.md` + `_results.json`. Every wrapper matches its card within 0.15 pt (eng and hin).
Phone line (G.711 mu-law / AMR-NB 7.95), Hindi (all synthetic TTS): Smart Turn acc 90.3 -> 86.4 / 82.8, cut-in 17.7% -> 27.5% / 35.3%;
chhotu collapses (-23 to -27 pt, cut-in up to 66%); deveshu and codewithmoin hold up (<=2.3 pt drop); abhishek is weak everywhere
(60 eng / 68 hin) and flat across conditions. Narrowband + codec only, no noise/echo/loss yet; in-distribution for the community models.
Cite Smart Turn's phone-line drop with that scope (synthetic Hindi, its own test set).

**GPU box, 2026-10-24 ( user OK'd using its last days for more baseline evals):** (a) **Job 2, codec sweep: DONE** (96 min, `--threads 24`), in `docs/results/week1_codec_sweep_report.md` + `_results.json`; the clean rows match Job 1 exactly. Smart Turn on Hindi: A-law -4.1, Opus -6.2, GSM-FR -6.4, AMR 12.2 -7.2, AMR 4.75 **-9.7** (cut-in 40%). AUC only 0.982 -> 0.951: mostly a calibration shift (best Hindi threshold 0.91-0.95 vs the card's 0.5, tuned on the test set, so optimistic: AMR 4.75 80.5 -> 87.5). So the recipe needs per-language and per-channel calibration, and reports should give AUC. English moves <= 2.2 points. deveshu/codewithmoin <= 2.7 points; chhotu -23 to -27 points. (b) LiveKit `v1-mini` (livekit-agents 1.8.9, livekit-local-inference 0.2.7, scratch venv on the GPU box, not `.venv`): **runs**. LICENSE: the LiveKit Model License 3(i) forbids standalone use or use with frameworks other than LiveKit Agents, and 3(ii) forbids using its outputs to improve or develop other models. So never call `livekit.local_inference.EOT` directly and never train/distill/label with its outputs; drive it only through `livekit.agents.inference.TurnDetector(version="v1-mini").stream()` (push_audio, then predict()). Whether even that counts as "with LiveKit Agents" outside an AgentSession is the user's call (asked 2026-10-24). How LiveKit uses it: the last 1.2 s of 16 kHz audio, scored after 200 ms of VAD silence; thresholds hi 0.305, en 0.36 (p < thr extends the endpointing delay). Smoke test on shard 0 (132 hin + 300 eng clips, clean): AUC hin 0.76 / eng 0.71 vs Smart Turn 0.99; at LiveKit thresholds acc 0.59 / 0.54 with cut-in 92% / 82%. Not caused by gain (0.25-3x) or trailing silence (0/200/400 ms); deterministic; ~35-40 ms/clip on 1 thread. Caveat: the Smart Turn test set is in-distribution for Smart Turn, not for LiveKit. Full run (~1 h for clean/mulaw/amr7.95) awaits the user's OK. (c) `scripts/kaggle_job.py` will be written on the **Dev box** (user's choice), then smoke-tested there once the token is in place.

**Kaggle account created + phone-verified (2026-10-24, on the Dev box).** No `kaggle.json` on the GPU box (not needed there).

**Pending, week 1:** user to confirm the API token is at
`%USERPROFILE%\.kaggle\kaggle.json` on the Dev box and note the weekly GPU quota; me (on the Dev box): `scripts/kaggle_job.py` + a smoke job on a T4; user to review the consent forms (contact details, deletion date, college ethics
approval?); Craig vs Zoom pilot with one pair (`scripts/check_alignment.py`); start recruiting and booking >=15 sessions; HingCoS access
email; PSTN provider signup (Twilio dual-channel first); LiveKit v1-mini baseline (time-boxed to 1 day; started on the GPU box, see above); user to try the ding-bot demo by voice.
Note: Pipecat's LocalSmartTurnAnalyzerV3 only fits Smart Turn's exact interface; abhishek's model would load there but give wrong
probabilities (logits, no normalisation), so the demo uses ClipModelTurnAnalyzer.
