---
name: pehleaap-status
description: "Locked decisions (compute = GPU box while it lasts, Kaggle fallback), 8-week schedule with dates, done/pending (PR #1-#13 merged; GPU-box baseline eval done; Kaggle runner + augmentation chain merged)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-24T20:12:14.688Z
---

The full plan is in the repo at `docs/PLAN.md` (approved 2026-10-23). Don't duplicate it here; read it.

**Locked decisions (user-chosen):** compute = develop on the Dev box; **main runs and fine-tuning on the GPU box (RTX 4000 Ada 20 GB)
for as long as it is available** (user, 2026-10-24, later the same day: end date unknown, "i will try to work as much as i can on it");
Kaggle (set up and smoke-tested) is the fallback the day it goes ([[two-machine-setup]], [[gpu-box-runs-untouched]]);
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

**Decision (user, 2026-10-24; PR #5, merge 5f5c935): Kaggle replaces the GPU box from Oct 29.** docs/PLAN.md §3a rewritten (T4 16 GB, fp16 AMP + GradScaler
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

**GPU box, 2026-10-24 (user OK'd using its last days for more baseline evals):** (a) **Job 2, codec sweep: DONE** (96 min, `--threads 24`), in `docs/results/week1_codec_sweep_report.md` + `_results.json`; the clean rows match Job 1 exactly. Smart Turn on Hindi: A-law -4.1, Opus -6.2, GSM-FR -6.4, AMR 12.2 -7.2, AMR 4.75 **-9.7** (cut-in 40%). AUC only 0.982 -> 0.951: mostly a calibration shift (best Hindi threshold 0.91-0.95 vs the card's 0.5, tuned on the test set, so optimistic: AMR 4.75 80.5 -> 87.5). So the recipe needs per-language and per-channel calibration, and reports should give AUC. English moves <= 2.2 points. deveshu/codewithmoin <= 2.7 points; chhotu -23 to -27 points. (b) LiveKit `v1-mini` (livekit-agents 1.8.9, livekit-local-inference 0.2.7, scratch venv on the GPU box, not `.venv`): **runs**. LICENSE: the LiveKit Model License 3(i) forbids standalone use or use with frameworks other than LiveKit Agents, and 3(ii) forbids using its outputs to improve or develop other models. So never call `livekit.local_inference.EOT` directly and never train/distill/label with its outputs; drive it only through `livekit.agents.inference.TurnDetector(version="v1-mini").stream()` (push_audio, then predict()). Whether even that counts as "with LiveKit Agents" outside an AgentSession is the user's call (asked 2026-10-24). How LiveKit uses it: the last 1.2 s of 16 kHz audio, scored after 200 ms of VAD silence; thresholds hi 0.305, en 0.36 (p < thr extends the endpointing delay). Smoke test on shard 0 (132 hin + 300 eng clips, clean): AUC hin 0.76 / eng 0.71 vs Smart Turn 0.99; at LiveKit thresholds acc 0.59 / 0.54 with cut-in 92% / 82%. Not caused by gain (0.25-3x) or trailing silence (0/200/400 ms); deterministic; ~35-40 ms/clip on 1 thread. Caveat: the Smart Turn test set is in-distribution for Smart Turn, not for LiveKit. User OK'd the full run (2026-10-24) under that reading. Built `scripts/eval_livekit_v1mini.py` + `requirements-livekit.txt` (own `.venv-livekit` on the GPU box; reuses eval_clip_baselines' conditions/phone subset/metrics so rows line up; 8-12 parallel streams, bit-identical to 1 stream; runbook in docs/machines.md). **Full run DONE** (8 min): `docs/results/week1_livekit_v1mini_report.md` + `_results.json`. AUC ~0.70 in every condition (Smart Turn 0.95-0.98); clean Hindi acc 55.1, cut-in 93.7%; phone lines 'raise' acc +1.4..+4.9 only by pushing scores below a too-low threshold. With +200 ms silence it flips to mostly unfinished (missed ends 60-77%), AUC 0.65-0.70. Keep it as the LiveKit reference point; recheck on our own two-track recordings. (c) the Kaggle runner was written on the **Dev box** (user's choice); see below.

**Kaggle (Dev box, 2026-10-24):** account created + phone-verified; API token at `%USERPROFILE%\.kaggle\access_token` on the Dev box
only (none on the GPU box); quota 30 GPU-h/week, 20 TPU-h; login works. Runner merged (PR #8, merge e4cbe86)
and smoke-tested on Kaggle: 2× T4 14.6 GB, 31 GB RAM, 2C/4T, full env install ~1 min; on e4cbe86 (CPU session) 68 passed,
1 skipped, whole job 1.4 min:
`scripts/kaggle_job.py` (run/status/fetch/quota; private script kernel `pehleaap-<name>`, waits, downloads to runs/kaggle/<name>/output)
+ `scripts/kaggle_job_template.py` (clones the pushed commit, uv 0.12.24 venv with Python 3.11.9 + requirements.txt, runs the command,
saves job.log/env_check.md/job.json/runs). ffmpeg on Kaggle: Ubuntu 6.1.1 has libgsm but no AMR-NB; johnvansickle 7.0.2 static
and all BtbN Linux builds have AMR-NB but no libgsm; no conda. Fix: template fetches the static build (sha256-pinned) into
PEHLEAAP_FFMPEG_EXTRA and `paths.ffmpeg_bin(encoder)` picks the first build with each encoder. /kaggle/input layout still unknown
(no dataset attached yet).

**Done 2026-10-24 (Dev box; PR #9, merge 5bcb7ac): phone-line augmentation chain** (plan §4.3, a week-2 deliverable done early).
`pehleaap/telephony/{noise,echo,channel,pipeline}.py`, `configs/augment.yaml` (test: equals the defaults), `scripts/augment_examples.py`
(writes listening examples of Hindi Smart Turn clips to runs/augment_examples/), `apply_codec_many` in codecs.py; 19 new tests (87 pass).
Checked: whole chain within 0.12 ms of the input timing; echo lands at its set delay (0.300 s measured); 24 ms/clip batched (2 s clips),
13.5 s for a 25-min track. Jitter model: FIFO queueing behind exponential delay spikes, fixed buffer = jitter x 3.5-6 → 0.3-7% late.
The first AR(1)-smoothed version almost never produced late packets, so it was replaced. Noise is synthetic (white/pink/brown/hum) until noise datasets
go in data/external/noise/ (licenses from memory, verify before downloading: MUSAN CC BY 4.0, DEMAND CC BY-SA 3.0, ESC-50 CC BY-NC 3.0; which to use is the user's call).
Pipecat live demo still G.711 only.

**Job 3 prep (2026-10-24, Dev box, merged in PR #10).** User chose MUSAN only (license checked on openslr.org/17: CC BY 4.0).
`scripts/get_noise.py` streams the 11 GB archive (order is music, speech, noise: noise is LAST, so the whole archive streams through),
keeps musan/noise/ → data/external/noise/musan/, fetches 16 MB byte ranges over 4 connections (each retried on its own; 2.5 MB/s
vs 1.2 for one stream; bytes checked identical to a plain download) and writes/verifies `data/manifests/noise_musan.jsonl`.
Mirror: openslr.elda.org; trmal.net (3 MB/s) and www.openslr.org had expired TLS certs on 2026-10-24 (never disable
verification). `eval_clip_baselines.py` gained CHAIN conditions (tandem_amr4.75, loss3_g711, agc_clip_g711, noise20/10/5_amr7.95,
mixed); a clip's seed is hashed from its id, so the noise is paired across SNRs; ~50 ms/clip of augmentation. Job 3 runbook in
docs/machines.md (after Job 2; ~2.5-3 h with --threads 24). Merged as PR #10 (merge bd2204c, which also brought in the GPU box's Job 2 branch) WITHOUT the manifest (the user
asked to commit before the download finished); `data/manifests/noise_musan.jsonl` follows in a small PR. **GPU box: don't run
get_noise.py or Job 3 until that manifest PR is merged** (it would write its own untracked manifest, and the next pull would fail on it).
Download finished 2026-10-24 (11.09 GB streamed in 39 min, 934 files = 930 noise wavs + LICENSE/ANNOTATIONS/README, 687 MB);
manifest goes in branch `musan-manifest` together with the **noise split**: `NoiseBank(part="train"|"test")` keeps 80/20 of the
files by md5 of the path (749 train / 181 test); Job 3 uses only `part="test"`, training only `part="train"`, so no noise recording
is shared. Licenses inside MUSAN noise: free-sound files public domain; sound-bible files per-file (e.g. CC BY 3.0, see its LICENSE)
— fine for augmentation since the audio is never redistributed.

**Fine-tuning prep (2026-10-24, Dev box, merged in PR #10; user asked "how can we do the fine tuning code here you dont know the results
from the gpu box": answer, training code doesn't depend on eval results, which only tune config later and arrive via git).**
Pipecat publishes Smart Turn v3.2 as ONNX only (HF pipecat-ai/smart-turn-v3: cpu=int8, gpu=fp32); training code is on GitHub
(pipecat-ai/smart-turn train.py, commit 4786657: whisper-tiny encoder with max_source_positions 400, attention pool, classifier;
lr 5e-5, 4 epochs, batch 384, warmup 0.2, wd 0.01, BCE pos_weight from batch stats; data pipecat-ai/smart-turn-data-v3.2-train,
86 files, CC BY 4.0). `pehleaap/models/smart_turn.py` rebuilds it in PyTorch from the fp32 ONNX (MatMul weights unnamed and transposed,
matched by the named bias after each; k_proj has no bias). Parity: max |p| diff 3.3e-6 on 256 real clips, 0 decision flips;
tests/test_smart_turn_torch.py. Note int8 (our baseline) vs fp32 differ by up to 0.66 on single clips at equal accuracy:
compare a fine-tune against both. Do NOT add an fp32 entry to turn_models.REGISTRY while GPU-box jobs use the default `--models`
(saved shards would be recomputed). Next: training script (bf16 on the GPU box, fp16 on Kaggle), smoke-tested on the Dev box.

**Training script (2026-10-24 evening, Dev box; PR #13, merge 148e778): GPU box Job 4 in docs/machines.md.** Train set
pipecat-ai/smart-turn-data-v3.2-train = 83 shards / 41.4 GB / 23 languages; per shard ~961 of 3,265 clips are eng+hin (~80k
total; Hindi all chirp3 TTS). `scripts/prepare_eot_train.py` → `data/external/eot-train/` (EotStore: audio.int16 memmap +
index.csv, last 8 s per clip, 3% dev by md5(id); resumable; `--delete-shards`). `pehleaap/train/eot_data.py` (EpochKeys sampler:
seeded per epoch, exact resume; Augmenter collate in workers: p_phone 0.8, train noise part, features = WhisperMelOnnx.prepare
exactly) and `train_eot.py` (TrainConfig = configs/train_eot.yaml: lr 5e-5, 4 epochs, batch 384 = 2 × 192, warmup 0.1 + cosine,
hin_repeat 2, 20 workers; dev eval at step 0 = published model, best by mean phone-line dev accuracy; ckpt every 30 min; exports
model.onnx via the legacy TorchScript exporter: no onnxscript needed, parity 6e-8). **GTX 1650 fp16 gives NaN from the first conv**
(GTX 16xx quirk) → precision "auto" picks fp32 there, bf16 on Ada, fp16 on T4; NaN loss stops the run. Augmentation ~77 ms/clip
per worker on the Dev box. `eval_clip_baselines.py --onnx NAME=PATH` scores a local ONNX at threshold 0.5 (defaults unchanged).
Smoke: 6 steps on 1 shard fine end to end (fp32 GTX 1650; 4 min mostly Windows worker start-up). 99 tests pass.

**Threshold calibration (2026-10-24 night, Dev box, uncommitted; user picked it over ASR / verify app).** `pehleaap/eval/calibration.py`
(best_threshold = accuracy-optimal, middle of ties; 2-fold cross-fitting by md5(id) % 2; variants card / channel (clean vs phone) /
channel+language / condition+language = upper bound) and `scripts/calibrate_thresholds.py --runs <eval runs>` (reads saved
predictions.csv only, phone subset, seconds). Runbook in docs/machines.md after Job 4: run on Jobs 1-2 now, again after Jobs 3-4.
Local check (132 Hindi clips, one shard): Smart Turn AMR 4.75 card 86.4 → 89.4 cross-fitted, cut-in 28.8% → 16.9%, phone/hin
threshold 0.93. "Deployment thresholds" in the report are tuned on test clips = reference only; train_eot.py now writes dev-tuned
thresholds per channel/language into summary.json (`dev_thresholds`) and `best_threshold` per dev row. 102 tests pass.
Merged as PR #14. Follow-up (2026-10-25, PR #16, merge 9688c68): `calibrate_thresholds.py --shipped eot_v0=runs/eot_v0/summary.json` adds a
"shipped" variant = the fine-tuned model scored with its own dev-tuned thresholds (fair "as deployed" number; missing keys → card).

**eot_v0 training (GPU box, started 2026-10-25 00:33):** 904 steps (226/epoch, 75,497 train clips), bf16. Dev (344 Hindi clips,
±1.6 pt noise): phone/hin 81.4 → ~91-93 acc, cut-in 29% → ~8-9%; clean dev slipped ~2-3 pt but AUC stays 0.99, and the published
model scores 97.1 on clean Hindi dev vs 90.3 on test, so dev clips were likely in upstream's training data (judge clean cost on
the test set only). GPU box session was given a prompt to run the eval, calibration_all, write docs/results/week1_eot_v0_report.md
and commit on branch `eot-v0-results`.
Training ended 00:57, best = step 900 (phone/hin dev 92.7, cut-in 8.7%); summary.json lacked dev_thresholds (run predated PR #14),
so the GPU box was told to rescore best.pt on dev with train_eot's evaluate() and add them before calibrating with --shipped.

**ASR hookup (2026-10-25 night, Dev box, uncommitted).** `pehleaap/labeling/asr.py` + `label_session.py --asr` (words cached as
data/labels/<session>/words.jsonl; `--asr-refresh`, `--asr-model`, `--asr-language`, `--asr-batch-size`), `pehleaap/device.py`
(precision_for moved out of train_eot), `scripts/make_tts_session.py` (stand-in session from Hindi TTS test clips, in runs/).
Whisper via transformers pipeline (no new deps), per-IPU with 0.3 s pad (Whisper hallucinates Devanagari on pure silence: seen),
max_new_tokens = 20/s of the longest clip in a length-sorted batch (word timestamps keep all cross-attentions: a loop OOM'd the
GTX 1650 at batch 16), default batch 8. transformers 5.20.2's temperature fallback + word timestamps crashes (IndexError in
_extract_token_timestamps) → own retry: zlib compression ratio > 2.4 → decode again with no_repeat_ngram_size=3, keep the less
repetitive text. Smoke with whisper-tiny on a stitched TTS session: words all inside their padded IPUs; loops fixed (183 → 86
words). Large-v3 quality/speed still unmeasured: GPU box runbook step in docs/machines.md. 109 tests pass.

**Pending, week 1:** user to review the consent forms (contact details, deletion date, college ethics
approval?); Craig vs Zoom pilot with one pair (`scripts/check_alignment.py`); start recruiting and booking >=15 sessions; HingCoS access
email; PSTN provider signup (Twilio dual-channel first); LiveKit v1-mini baseline (time-boxed to 1 day; started on the GPU box, see above); user to try the ding-bot demo by voice.
Note: Pipecat's LocalSmartTurnAnalyzerV3 only fits Smart Turn's exact interface; abhishek's model would load there but give wrong
probabilities (logits, no normalisation), so the demo uses ClipModelTurnAnalyzer.

**GPU box, 2026-10-24 evening:** LiveKit v1-mini also run on the 8 chain conditions (`runs/livekit_v1mini_chain/`, 20 min;
Hindi AUC 0.71 clean -> 0.60 at 5 dB noise, cut-in 76-82% throughout). Job 3 process died silently after shard 3 (~21:16-21:50;
no traceback, no low-memory event: its PowerShell window was most likely closed or interrupted); resumed at 21:52 from Claude's
session (log `runs/clip_baselines_chain_resume.log`), ETA ~00:15. Job 4 step 1 done: `prepare_eot_train.py` stored 77,808 clips
(eng 63,835 train / 1,967 dev, hin 11,662 / 344; 15 GB in data/external/eot-train). Step 2 smoke PASSED (cuda bf16, 75,497 train
clips, 226 steps/epoch, 57 clips/s with 4 workers while Job 3 ran; ONNX export OK). Full `eot_v0` training waits for Job 3.

GPU box, 2026-10-25 00:01: Job 3 DONE; report `docs/results/week1_chain_baselines_report.md` (+ `_results.json`) covers all
6 models. Smart Turn Hindi: -14 pts tandem AMR4.75->G.711, -17 mixed, -21 at 5 dB SNR (cut-in up to 63%). Codec/loss/AGC drops
are mostly calibration (AUC 0.93-0.96, best thr ~0.95), but noise loses real information (AUC 0.86 at 5 dB; best-threshold acc
still 12-14 pts below clean), so the fix is training on noisy phone audio (eot_v0), not recalibration alone. codewithmoin is
the most robust (worst -7.6, AUC 0.944). chhotu collapses (-32 to -37). Next: full `train_eot.py --out eot_v0`, then eot_v0_eval.

**eot_v0 RESULTS (GPU box, 2026-10-25 02:10):** report `docs/results/week1_eot_v0_report.md`. Best step 900 (22 min train).
Test set, Hindi n=1284 @0.5: clean 90.3 -> 94.1 (McNemar p=2e-6; cut-in 17.7% -> 6.0%, missed end 2.4% -> 5.8%), noise5 69.5 -> 88.6,
mixed 73.5 -> 91.9, tandem 76.4 -> 93.1; beats test-tuned Smart Turn thresholds by 8-11 pts under noise/chains and every baseline
(codewithmoin closest, 0.5-2.6 behind). English clean (7820) 94.3 -> 95.4. Clean audio did NOT get worse on test (dev slip was a
dev-set artifact). Shipped dev thresholds (0.44-0.57) ~= 0.5; no recalibration needed. dev_thresholds were recomputed after
training (run predated PR #14; rescoring matched best_dev exactly); original kept as runs/eot_v0/summary.orig.json (GPU box).
Caveats: synthetic Hindi TTS, chains = training augmentation family. Weakest: 5 dB noise (-5.5 vs clean). model.onnx stays on the
GPU box (runs/eot_v0/, 32 MB, not in git). Next: real-call validation, int8 quantisation, Pipecat demo wiring.

**eot_v0 int8 (GPU box, 2026-10-25 03:30):** `scripts/quantize_eot.py` -> runs/eot_v0/model_int8.onnx (8.7 MB, not in git); doc
`docs/results/week1_eot_v0_int8.md`. Upstream recipe (QDQ, entropy, Conv/MatMul/Gemm only, 1024 CLEAN train clips) on eot_v0 weights
grafted into Pipecat's fp32 graph (`onnx_with_state_dict`, matches TorchScript export within 3e-6). Dev: int8 costs 0.6-2.0 pts;
phone/hin 92.2 -> 90.1 but cut-in 8.7% -> 16.9% (dev-tuned int8 threshold not done). Pipecat's own int8 loses more (phone/hin
82.0 -> 71.8). Latency 1 thread: fp32 56.9 ms, int8 28.6 ms = Smart Turn int8. Gotcha: quantising all op types / phone-mix
calibration shifted probs -0.1..-0.2 (up to -23 pts). Test-set eval of int8 not run yet (~75 min CPU).
