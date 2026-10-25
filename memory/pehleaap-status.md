---
name: pehleaap-status
description: "Locked decisions (compute = GPU box while it lasts, Kaggle fallback), 8-week schedule with dates, done/pending (PR #1-#13 merged; GPU-box baseline eval done; Kaggle runner + augmentation chain merged)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0952188d-bc75-46ae-a49b-65b82c0ee291
  modified: 2026-10-25T05:38:21.070Z
---

The full plan is in the repo at `docs/PLAN.md` (approved 2026-10-23). Don't duplicate it here; read it.

**Locked decisions (Priyansh's choices):** compute = develop on the Laptop; **main runs and fine-tuning on the GPU box (RTX 4000 Ada 20 GB)
for as long as it is available** (Priyansh, 2026-10-24, later the same day: end date unknown, "i will try to work as much as i can on it");
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
Laptop is set up (autoMemoryDirectory -> D:/pehleaap/memory); GPU box (2026-10-24): repo cloned, `.venv` built, 64 tests pass,
smoke eval passed (chhotu, 1 shard, 39 s), full Job 1 started with `--threads 24`; `setup_claude_memory.ps1` done there too (autoMemoryDirectory ->
C:/Users/FITT/Desktop/PRIYANSH!!/pehleaap/memory). From Bash `!`, use forward slashes: `scripts/setup_claude_memory.ps1`.

**Decision (Priyansh, 2026-10-24):** the full baseline eval (~2.5 h, too heavy for the laptop's 8 GB RAM) runs on the GPU box; runbook in
`docs/machines.md`. The laptop runs the demo and small smoke tests.

**Decision (Priyansh, 2026-10-24; PR #5, merge 5f5c935): Kaggle replaces the GPU box from Oct 29.** docs/PLAN.md §3a rewritten (T4 16 GB, fp16 AMP + GradScaler
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

**GPU box, 2026-10-24 (Priyansh OK'd using its last days for more baseline evals):** (a) **Job 2, codec sweep: DONE** (96 min, `--threads 24`), in `docs/results/week1_codec_sweep_report.md` + `_results.json`; the clean rows match Job 1 exactly. Smart Turn on Hindi: A-law -4.1, Opus -6.2, GSM-FR -6.4, AMR 12.2 -7.2, AMR 4.75 **-9.7** (cut-in 40%). AUC only 0.982 -> 0.951: mostly a calibration shift (best Hindi threshold 0.91-0.95 vs the card's 0.5, tuned on the test set, so optimistic: AMR 4.75 80.5 -> 87.5). So the recipe needs per-language and per-channel calibration, and reports should give AUC. English moves <= 2.2 points. deveshu/codewithmoin <= 2.7 points; chhotu -23 to -27 points. (b) LiveKit `v1-mini` (livekit-agents 1.8.9, livekit-local-inference 0.2.7, scratch venv on the GPU box, not `.venv`): **runs**. LICENSE: the LiveKit Model License 3(i) forbids standalone use or use with frameworks other than LiveKit Agents, and 3(ii) forbids using its outputs to improve or develop other models. So never call `livekit.local_inference.EOT` directly and never train/distill/label with its outputs; drive it only through `livekit.agents.inference.TurnDetector(version="v1-mini").stream()` (push_audio, then predict()). Whether even that counts as "with LiveKit Agents" outside an AgentSession is Priyansh's call (asked 2026-10-24). How LiveKit uses it: the last 1.2 s of 16 kHz audio, scored after 200 ms of VAD silence; thresholds hi 0.305, en 0.36 (p < thr extends the endpointing delay). Smoke test on shard 0 (132 hin + 300 eng clips, clean): AUC hin 0.76 / eng 0.71 vs Smart Turn 0.99; at LiveKit thresholds acc 0.59 / 0.54 with cut-in 92% / 82%. Not caused by gain (0.25-3x) or trailing silence (0/200/400 ms); deterministic; ~35-40 ms/clip on 1 thread. Caveat: the Smart Turn test set is in-distribution for Smart Turn, not for LiveKit. Priyansh OK'd the full run (2026-10-24) under that reading. Built `scripts/eval_livekit_v1mini.py` + `requirements-livekit.txt` (own `.venv-livekit` on the GPU box; reuses eval_clip_baselines' conditions/phone subset/metrics so rows line up; 8-12 parallel streams, bit-identical to 1 stream; runbook in docs/machines.md). **Full run DONE** (8 min): `docs/results/week1_livekit_v1mini_report.md` + `_results.json`. AUC ~0.70 in every condition (Smart Turn 0.95-0.98); clean Hindi acc 55.1, cut-in 93.7%; phone lines 'raise' acc +1.4..+4.9 only by pushing scores below a too-low threshold. With +200 ms silence it flips to mostly unfinished (missed ends 60-77%), AUC 0.65-0.70. Keep it as the LiveKit reference point; recheck on our own two-track recordings. (c) the Kaggle runner was written on the **Laptop** (Priyansh's choice); see below.

**Kaggle (Laptop, 2026-10-24):** account created + phone-verified; API token at `%USERPROFILE%\.kaggle\access_token` on the Laptop
only (none on the GPU box); quota 30 GPU-h/week, 20 TPU-h; login works. Runner merged (PR #8, merge e4cbe86)
and smoke-tested on Kaggle: 2× T4 14.6 GB, 31 GB RAM, 2C/4T, full env install ~1 min; on e4cbe86 (CPU session) 68 passed,
1 skipped, whole job 1.4 min:
`scripts/kaggle_job.py` (run/status/fetch/quota; private script kernel `pehleaap-<name>`, waits, downloads to runs/kaggle/<name>/output)
+ `scripts/kaggle_job_template.py` (clones the pushed commit, uv 0.12.24 venv with Python 3.11.9 + requirements.txt, runs the command,
saves job.log/env_check.md/job.json/runs). ffmpeg on Kaggle: Ubuntu 6.1.1 has libgsm but no AMR-NB; johnvansickle 7.0.2 static
and all BtbN Linux builds have AMR-NB but no libgsm; no conda. Fix: template fetches the static build (sha256-pinned) into
PEHLEAAP_FFMPEG_EXTRA and `paths.ffmpeg_bin(encoder)` picks the first build with each encoder. /kaggle/input layout still unknown
(no dataset attached yet).

**Done 2026-10-24 (Laptop; PR #9, merge 5bcb7ac): phone-line augmentation chain** (plan §4.3, a week-2 deliverable done early).
`pehleaap/telephony/{noise,echo,channel,pipeline}.py`, `configs/augment.yaml` (test: equals the defaults), `scripts/augment_examples.py`
(writes listening examples of Hindi Smart Turn clips to runs/augment_examples/), `apply_codec_many` in codecs.py; 19 new tests (87 pass).
Checked: whole chain within 0.12 ms of the input timing; echo lands at its set delay (0.300 s measured); 24 ms/clip batched (2 s clips),
13.5 s for a 25-min track. Jitter model: FIFO queueing behind exponential delay spikes, fixed buffer = jitter x 3.5-6 → 0.3-7% late.
The first AR(1)-smoothed version almost never produced late packets, so it was replaced. Noise is synthetic (white/pink/brown/hum) until noise datasets
go in data/external/noise/ (licenses from memory, verify before downloading: MUSAN CC BY 4.0, DEMAND CC BY-SA 3.0, ESC-50 CC BY-NC 3.0; which to use is Priyansh's call).
Pipecat live demo still G.711 only.

**Job 3 prep (2026-10-24, Laptop, merged in PR #10).** Priyansh chose MUSAN only (license checked on openslr.org/17: CC BY 4.0).
`scripts/get_noise.py` streams the 11 GB archive (order is music, speech, noise: noise is LAST, so the whole archive streams through),
keeps musan/noise/ → data/external/noise/musan/, fetches 16 MB byte ranges over 4 connections (each retried on its own; 2.5 MB/s
vs 1.2 for one stream; bytes checked identical to a plain download) and writes/verifies `data/manifests/noise_musan.jsonl`.
Mirror: openslr.elda.org; trmal.net (3 MB/s) and www.openslr.org had expired TLS certs on 2026-10-24 (never disable
verification). `eval_clip_baselines.py` gained CHAIN conditions (tandem_amr4.75, loss3_g711, agc_clip_g711, noise20/10/5_amr7.95,
mixed); a clip's seed is hashed from its id, so the noise is paired across SNRs; ~50 ms/clip of augmentation. Job 3 runbook in
docs/machines.md (after Job 2; ~2.5-3 h with --threads 24). Merged as PR #10 (merge bd2204c, which also brought in the GPU box's Job 2 branch) WITHOUT the manifest (Priyansh
asked to commit before the download finished); `data/manifests/noise_musan.jsonl` follows in a small PR. **GPU box: don't run
get_noise.py or Job 3 until that manifest PR is merged** (it would write its own untracked manifest, and the next pull would fail on it).
Download finished 2026-10-24 (11.09 GB streamed in 39 min, 934 files = 930 noise wavs + LICENSE/ANNOTATIONS/README, 687 MB);
manifest goes in branch `musan-manifest` together with the **noise split**: `NoiseBank(part="train"|"test")` keeps 80/20 of the
files by md5 of the path (749 train / 181 test); Job 3 uses only `part="test"`, training only `part="train"`, so no noise recording
is shared. Licenses inside MUSAN noise: free-sound files public domain; sound-bible files per-file (e.g. CC BY 3.0, see its LICENSE)
— fine for augmentation since the audio is never redistributed.

**Fine-tuning prep (2026-10-24, Laptop, merged in PR #10; Priyansh asked "how can we do the fine tuning code here you dont know the results
from the gpu box": answer, training code doesn't depend on eval results, which only tune config later and arrive via git).**
Pipecat publishes Smart Turn v3.2 as ONNX only (HF pipecat-ai/smart-turn-v3: cpu=int8, gpu=fp32); training code is on GitHub
(pipecat-ai/smart-turn train.py, commit 4786657: whisper-tiny encoder with max_source_positions 400, attention pool, classifier;
lr 5e-5, 4 epochs, batch 384, warmup 0.2, wd 0.01, BCE pos_weight from batch stats; data pipecat-ai/smart-turn-data-v3.2-train,
86 files, CC BY 4.0). `pehleaap/models/smart_turn.py` rebuilds it in PyTorch from the fp32 ONNX (MatMul weights unnamed and transposed,
matched by the named bias after each; k_proj has no bias). Parity: max |p| diff 3.3e-6 on 256 real clips, 0 decision flips;
tests/test_smart_turn_torch.py. Note int8 (our baseline) vs fp32 differ by up to 0.66 on single clips at equal accuracy:
compare a fine-tune against both. Do NOT add an fp32 entry to turn_models.REGISTRY while GPU-box jobs use the default `--models`
(saved shards would be recomputed). Next: training script (bf16 on the GPU box, fp16 on Kaggle), smoke-tested on the Laptop.

**Training script (2026-10-24 evening, Laptop; PR #13, merge 148e778): GPU box Job 4 in docs/machines.md.** Train set
pipecat-ai/smart-turn-data-v3.2-train = 83 shards / 41.4 GB / 23 languages; per shard ~961 of 3,265 clips are eng+hin (~80k
total; Hindi all chirp3 TTS). `scripts/prepare_eot_train.py` → `data/external/eot-train/` (EotStore: audio.int16 memmap +
index.csv, last 8 s per clip, 3% dev by md5(id); resumable; `--delete-shards`). `pehleaap/train/eot_data.py` (EpochKeys sampler:
seeded per epoch, exact resume; Augmenter collate in workers: p_phone 0.8, train noise part, features = WhisperMelOnnx.prepare
exactly) and `train_eot.py` (TrainConfig = configs/train_eot.yaml: lr 5e-5, 4 epochs, batch 384 = 2 × 192, warmup 0.1 + cosine,
hin_repeat 2, 20 workers; dev eval at step 0 = published model, best by mean phone-line dev accuracy; ckpt every 30 min; exports
model.onnx via the legacy TorchScript exporter: no onnxscript needed, parity 6e-8). **GTX 1650 fp16 gives NaN from the first conv**
(GTX 16xx quirk) → precision "auto" picks fp32 there, bf16 on Ada, fp16 on T4; NaN loss stops the run. Augmentation ~77 ms/clip
per worker on the Laptop. `eval_clip_baselines.py --onnx NAME=PATH` scores a local ONNX at threshold 0.5 (defaults unchanged).
Smoke: 6 steps on 1 shard fine end to end (fp32 GTX 1650; 4 min mostly Windows worker start-up). 99 tests pass.

**Threshold calibration (2026-10-24 night, Laptop, uncommitted; Priyansh picked it over ASR / verify app).** `pehleaap/eval/calibration.py`
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

**ASR hookup (2026-10-25 night, Laptop, uncommitted).** `pehleaap/labeling/asr.py` + `label_session.py --asr` (words cached as
data/labels/<session>/words.jsonl; `--asr-refresh`, `--asr-model`, `--asr-language`, `--asr-batch-size`), `pehleaap/device.py`
(precision_for moved out of train_eot), `scripts/make_tts_session.py` (stand-in session from Hindi TTS test clips, in runs/).
Whisper via transformers pipeline (no new deps), per-IPU with 0.3 s pad (Whisper hallucinates Devanagari on pure silence: seen),
max_new_tokens = 20/s of the longest clip in a length-sorted batch (word timestamps keep all cross-attentions: a loop OOM'd the
GTX 1650 at batch 16), default batch 8. transformers 5.20.2's temperature fallback + word timestamps crashes (IndexError in
_extract_token_timestamps) → own retry: zlib compression ratio > 2.4 → decode again with no_repeat_ngram_size=3, keep the less
repetitive text. Smoke with whisper-tiny on a stitched TTS session: words all inside their padded IPUs; loops fixed (183 → 86
words). Large-v3 quality/speed still unmeasured: GPU box runbook step in docs/machines.md. 109 tests pass.

**Pending, week 1:** Priyansh to review the consent forms (contact details, deletion date, college ethics
approval?); Craig vs Zoom pilot with one pair (`scripts/check_alignment.py`); start recruiting and booking >=15 sessions; HingCoS access
email; PSTN provider signup (Twilio dual-channel first); LiveKit v1-mini baseline (time-boxed to 1 day; started on the GPU box, see above); Priyansh to try the ding-bot demo by voice.
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
**Backed up to the Laptop (2026-10-25 02:36, via Chrome Remote Desktop file transfer):** `weights/eot_v0/model.onnx`
(sha256 38fbe51a…a40202) and `best.pt` (step 900, sha256 158b561e…5cf40c), gitignored. Checked on the Laptop: ONNX matches
best.pt within 5e-6; clean Hindi shard 0 (n=132): eot_v0 93.9% / cut-in 3.4% / AUC 0.993 vs Smart Turn int8 93.2% / 11.9% / 0.986.
int8 quantisation of eot_v0 is being written on the GPU box (pehleaap/models/quantize.py, scripts/quantize_eot.py): don't
touch those files from the Laptop until that PR is merged. The GPU box also adds `onnx_with_state_dict()` to
pehleaap/models/smart_turn.py (eot_v0 weights grafted into Pipecat's fp32 graph, for upstream-identical quantisation).
**Also backed up (2026-10-25 07:10, uploaded by Priyansh from the GPU box over Chrome Remote Desktop, "Upload file" since Priyansh
sat at the GPU box; uploads land on the laptop's Desktop):** weights/eot_v0/model_int8.onnx (8.7 MB, sha256 d694d401…567a1a)
and model_pipecat_graph.onnx (32.4 MB, sha256 8dbd011d…36fad7). Checked on the Laptop, clean Hindi shard 0 (n=132): Pipecat
graph = fp32 within 1.5e-6; int8 acc 94.7 / cut-in 8.5% vs fp32 93.9 / 3.4% (cut-ins up at 0.5, as on the GPU box).

**Demo wired to eot_v0 (2026-10-25 ~03:00, Laptop; PR #19, merge c5aae30).** `turn_models.load_any(name)`: registry name, or
weights/<name>/model.onnx / a .onnx path at threshold 0.5 (our models stay OUT of REGISTRY: eval defaults). pipecat_ext:
`ModelSwitch` + `SwitchTurnAnalyzer` (active model decides; the others are scored on a background thread after the decision and
logged as "would say"), models cached per process. demo/bot.py: loads PEHLEAAP_TURN_MODELS (default smart_turn_v3.2_int8,eot_v0;
missing weights skipped with a warning), starts on eot_v0, `/switch` page + `/switch/state` + `POST /switch/{name}` on Pipecat's
runner `app`. Tests: 6 pass in .venv-demo (incl. the end-to-end ding test), load_any test in .venv. Live check: server up, routes
flip the active model.
**Priyansh's live test (2026-10-25 03:04-03:07, one speaker, mic → simulated G.711):** the beep (bot takes the turn) confused the
Priyansh at first: explain it as "the bot would start answering now" (no TTS yet). Clear sentence ends: both models END ~0.3 s after
speech (VAD stop 0.2 s + ~0.1 s inference; eot_v0 fp32 ~95 ms vs Smart Turn int8 ~130 ms on the laptop CPU, 1 thread). Priyansh
confirmed 3 mid-sentence pauses: eot_v0 said wait at all 3 (p 0.06/0.08/0.26) where Smart Turn said END (0.71/0.91/0.93), and
Smart Turn, while deciding, actually cut in once. Anecdotal (one speaker, a few turns), not a result to cite, but it's the
behaviour the benchmark predicts. The 3.2 s beep after a "wait" is Pipecat's silence fallback when the caller stays quiet.
Priyansh asked "were the demo results good": answered honestly (right direction, matches the benchmark, but ~20 decisions from
one speaker on a clean line; the real test is labelled real recordings). Priyansh works through the night, no breaks (2026-10-25).

**Label-check app (2026-10-25 ~03:30, Laptop; PR #20, merge f284dc5).** `pehleaap/labeling/verify.py` (needs_review: all BC/INTERRUPT/
HOLD_REQ + PSTN sessions (device "pstn" in sessions.csv) + md5(event_id) 10% of SHIFT/HOLD; double_annotated: independent 10%
hash; verdicts appended to data/labels/<session>/verify/<annotator>.jsonl, last wins; apply_verdicts: correct → verified,
wrong → retyped, "none"/unsure/unchecked → dropped; cohens_kappa) and `scripts/verify_app.py` (stdlib ThreadingHTTPServer on
127.0.0.1:8765, single page: stereo clip agent L / customer R, loudness envelopes with event marker, keys 1/2/3, S H B I Q X 0
for corrections, N/E/Space/arrows; `--all` for the pilot, `--double-only` for annotator 2, `--summary` → events_verified.jsonl +
per-type auto-label accuracy + kappa). Not Gradio (would add ~20 packages to every install; plan §4.2 notes it). NOISE_ECHO and
"none" exist only as correction labels (no automatic rule yet). Checked: 5 tests incl. a live-server test, page JS passes
`node --check`; the Chrome extension was not connected, so the page hasn't been seen rendered by Claude.

**eot_v0 int8 (GPU box, 2026-10-25 03:30):** `scripts/quantize_eot.py` -> runs/eot_v0/model_int8.onnx (8.7 MB, not in git); doc
`docs/results/week1_eot_v0_int8.md`. Upstream recipe (QDQ, entropy, Conv/MatMul/Gemm only, 1024 CLEAN train clips) on eot_v0 weights
grafted into Pipecat's fp32 graph (`onnx_with_state_dict`, matches TorchScript export within 3e-6). Dev: int8 costs 0.6-2.0 pts;
phone/hin 92.2 -> 90.1 but cut-in 8.7% -> 16.9% (dev-tuned int8 threshold not done). Pipecat's own int8 loses more (phone/hin
82.0 -> 71.8). Latency 1 thread: fp32 56.9 ms, int8 28.6 ms = Smart Turn int8. Gotcha: quantising all op types / phone-mix
calibration shifted probs -0.1..-0.2 (up to -23 pts). Test-set eval of int8 not run yet (~75 min CPU).

**Session evaluation (2026-10-25 ~04:00, Laptop; PR #22, merge 9f577bd): the main metric for real recordings.** CHAIN conditions moved
from eval_clip_baselines.py into `pehleaap/telephony/conditions.py` (identical code; the script imports them, tests pass).
`pehleaap/eval/sessions.py`: condition_track (whole track through a condition; chain seed = "<session>_<channel>"),
checkpoint_windows/probs (last 8 s up to pause start + 0.2/0.3/0.5/0.8/1.2/2.0 s; NaN once the speaker resumed).
`scripts/eval_sessions.py`: verified labels if present (else marked UNVERIFIED), --channels (default customer), per condition
and model: cut-in / median & p90 gap / missed at the model's threshold, cut-in at matched 0.5 / 0.8 s gap (threshold sweep),
silence-timeout floor (VAD on the clean recording, so condition-independent); hard timeout 3 s (Pipecat's fallback).
Smoke on a stitched 20-turn TTS session (TTS02): works, but numbers are meaningless: stitched Smart Turn clips are half
deliberately unfinished, and the stitched holds are all short (flatters the timeout). Runbook in docs/machines.md.

**NOISE_ECHO rule (2026-10-25 ~05:00, Laptop; PR #23, merge 600413b).** `pehleaap/labeling/echo.py` `split_echo`, run by label_session in
both directions (customer vs agent first, then agent vs the customer's real speech); schema NOISE_ECHO now in TYPES, Event +
IPU fields echo_lag_s / echo_corr; verify ALWAYS includes NOISE_ECHO; LabelConfig echo_* fields (yaml in sync). Frame-level
(10 ms log-loudness, 300-3400 Hz): 1 s windows vote for delay+gain (r >= 0.8, >= 3 agreeing votes within 20 ms, else no echo);
predicted echo = other track at the delay + gain with a 1 dB/10 ms reverb tail; echo only expected while the other speaker's
IPUs (delayed) are active + 0.3 s tail (bug found by the unit test: without that, the other track's noise floor x gain marked
quiet frames of real replies as echo); own frames > 6 dB above the prediction, or above it with 0.3 s local correlation < 0.6
(double-talk), are speech; speech frames re-cut into IPUs (words re-attached), echo runs >= 0.3 s -> NOISE_ECHO events.
Why frame-level: a per-IPU first version left echo merged with the customer's reply as 10-12 s fake INTERRUPTs.
Measured (simulated, Smart Turn TTS clips): echo windows r 0.85-0.99, independent speech <= 0.68 (1 s), short windows unreliable
(0.3 s free search: 23% of real speech >= 0.8); stitched TTS02 + echo 3 dB below the customer at 0.35 s: all 26 shift/hold
events recovered within 0.15 s, 0 spurious bc/interrupt, 0 echo found on echo-free TTS02; speakerphone barge-ins (customer cuts
into 7 s agent turns): 5/6 found with echo 15 dB down, 2/6 with 3 dB (AEC off: inherently ambiguous; 0/6 without the
double-talk cue; cutoff safe up to 0.7, spurious from 0.8). Tests: tests/test_echo_labels.py (random-syllable stand-in).

**Session pipeline (2026-10-25 ~05:30, Laptop; PR #24, merge 726557a).** `scripts/process_session.py` replaces the protocol's three
after-recording commands: per session (1) sessions.csv row + both speakers' consent_date in speakers.csv; **no consent =
nothing processed, not even with --force** (it reads speakers.csv only for consent, prints nothing personal); valid devices;
spk ids match track names; (2) audio health per track (sample rate >= 16 kHz, clipping > 0.1%, speech 95th-pct frame level
< -40 dBFS, speech-to-background < 20 dB, track lengths within 2 s); (3) clock drift between the sync beeps <= 10 ms via
align.measure_drift (PSTN sessions skip: one dual-channel recording); (4) label_session (with --asr on the GPU box);
(5) per-session checksums data/manifests/sessions/<session>.jsonl (replaces build_manifest('raw')); writes
data/labels/<session>/process.json; prints next steps. `--data-root` for tests. recording_protocol.md §5 and machines.md
updated. Tests: 4 (good / no consent / drift stopped unless --force / PSTN skip).
CSV templates created on the Laptop only (gitignored): data/raw/sessions.csv + speakers.csv (headers from protocol §5) and
data/raw/sessions/. Priyansh was walked through filling them (IDs only, consent_date YYYY-MM-DD, device values).

**Outreach + real-speech test (2026-10-25 ~05:15, Laptop; branch outreach-fleurs).** docs/recruiting.md: WhatsApp invite (English +
Hinglish, placeholders for name/thank-you/slots, ~1 h as in the consent form), private booking-sheet columns (device mix
50/30/20; 15 sessions booked = week-1 exit), HingCoS access email (contact unknown: corresponding author / IITG EMST lab).
recording_protocol.md §1 "Pilot day, step by step" (P01 Craig / P02 Zoom as separate sessions, stop within 10 s of the last
clap because measure_drift looks at the first/last 20 s, process_session then verify_app --all). FLEURS: google/fleurs hi_in
(ungated, CC BY 4.0) test+dev -> `scripts/make_fleurs_eot.py` -> data/external/fleurs-hi-eot/data/{test,dev}.parquet in Smart
Turn schema: 657 finished (speech end + 0.2 s) + 389 not-finished (0.2 s into one mid-sentence VAD pause >= 0.25 s, not in the
first second; chosen by md5 of the file name) = 1,046 real-voice clips, built in 1 min. eval_clip_baselines.py: with a
non-Smart-Turn --data-root the report names the data (README.md first line) and skips the card check (default output
byte-identical).
**FLEURS RESULT (Laptop, 27 min, docs/results/week1_fleurs_eot_report.md): real voices are a different world from TTS.**
Hindi n=1046, acc / cut-in / AUC: clean: Smart Turn 72.4 / 61% / 0.78, eot_v0 74.6 / 33% / 0.79 (missed-end 21%),
codewithmoin 81.6 / 37% / 0.82. noise10+AMR: ST 64.6 / 85% / 0.59, eot_v0 74.2 / 42% / 0.75, cwm 76.3 / 52% / 0.75. mixed:
ST 65.3, eot_v0 73.6, cwm 75.4. => eot_v0's phone-line robustness transfers to human voices (flat across conditions, half
of ST's cut-ins), but on clean real voices it barely out-ranks ST: it learned the line from TTS Hindi, not real prosody.
codewithmoin generalises best. Caveat: read speech, comma pauses can sound final. NEXT (agreed direction): eot_v1 with
real-voice clips in the training mix: FLEURS train split (~2.1k recordings, never used for testing) cut the same way,
then our recordings. GPU box job.
(Merged as part of PR #26, merge c9d92a7.)

**eot_v1 prep (2026-10-25 ~06:15, Laptop, uncommitted; Priyansh: "start eot_v1 with FLEURS train").** make_fleurs_eot.py
`--to-store DIR --max-pauses N` appends FLEURS train to the EotStore (refuses test/dev: they are the real-voice test set);
FLEURS train = 2,120 recordings -> 2,120 finished + 1,641 pause clips = 3,761 real Hindi (105 land in dev by md5). TrainConfig
`real_hin_repeat` (0 in configs/train_eot.yaml: eot_v0 recipe unchanged, real-voice rows excluded; 3 in the new
configs/train_eot_v1.yaml, the only difference); `epoch_repeats()`; eot_data `real_flags` + `group` -> dev groups eng / hin /
hin_real (dev_score averages phone accuracy over them, so selection rewards real voices). Laptop smoke (4 steps): published
Smart Turn on hin_real dev 68.6 clean / 66.7 phone (AUC ~0.7) vs 100 on TTS dev clips. GPU box runbook = Job 5 in
docs/machines.md (store add, train ~25 min, FLEURS eval + Smart Turn test eval, report week1_eot_v1_report.md). Caveat:
FLEURS splits have different sentences but no speaker IDs, so speaker overlap train/test can't be ruled out. The GPU box was told
(before this PR) to git pull, run the ASR check, build the FLEURS test set and pre-download FLEURS train.

**eot_v1 RESULTS (GPU box, 2026-10-25 07:04-08:33, Job 5 at f66ff32; report `docs/results/week1_eot_v1_report.md`).**
Store 77,808 -> 81,569 clips (backup of the old index: data/external/eot-train/index.pre_fleurs.csv, GPU box). Train 25 min,
1020 steps, best = step 400 (dev score 90.26; hin_real dev 93.3 clean / 86.7 phone, n=105, noisy). FLEURS real-voice test
(n=1046): clean eot_v1 **88.8** (cut-in 18%, missed 7%, AUC 0.93) vs eot_v0 74.6, codewithmoin 81.6, Smart Turn 72.8; beats cwm
by 7-10 pts in every condition (McNemar p <= 3e-8); noise10+AMR 83.4, mixed 82.8. TTS test: Hindi = eot_v0 (all |diff| <= 0.7,
n.s.); English slips (clean 95.40 -> 94.69, p 3e-4; 5 dB -1.7) but stays >= Smart Turn. VERDICT: eot_v1 is the new best model.
Caveats: read speech; FLEURS has no speaker IDs (train/test speaker overlap possible). model.onnx (sha256 0ccac636…89a52b) on the
GPU box at runs/eot_v1/: Priyansh to copy to the Laptop's weights/eot_v1/ (Chrome Remote Desktop) and add to the demo's
PEHLEAAP_TURN_MODELS. For the Oct 28 v1 checklist: eot_v1 is the recommended shipped model (Priyansh's call).
**eot_v1 int8 (GPU box, 2026-10-25 09:20-09:45; `docs/results/week1_eot_v1_int8.md`).** runs/eot_v1/model_int8.onnx (8.7 MB,
sha256 fe3c6e0a…0356f7) + model_pipecat_graph.onnx (d30e81b4…1e4f), not in git. int8 shifts Hindi probs UP (more than eot_v0's:
dev Hindi −4 to −6 at 0.5, cut-ins ~2-4x; AUC ~same). FLEURS real voices @0.5: int8 78-86 vs fp32 83-89, cut-ins 34-56%; with
Hindi dev thresholds (clean 0.89 / phone 0.75): phone conditions 81.5-85.7 at fp32's cut-in level; still >= Smart Turn int8 +11-17.
Latency 2.3x faster than fp32 (= Smart Turn int8). Guide: fp32 for the demo; int8 + Hindi phone threshold 0.75 when CPU-bound.
TTS test set (GPU box, 10:04-10:58): Hindi int8 @0.5 −2.6 to −4.9 vs fp32, cut-ins 2x; with the Hindi thresholds −1.0 to −2.0
(p 0.002-0.10), cut-ins 5-19%; English int8 @0.5 within 1.5. Beats Smart Turn int8 everywhere (+16-18 under noise).
`docs/overview.html` (2026-10-25, Priyansh asked): plain-language project overview page with framework diagram, progress, results.

**eot_v0 int8 test set (GPU box, 2026-10-25 05:30):** `docs/results/week1_eot_v0_int8.md` (test-set section). Hindi @0.5: int8 1.3-3.7
pts below fp32 (mean -2.3), cut-ins ~double. Dev-tuned int8 thresholds (clean/hin 0.81, phone/hin 0.71; English 0.37-0.39 don't help,
keep 0.5) recover half: mean -1.2. int8 still beats Smart Turn int8 everywhere (+18.0 at 5 dB, +3.3 clean, Hindi). DECIDED (Priyansh confirmed 2026-10-25):
fp32 eot_v0 where ~57 ms/1 thread is OK (demo); int8 + Hindi thresholds only when CPU-bound. weights/eot_v0/model.onnx copied on
the GPU box for eval_sessions.py.

**GPU box state, 2026-10-25 ~06:00 (main dae7135, PR #27; idle, nothing running).** Done there: Jobs 1-4, LiveKit, eot_v0
training + eval + calibration, int8 + its test-set eval (PRs #10-#27). On disk, NOT in git and not yet on the Laptop:
`runs/eot_v0/model_int8.onnx` (8.7 MB) and `model_pipecat_graph.onnx` (32.4 MB): Priyansh to copy them over by Chrome Remote Desktop
before the box goes (model.onnx + best.pt already backed up). Also there: data/external/eot-train (15 GB, 77,808 clips),
`weights/eot_v0/model.onnx` (for eval_sessions.py), runs/eot_v0/summary.orig.json (pre-dev_thresholds copy). Leftover hung python
PID 1808 (harmless; Priyansh can `Stop-Process -Id 1808 -Force`). Next GPU-box work, if it is still up: `eval_sessions.py` once real
sessions are recorded and labelled (copy labels + audio there first); optional noise round 2 only after real calls show noise matters.
(The int8 + Pipecat-graph files were copied to the Laptop at 07:10; hashes match: see the eot_v0 backup note above.)

**Splits + our recordings into training (2026-10-25 ~07:40, Laptop, uncommitted; Priyansh: "build the splits and
sessions_to_store").** `pehleaap/data/splits.py`: sessions grouped by shared speakers (union-find), each group to one split;
groups with a PSTN session -> test; rest greedy largest-first to the split furthest below 70/10/20 of minutes, ties to the more
balanced gender/device mix; freeze() writes benchmark/splits/{train,dev,test}.txt (session ids only) + lock.json (sha256 per
file); load() raises if a file changed; add_to_train() appends post-freeze sessions only if none of their speakers is in dev/test
(re-locks train only). `scripts/make_splits.py` (proposal by default; --freeze; --add-new; minutes from labels summary.json, else
25; genders from speakers.csv, never printed per person). `scripts/sessions_to_store.py`: SHIFT/HOLD of both channels -> clips
(audio to pause start + 0.2 s; HOLD with pause < tail skipped), split = the session's frozen split (EotStore.append now honours
an explicit "split"); refuses everything before the freeze and every test session; verified labels when present. Tests:
tests/test_splits.py (5). Freeze is due end of week 3 (Nov 15), before any test-set run.

**Datasets / HingCoS (2026-10-25, web check).** No open two-channel Hinglish phone-conversation dataset exists (that gap is the
benchmark's contribution). Hindi phone conversations exist only commercially (Nexdata 760 h, SpeechDataAI 2,000 h, ELRA 759 h;
Kaggle copies are 0.3 MB teasers, checked via the Kaggle API). IITG-HingCoS is PAID: INR 50,000 + 18% GST by demand draft,
research-only, no redistribution, read speech -> not pursued (FLEURS covers it); PLAN §2 table + docs/recruiting.md updated
(no email addresses in git). Send-ready consent copies made on the Laptop only (gitignored): data/raw/consent_{en,hi}_SEND.md
without the reviewer note and the Sarvam line; Priyansh still fills contact + deletion date and decides on ethics approval.
Priyansh asked whether datasets could replace recruiting: answered no; recruiting + pilot are theirs to do.

**MILESTONE CHANGE (Priyansh, 2026-10-25 ~08:00): a good, working v1 by Oct 28** = the GitHub repo + README (Priyansh: "till 28
oct i want a good working of this project we can upgrade it continously"). The 8-week plan continues after Oct 28 as upgrades.
Deferred past Oct 28: recruiting 40+ speakers, booking 15 sessions, HingCoS (paid anyway). Before Oct 28: ONE real
conversation with one friend (Priyansh agreed), processed end to end (process_session -> verify_app --all -> eval_sessions)
as the first real two-person Hinglish test. v1 checklist: eot_v1 (Job 5) results and the choice of the shipped model, the pilot
session's numbers, a README a stranger can follow (results, demo, benchmark, model download), model weights published
(GitHub release or Hugging Face: Priyansh's account and click), demo on the shipped model. The GPU box also ends ~Oct 28.

**After Oct 28 (Priyansh, 2026-10-25): Priyansh can supply more recorded conversations** for the upgrades. Told them what
qualifies: recorded fresh under docs/recording_protocol.md (consent form signed by every speaker *before* recording, adults,
scenario + fake-PII cards, sync beeps), two tracks with one speaker each (Craig/Zoom per-participant, or Twilio dual-channel
for PSTN). Not usable: real customer calls (never), and old personal calls (no prior consent, real personal details, and a
phone recorder gives one mixed track that the labeler can't split). Each new session goes through process_session; new
speakers join train via `make_splits.py --add-new` (the frozen test split never changes).

**Pilot pipeline rehearsed on the Laptop (2026-10-25)** with a synthetic session (TTS turns on two 48 kHz FLAC tracks, Craig
naming, claps at start/end, fake paperwork, a scratch --data-root): process_session OK (drift +0.0 ms, the planted 40 ms offset
measured exactly, 24 s for 5.5 min), verify_app --all serves the events and stereo audio, eval_sessions runs (82 s for 68
pauses x 2 models x 2 conditions on the Laptop; a 25-min call is ~6-8 min). ASR can't run on the Laptop (Whisper large-v3 fp32
~6 GB > 4 GB VRAM and 8 GB RAM): pilot transcripts are a GPU-box job while it lasts, else labels use duration rules only.
Found and fixed a metric bug: cut_in_at_gap interpolated from the *worst* threshold among ties (median gaps sit on the
checkpoint times, so ties are the norm); it now keeps the fewest cut-ins per gap, non-increasing with the gap. Only
eval_sessions uses it and no reported result depended on it.

**v1 release work (Laptop, 2026-10-25, after Job 5).** Shipped model = **eot_v1** (recommended in the Job 5 report; Laptop proceeded
on it, Priyansh can overrule). Added: LICENSE (BSD-2-Clause, holder = GitHub handle only), THIRD_PARTY_NOTICES.md (Smart Turn's
BSD-2 notice "Copyright (c) 2024-2025, Daily" + CC BY 4.0 data credits), `scripts/get_weights.py` (downloads from a GitHub release
tagged **models-v1**, model-files-only and never changed, assets eot_v1.onnx / eot_v0.onnx / eot_v0_int8.onnx; every file's full
sha256 pinned in the script; a test fails while any sha is missing), demo default = Smart Turn + the newest of eot_v1/eot_v0 on the
machine, README rewritten for strangers (results first, use in Pipecat/Python, demo, benchmark, retrain). Verified: our exported
model.onnx is a drop-in `LocalSmartTurnAnalyzerV3(smart_turn_model_path=...)` file (probabilities within 1e-6; test added).
Fairness gap found: FLEURS lacked deveshu / chhotu / abhishek / Smart Turn fp32 (deveshu is 93.2 on TTS Hindi, near eot_v1):
Laptop ran them (`runs/fleurs_more_baselines`). The eot_v1 copy to the Laptop and the models-v1 release were both done the same day (below).
FLEURS other baselines DONE (Laptop, 10:31; `docs/results/week1_fleurs_more_baselines_report.md`): clean acc (cut-in) deveshu 80.9 (36%),
abhishek 79.5 (49%; AUC 0.86, most noise-robust community model, 76.8 under noise/mixed), Smart Turn fp32 75.0 (36%) vs int8 72.8
(60%: int8 shifts toward "finished"), chhotu 67.2 (fails on phone audio). eot_v1 leads every model in every condition by 6.0-7.2.
On clean TTS Hindi codewithmoin edges eot_v1 (93.9 vs 93.7; in-distribution for it); README says so. models-v1 release PUBLISHED by
Priyansh (2026-10-25); get_weights.py verified against the real release (all 3 sha256 OK). eot_v1 model.onnx + best.pt copied
to the Laptop's weights/eot_v1/ (hashes match the Job 5 report). The GPU box also has an unscored eot_v1 model_int8.onnx (fe3c6e0a…).
**v1 merged (PR #36, main f9527ed) and dress-rehearsed from a fresh GitHub clone (Laptop, 2026-10-25):** get_weights.py downloads
eot_v1 from the release (sha OK); the README's Python and Pipecat snippets both run with the light .venv-demo install on real
FLEURS clips (finished p=0.987 -> reply, mid-sentence pause p=0.004 -> keep listening; Pipecat's analyzer gives the same p); in the
clone the demo tests pass (1 skip: needs Smart Turn's test set) and the full research suite passes. Shipped release stays fp32 only;
int8 eot_v1 (GPU box: needs Hindi threshold 0.75 on phone audio) stays on the roadmap until it ships with its own threshold.
Left for v1: pilot recording Oct 26 -> pilot results in the README Oct 27 -> final check + Priyansh tags v1.0 Oct 28.
**Pilot plan changed (Priyansh, 2026-10-25):** the pilot is a free-topic chat with one friend (role-play optional), two people on
Discord + Craig in separate rooms. Priyansh may supply more conversations later; told them: 2 people per recording on
Craig/Zoom per-person tracks only (no phone-recorder apps, no 3+ people in one room), and rotate pairs, since a speaker in every
session forces all sessions into one split. Laptop updated the send-ready consent copies (data/raw, gitignored): "a conversation
on any topic, ~25 min / 30 min total", no private details about third parties, and storage = "my laptop and a GPU workstation
where they are processed; the workstation copy is deleted once processing is done". So: after the pilot's --asr run on the GPU
box, DELETE the pilot audio there (before the box goes, ~Oct 28). recording_protocol.md Option A expanded (Voice Activity +
manual sensitivity, bitrate, rename Craig's username-named files before processing, 1-min test recording first).

**CHECKPOINT (Priyansh: "save till here", 2026-10-25 ~11:30, main c20f403 after PR #38).** Merged today: #31-#38 (splits, v1
milestone, cut-in fix, eot_v1 results, v1 release, craig guide, eot_v1 int8 TTS: with Hindi thresholds int8 is 1-2 pts below fp32,
beats Smart Turn int8 everywhere). Release models-v1 is live. Open items, in order:
1. GPU box Job 6 (FLEURS paired McNemar, eot_v1 vs codewithmoin / deveshu / abhishek; reproduces the Laptop's numbers) -> branch
   fleurs-significance; Priyansh runs its commit commands (push rule kept); Laptop reviews; then README's "6.0-7.2 points" line can
   cite p-values for every rival.
2. Craig setup + 1-minute test recording P00 today (files P00_SPK001_agent.flac / P00_SPK002_customer.flac in
   data/raw/sessions/P00, a P00 row in sessions.csv) -> Laptop runs process_session to check settings and the sync sound.
3. Priyansh fills Contact + deletion date in data/raw/consent_*_SEND.md, sends it; friend signs before any recording.
4. Oct 26: pilot P01 (free chat, Craig) -> process_session -> verify_app --all -> eval_sessions; GPU box --asr if still up, then
   delete the pilot audio from the GPU box.
5. Oct 27: pilot results in the README. Oct 28: fresh-clone check, Priyansh tags v1.0.

**Job 6 DONE (GPU box, 2026-10-25 10:57-11:35; branch fleurs-significance, `docs/results/week1_fleurs_significance.md`).**
Exact McNemar at each card threshold (deveshu 0.63): eot_v1 beats codewithmoin, deveshu and abhishek in all 5 FLEURS conditions,
all 15 p <= 4e-5 (Bonferroni x15 max 7e-4); closest = abhishek under noise/mixed (+6.6 / +6.0). Lead over the next best per
condition +6.0 to +7.3 (clean 7.27 unrounded; README says 7.2 from rounded values; README update is the Laptop's). Reproduction:
deveshu/abhishek match the Laptop exactly on the 4 fixed conditions; the MIXED chain differs slightly between machines for every
model (1-2 clips; cwm 75.43 Laptop vs 75.05 GPU box already in Job 5), repeatable within one machine: cause not checked
(likely a codec/ffmpeg stage). Compare models on one machine's mixed audio only.

**After Job 6 (Laptop, 2026-10-25).** PR #39 (memory checkpoint) and PR #40 (Job 6) merged, main 4091499. Laptop recomputed all
15 McNemar p-values from the b/c counts in `week1_fleurs_significance_mcnemar.json`: all match. README now cites all 15 tests
(lead 6.0-7.3, p <= 4e-5), notes the mixed row mixes machines (up to ~0.4 pts: cwm 75.43 Laptop vs 75.05 GPU box), and adds the
clips-not-independent caveat; `week1_fleurs_more_baselines_report.md` got a correction (its "same on both machines" holds only on
the 4 fixed conditions). Open item 1 of the CHECKPOINT is done; next: Craig test P00, consent placeholders, pilot P01 on Oct 26.
