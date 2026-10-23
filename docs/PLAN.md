# Ultraplan: `pehleaap`, a Hinglish telephony turn-taking stack

*Pehle aap* ("after you") is the project name and the Python package name.

---

## 0. Context

**Problem.** Voice agents on Indian phone lines cut callers off and stop for "haan/hmm". They also go silent at the wrong moments. Four Hinglish-specific causes:

- **Verb-final pauses.** In "main kal payment… kar dunga" the pause comes before the verb, mid-sentence.
- **Ambiguous "haan".** It can be a backchannel or a real answer.
- **Hold requests.** "ek minute" means a long silence is expected.
- **Digit and amount dictation.** People pause between groups of digits.

On top of these, 8 kHz codecs and speakerphone echo cause false barge-ins.

**Intended outcome (8 weeks, ending around Dec 20, 2026):**

1. An open training recipe and models for end-of-turn, barge-in, backchannel and hold intent on 8 kHz Hinglish telephony audio.
2. A public, speaker-disjoint benchmark with a scorer.
3. Measured CPU cost: ms per frame and concurrent streams per core.
4. A web-call demo that switches between a baseline and this project's model.

**Positioning (keep this exact framing):** "a training recipe plus a benchmark that a platform company can rerun on its own call data." It is not a production-ready model. Every claim is scoped as "on my test set of N speakers".

**Decisions locked with the user:**

| Area | Decision |
|---|---|
| Compute | **Two machines** (see §3a). The **Dev box** is this laptop: GTX 1650 4 GB, 8 GB RAM, used for code, tests, labeling UI, eval and the Pipecat demo. The **GPU box** has an RTX 4060, used for all training, batch ASR and heavy evaluation sweeps. Free Kaggle/Colab is a fallback only. |
| Recording | Bulk two-channel recording over a browser or app at 16 kHz or higher, with telephony simulated afterwards. A small real-phone (PSTN) set of about 1-2 h, used only as a test set. |
| Speakers | 40 or more distinct adult speakers, targeting about 15-20 h. |
| Demo | A web call with an 8 kHz telephony simulator in the pipeline and a baseline/ours toggle, plus recorded real-phone side-by-sides. |

**Dev box (this machine):** Windows 11, i5-11300H (4 cores / 8 threads), GTX 1650 4 GB, 8 GB RAM, Python 3.11.9 at `D:\python`, git present, ffmpeg not installed, about 460 GB free on D:. `D:\pehleaap` is empty and is not a git repo.
**GPU box:** RTX 4060, reported as "20 GB VRAM". Every RTX 4060 and 4060 Laptop has **8 GB dedicated**. The 4060 Ti exists in 8 GB and 16 GB versions. A "20 GB" figure is almost certainly Windows Task Manager's "GPU memory", which adds dedicated memory to *shared* system RAM, and shared RAM is too slow for training. **The plan assumes 8 GB dedicated.** Confirm with `nvidia-smi` on day 1 (§3a).

---

## 1. Fact check of the pasted plan (verified Oct 23, 2026)

These corrections change the baselines and the "gap" claim.

| Pasted claim | Reality | Effect on the plan |
|---|---|---|
| Smart Turn v3.2: Whisper-Tiny encoder, attention pooling, MLP head, 8M params, BSD-2, 16 kHz, last 8 s | ✅ Confirmed in `train.py` (`SmartTurnV3Model`, `max_source_positions=400`, BCE with pos_weight). The ONNX takes `input_features` (B,80,800) and outputs `logits`. | Fine-tune it directly. Keep the ONNX interface identical so the result is a drop-in for Pipecat's `LocalSmartTurnAnalyzerV3(smart_turn_model_path=...)`. |
| 92.9% acc, 8.48% FPR, 5.61% FNR, 36 ms / 51 ms p95 | These numbers come from a community CoreML card and were not verified against official sources. | Re-measure on our own hardware and test subset. Don't quote them. |
| "None target Hinglish" | ❌ **Smart Turn v3.2 lists Hindi and Marathi among its 23 languages.** Its Hindi train/test audio is reportedly all **synthetic TTS**. There are also **at least four community Hinglish Smart Turn fine-tunes**: `deveshu/hinglish-turn-detector`, `codewithmoin/smart-turn-hinglish`, `abhishek-040010/smart-turn-hinglish` and `Vishwas07/chhotu-turn-detection`. All are 16 kHz, end-of-turn only, mostly trained on TTS or synthetic audio, and none handle telephony. On one card's own 518-clip real Hinglish benchmark, *stock v3.2 beat the fine-tune*. | The gap is now **real human two-channel conversation + 8 kHz telephony + multi-task heads (barge-in, backchannel, hold) + an open benchmark**. Add the community models as cheap baselines and cite them. |
| LiveKit works on STT transcripts only | ❌ The text `MultilingualModel` (Qwen2.5-0.5B, about 400 MB) is now **deprecated**. LiveKit ships an **audio** `inference.TurnDetector` in two versions: `v1`, which runs in the cloud, and `v1-mini`, which runs locally under the LiveKit Model License. Both include Hindi. LiveKit also has **adaptive interruption handling**, a model that separates barge-ins from backchannels. It is closed and runs on LiveKit Cloud. | LiveKit `v1-mini` becomes an end-of-turn baseline (best effort). LiveKit adaptive interruption is an optional black-box barge-in baseline. |
| (not mentioned) | **Krisp VIVA** has proprietary turn-taking and interruption-prediction models, integrated in Pipecat (`KrispVivaIPUserTurnStartStrategy`). | Cite it as prior art. Use it as a baseline only if it's free to access. |
| (not mentioned) | **TurnBench** (Sesame, IEEE SLT 2026): 30 h of studio-recorded two-channel dyadic speech plus 104 h of training data. It has end-of-turn and interruption tracks. Its metrics are recall, FPR (a system must stay ≤ 0.15 to qualify) and p50 latency. The causal-only scorer is on GitHub. It finds that at onset an interruption is indistinguishable from a backchannel, and that VAP is the strongest baseline. | **Adopt TurnBench's metric definitions and causal rule.** Run our model on TurnBench dev as a check that English doesn't regress. Use its 104 h train set for pretraining the two-channel model if the license allows it. |
| IITG-HingCoS: about 25 h, phone, 9,251 sentences | ✅ It is 8 kHz, 101 speakers, read speech. Access and license are unclear. | Email the IIT Guwahati EMST lab in week 1. Use it for acoustic robustness and to make truncated "incomplete turn" negatives (§4.4). |
| Check IndicVoices | It is CC BY 4.0: 76% extempore speech and 15% conversational. No two-channel telephony subset was found. | Use it as clean Hindi speech for synthetic mixing and echo simulation only, not for turn labels. |
| Backchannel 3-class scheme (NAACL) | ✅ Inoue et al., NAACL 2025 (2025.naacl-long.367): non-backchannel / continuer / assessment, fine-tuned from VAP, scored frame-wise. | Adopt it. |
| VAP | ✅ `ErikEkstedt/VoiceActivityProjection`: stereo CPC encoder plus a GPT-style transformer with ALiBi, 50 Hz frames, runs on CPU in real time. | Tier B starts from its checkpoint. |
| Small public Hinglish dialogue data | `Kratos-AI/Natural-ASR-Samples` (CC-BY-4.0, two-person Hinglish, tiny). Paid options: Appen HIN_ASR002 (32 h of stereo Hindi telephony) and Shaip. | Use Kratos as a sanity-check set. Name the paid corpora as "what a company would plug in". |

---

## 2. System architecture

```
 customer 8 kHz ──┐                      ┌─► EOT p(turn done)
 agent TTS ref ───┼─► [Per-frame model] ─┤   (trigger)          ┌─► respond / wait / hold
 agent speaking ──┘   10 ms hop, ~1M     └─► event ──► [Event head]──► barge-in {interrupt, BC, noise/echo}
                      causal CNN-GRU          (pause or onset    ├─► hold intent
                                               during agent talk)└─► BC type {non-BC, continuer, assessment}
 partial ASR (opt) ── text features ─────────────────────────────┘        │
                                                                    [Decision policy] ──► Pipecat
```

### 2.1 Per-frame model (always on, must be cheap)

- **Inputs:** both run at native 8 kHz.
  - The customer channel as 40 log-mel bins, 25 ms window, 10 ms hop.
  - The agent TTS reference as the same 40-bin log-mel.
  - An `agent_speaking` flag.
  - Cheap echo features: per-frame coherence between the customer and agent channels, and the energy ratio.
- **Model:** 3 causal 1-D conv layers, then a 2-layer GRU (hidden 128-192). About 0.5-1.5M params. Int8 ONNX.
- **Outputs per frame:**
  - p(customer speech), which replaces Silero VAD for the customer channel.
  - p(trigger), the cue to run the event head.
  - Early barge-in logits over 3 classes.
  - p(echo).
- **Training:** VAD targets from per-channel Silero on the clean 16 kHz source. Event labels come from §4. Barge-in targets are also distilled from the event head's soft outputs.
- **Target cost:** < 0.5 ms per 10 ms frame on one core. Measure it, don't promise it.

### 2.2 Event head (Tier A core: run only on events)

- **Base:** Smart Turn v3.2 (`SmartTurnV3Model`), with its code vendored under BSD-2 and the license kept.
- **Input:** the last 8 s, band-limited to telephony and upsampled to 16 kHz for the Whisper front end. Upsampling adds no information, so we **retrain on band-limited audio** to remove the domain shift. A native 8 kHz front end is an ablation only, because it would throw away the pretrained weights.
- **Two-channel extension:** a shared Whisper-tiny encoder runs on the customer channel and on the agent channel. Each is attention-pooled, then concatenated with [cross-channel coherence stats, agent_speaking, time since agent onset, pause duration so far]. This costs two encoder passes per event. An **EOT-only export** keeps the exact Smart Turn ONNX signature (customer channel only) so it drops straight into Pipecat.
- **Heads:**
  - EOT: binary.
  - Barge-in: 3 classes (interrupt / backchannel / noise-echo), valid only while the agent is speaking.
  - Hold intent: binary.
  - BC type: 3 classes, NAACL scheme.
- **Loss:** a weighted sum, with each head masked to the events where its label exists. EOT keeps Smart Turn's pos_weight BCE. Class-balanced focal loss for the barge-in head.

### 2.3 Optional text fusion (week 5, only if on schedule)

- **Text source:** partial ASR from Sarvam `saaras:v3` streaming (telephony-tuned, `codemix` mode). The consent form must cover third-party processing.
- **Features:**
  - Trailing filler or connective: *toh, matlab, ki, aur, lekin, kyunki, jo, umm, woh*.
  - Hold-phrase regex: *ek minute/second, ruko, ruk jao, hold karo, wait karo, abhi batata/batati, dhoondh raha/rahi*.
  - Digit count so far versus the expected length: 10 for a mobile number, 12 for an Aadhaar-format number, 16 for a card.
  - Pending amount unit ("das hazaar…").
- **Fusion:** late fusion through a logistic layer on the acoustic logits. Report results with and without it.

### 2.4 Decision policy (`policy/decision.py`)

**States:** `AGENT_SPEAKING`, `LISTENING`, `CUSTOMER_PAUSED`, `HOLDING`.

- **Customer pause:** query the EOT head at 150 ms, 300 ms, 500 ms, 800 ms and 1200 ms of silence. The threshold θ(t) decreases as silence grows. A hard timeout of about 2.5 s is extended inside digit spans.
- **Hold intent:** enter `HOLDING` and suppress replies for up to `T_hold` (default 45 s). An optional soft check-in after 20 s: "ji, main wait kar rahi hoon".
- **Customer onset during agent speech:**
  - At about 150-200 ms, *duck* the TTS volume.
  - At about 300-500 ms, run the event head:
    - Interrupt: stop TTS and flush.
    - Backchannel: restore volume and continue.
    - Noise/echo: continue.
  - This deals with the TurnBench finding that interruptions and backchannels look the same at onset.
- **Cost-sensitive thresholds:** chosen from costs C_cutin, C_latency per 100 ms, C_false_stop and C_missed_bargein. They live in YAML **use-case profiles**: `collections_kyc` (patient), `support` (balanced) and `quick_faq` (snappy).

---

## 3. Repository layout (created in week 1)

```
D:\pehleaap\
  README.md  pyproject.toml  requirements*.txt  .gitignore (data/, runs/, *.wav)
  configs/   data.yaml  augment.yaml  train_eot.yaml  train_multitask.yaml  train_frame.yaml  eval.yaml  policy_profiles.yaml
  docs/      consent_en.md  consent_hi.md  recording_protocol.md  scenario_cards/  fake_pii_cards/
             labeling_guide.md  datasheet.md  model_card.md  prereg.md (frozen hypotheses)
  pehleaap/
    audio/       io.py  resample.py  vad.py (Silero wrapper, 8k+16k)  align.py (xcorr offset + drift fix)
    telephony/   codecs.py (ffmpeg: G.711 μ/A, AMR-NB modes, GSM, Opus-8k; delay compensation)
                 channel.py (300–3400 Hz bandpass, packet loss + PLC, jitter, AGC/clipping)
                 noise.py (SNR mixing)  echo.py (speakerphone leak: RIR + nonlinearity + residual-AEC)  pipeline.py
    labeling/    ipu.py  events.py (SHIFT/HOLD/BC/INTERRUPT/NOISE_ECHO/HOLD_REQ)  asr.py  lexicons.py (hi+en)
                 digits.py  verify_app.py (Gradio)  schema.py
    models/      smart_turn.py (vendored)  multitask.py  frame_model.py  export.py  quantize.py
    train/       datasets.py  train_eot.py  train_multitask.py  train_frame.py
    baselines/   silence_vad.py  smart_turn.py  hf_hinglish.py  livekit_v1mini.py  livekit_text.py  bargein_vad.py  bargein_minwords.py
    eval/        events.py  metrics.py  curves.py  breakdowns.py  bootstrap.py  compute_bench.py  report.py  run.py
    policy/      decision.py
    pipecat_ext/ turn_analyzer.py  bargein_start_strategy.py  hold_processor.py  telephony_sim.py
  demo/        bot.py (Pipecat + SmallWebRTCTransport)  client/ (web UI + baseline/ours toggle)
  recorder/    (fallback only) aiortc relay that records each track on the server clock
  notebooks/   kaggle_train_eot.ipynb  kaggle_train_multitask.ipynb  kaggle_asr.ipynb
  benchmark/   scorer/  label_schema.json  splits/  README.md
  scripts/     env_check.py  sync_data.ps1  remote.ps1 (run a job on the GPU box)  fetch_ckpt.ps1
  tests/
  requirements.txt (shared)  requirements-gpu.txt (CUDA torch build + training extras)
```

---

## 3a. Two-machine setup (develop on the Dev box, train on the GPU box)

### Specs and roles

| | **Dev box** (this laptop) | **GPU box** |
|---|---|---|
| GPU | GTX 1650, 4 GB (Turing, fp16 only) | RTX 4060, **8 GB dedicated assumed** (Ada, bf16/fp16, much faster) |
| RAM | 8 GB | Unknown; check with `env_check.py` |
| CPU | i5-11300H, 4C/8T | Unknown; check |
| Role | Write code, unit tests, the Gradio label-verification app, the eval harness on small slices, the **Pipecat demo**, the recording pilot and alignment checks, smoke runs (overfit 64 clips, 1 epoch on 1% of data), **CPU benchmark #1** | **All real training**: EOT fine-tune, multi-task head, per-frame model, Tier B VAP. **Batch ASR** (faster-whisper large-v3 `int8_float16`, about 4-5 GB, or IndicConformer). Precomputing the augmentation variants. Full evaluation sweeps (every system × condition × threshold). **CPU benchmark #2** |
| Never does | Full training runs, which would be too slow, run out of memory and block development | Interactive development. Code is edited only on the Dev box and pulled. |

### What fits in 8 GB (planned batch sizes)

All training uses AMP bf16, gradient accumulation up to Smart Turn's effective batch of 384, and gradient checkpointing if needed.

| Job | Model | Planned batch | Notes |
|---|---|---|---|
| EOT fine-tune | Whisper-tiny encoder, 8 s, (80,800) mel | 128 | About 2-3 GB at bf16. Raise to 256 if there's headroom. |
| Multi-task, two-channel | 2 encoder passes per sample | 64 | Grad-accum ×6 |
| Per-frame CNN-GRU | About 1M params, 20 s chunks | 256 | Trivial |
| Batch ASR | faster-whisper large-v3 int8_float16 | n/a | Fits; runs overnight over all sessions |
| Tier B VAP | CPC + transformer, 20 s stereo | 16-32 | Feasible; the reason Tier B is now realistic |

If `nvidia-smi` shows **16 GB** (a 4060 Ti 16 GB), double the batches and drop gradient checkpointing. Nothing else changes.

### Workflow and sync

1. **Code:** a private GitHub repo. Develop on the Dev box, `git push`, then `git pull` on the GPU box. Never edit code on the GPU box. Each training run records its git commit hash in `runs/<id>/meta.json`.
2. **Data:** large and private, so never put it in git. `data/raw` (recordings) is mastered on the Dev box, which is where recording and verification happen. `scripts/sync_data.ps1` mirrors it to the GPU box with `robocopy /MIR` over a LAN SMB share, or a portable SSD if the boxes aren't on the same network. `data/manifests/*.jsonl`, with a sha256 per file, **is** committed, so both machines can confirm they hold identical data (`env_check.py --verify-data`).
3. **Derived data** (augmented variants, mel caches, ASR outputs) is generated **on the GPU box** and stays there. Only ASR transcripts and labels (small JSONL) are synced back to the Dev box for verification in Gradio.
4. **Running jobs:** enable Windows OpenSSH Server on the GPU box. Use Tailscale if the boxes aren't on the same LAN. Then `scripts/remote.ps1 train_eot configs/train_eot.yaml` runs `ssh gpu "cd D:\pehleaap && git pull && .venv\Scripts\python -m pehleaap.train.train_eot --config ..."` in the background and logs to `runs/`. An alternative is to run Claude Code on the GPU box itself for long jobs.
5. **Checkpoints back:** `scripts/fetch_ckpt.ps1 <run_id>` copies `best.pt`, the `*.onnx` files (8-32 MB) and `metrics.csv` to the Dev box for evaluation and the demo.
6. **Environment parity:** both machines use Python 3.11.9, the shared `requirements.txt` and pinned versions. The GPU box adds `requirements-gpu.txt` (CUDA 12.x torch). `scripts/env_check.py` prints the GPU name, **dedicated VRAM**, CUDA/cuDNN, torch, onnxruntime and pipecat versions, ffmpeg AMR-NB support, RAM, CPU and free disk. Run it on both boxes on day 1 and commit the output to `docs/machines.md`.
7. **Logging:** set `WANDB_MODE=offline` by default (Smart Turn's `train.py` expects W&B) and also write local CSVs. A W&B account is optional, for viewing GPU-box runs from the Dev box.

---

## 4. Data plan (the bottleneck, so it starts in **week 1**, not week 2)

### 4.1 Recording protocol

- **Speakers and pairs:** 40+ adults (18+ only, no minors), in about 25-30 pairs. Each pair records 2 sessions of about 20-25 min and **swaps roles**: one plays the "agent", the other the "customer". That yields about 17-20 h.
- **Mix:** balance gender, region and code-mix level. Log these per speaker in `speakers.csv`, along with device, environment and phone model.
- **Scenario cards** (`docs/scenario_cards/`):
  - Loan EMI collection
  - KYC re-verification
  - Delivery address confirmation
  - Card block
  - Appointment booking
  - Insurance renewal
  - Recharge complaint
  - 5 min of free chat (natural turn-taking)

  Each card **forces these phenomena**:
  - Digit dictation, using **fake** mobile/account/Aadhaar-format/card numbers from `fake_pii_cards/`. Never use real PII.
  - Amounts ("pandrah hazaar paanch sau").
  - At least 2 hold requests ("ek minute, card dhoondh raha hoon").
  - At least 2 deliberate interruptions of a long agent monologue.
  - Natural backchannels.
- **Device conditions per session:** handset or earphones (most sessions) and speakerphone (about 20%). For speakerphone sessions, turn **off** echo cancellation so real echo leaks into the mic. The other channel is the clean reference.
- **Bulk capture tool:**
  1. Pilot on day 2 with one pair using **Craig (Discord multitrack)** and **Zoom local recording with "separate audio file per participant"**. Both record on a single clock.
  2. Pick whichever gives aligned tracks without processing (turn noise suppression off, Zoom "original sound" on).
  3. If neither works, build the fallback `recorder/`: an aiortc relay on a free-tier VM with a public IP that records each track on the server clock.
  4. Verify every session with a **clap or beep at start and end**: offset and drift must be ≤ 10 ms, checked by cross-correlation in `audio/align.py`.
- **Real-phone (PSTN) test set (about 1-2 h, ~6 pairs, test only):** a two-leg call bridged by a telephony provider with **dual-channel recording**, using the same scenario cards. Try Twilio (`record-from-answer-dual`) first, and Exotel second if KYC allows. Verify current pricing and India call routing in week 1. Budget is roughly ₹1.5-3k.
- **Consent** (`consent_en.md` / `consent_hi.md`) covers:
  - Purpose, and storage including private Kaggle datasets.
  - Third-party ASR (Sarvam), if used.
  - **A separate opt-in for public release of the audio.** Labels and features are released for everyone; audio only for those who opt in.
  - The right to withdraw.
  - DPDP Act 2023 language. No real customer calls, ever.

### 4.2 Automatic labeling (`labeling/`)

1. **Per-channel VAD → IPUs.** Run Silero on the clean 16 kHz source. Merge gaps < 200 ms and drop segments < 100 ms.
2. **ASR with word timestamps.** Run faster-whisper large-v3 or AI4Bharat IndicConformer **on the GPU box** (local, so no data goes to a third party). Sarvam is optional, with consent.
3. **Event rules** (`events.py`; the thresholds are in config):
   - **SHIFT (EOT = 1):** the customer pauses ≥ 200 ms, and the next non-BC IPU belongs to the other speaker.
   - **HOLD (EOT = 0):** the customer resumes first. The other speaker's backchannels are ignored.
   - **BC:** a listener IPU ≤ 1.0 s that matches the BC lexicon (*haan, hmm, ji, haan ji, achha, theek hai, ok, sahi, right*), or ≤ 0.6 s with no transcript, where the main speaker continues within 1.5 s. Continuer vs assessment is decided by lexicon plus prosody, then verified by a human.
   - **INTERRUPT:** the customer starts speaking while the agent speaks, with an IPU ≥ 1 s or ≥ 3 words. The agent either yields within 2 s or a competitive overlap persists.
   - **HOLD_REQ:** the hold-phrase regex on the transcript, followed by a silence ≥ 3 s. Verified by a human.
   - **NOISE_ECHO:** from real speakerphone sessions where customer-VAD fires with high coherence to the agent channel, plus synthesized cases (§4.3).
   - **Digit spans:** tag IPUs from the ASR digits (Hindi and English number words) so breakdowns are possible.
4. **Human verification** (`verify_app.py`, Gradio):
   - About 2,000 events: all BC, INTERRUPT, HOLD_REQ and PSTN events, plus a random SHIFT/HOLD sample.
   - Double-annotate 10% and report **Cohen's κ**.
   - **The test set uses verified labels only.**
5. **Schema** (`schema.py`, JSONL): `session, speaker, channel, t_event, type, bc_type, verified, codec, device, snr_bucket, digit_span, scenario`.

### 4.3 Telephony augmentation (`telephony/`)

- **Chain:** RIR (optional) → noise at SNR ∈ {clean, 20, 10, 5, 0} dB → speakerphone echo (agent channel → RIR → soft clipping → residual-AEC attenuation at ERLE 10-30 dB → mixed into the customer channel) → AGC/clipping → band-pass 300-3400 Hz → resample to 8 kHz → codec ∈ {G.711 μ-law, A-law, AMR-NB 4.75/7.95/12.2 kbps, GSM-FR, Opus 8k} → packet loss 0-5% with PLC → jitter.
- **Critical detail:** codecs add priming delay. `codecs.py` measures it by cross-correlation and removes it, so labels stay within 1 ms. There is a unit test for this.
- **Noise sources:** MUSAN, DEMAND and ESC-50 (check each license), plus about 1 h of self-recorded Indian street, traffic, TV and market noise.
- **ffmpeg:** use the gyan.dev *full* build and confirm `ffmpeg -codecs | findstr amr` lists libopencore-amrnb. If AMR is missing, fall back to WSL2 Ubuntu ffmpeg.
- **Precompute augmented variants to disk on the GPU box** (CPU-bound, run overnight) so the GPU isn't left idle waiting on augmentation during training.

### 4.4 Extra training sources

- **Smart Turn v3.2 train set:** English and Hindi slices, telephony-augmented. On the 8 GB machine, stream it or select subsets. Check the `dataset` column to confirm the Hindi audio is TTS.
- **HingCoS** (if access is granted): complete read sentences are **EOT = 1**. Sentences truncated at forced-alignment word boundaries, especially just **before the final verb**, are **EOT = 0**. This gives synthetic verb-final hold examples.
- **TurnBench 104 h train** (if licensed): pretraining for the barge-in/BC head and Tier B.
- **IndicVoices:** clean Hindi speech used as the "agent" signal in echo synthesis.

### 4.5 Splits (frozen at the end of week 3)

- **Speaker-disjoint:** about 70/10/20 train/dev/test by speaker pair, stratified by gender, region and device.
- The **PSTN set is test-only.**
- `benchmark/splits/*.txt` is hash-locked.
- `docs/prereg.md` records the hypotheses and primary metrics *before* the first test-set run.

---

## 5. Baselines (`baselines/`)

**End-of-turn:**

1. Silence-timeout VAD: Silero, with the timeout swept from 200 to 2000 ms.
2. Smart Turn v3.2 stock ONNX (fp32 and int8).
3. Community Hinglish fine-tunes: deveshu, codewithmoin and Chhotu (whichever load as ONNX/HF).
4. LiveKit audio `TurnDetector v1-mini`, local. Best effort: if it can't be driven outside `AgentSession` within 1 day, document that and drop it.
5. LiveKit text `MultilingualModel` (deprecated) over `saaras:v3` transcripts. Best effort.

**Barge-in:**

1. VAD-only: any speech ≥ N ms interrupts, swept.
2. Min-words over ASR: at least 2-3 words (mirrors Pipecat's min-words strategy).
3. LiveKit adaptive interruption: optional, black box, through the cloud dev allowance.

**Backchannel:**

1. Lexicon + duration rule.
2. VAP backchannel fine-tune (if Tier B runs).

---

## 6. Evaluation harness (`eval/`)

All evaluation is **causal**: a decision at time t uses audio ≤ t only, following TurnBench.

**End-of-turn:**

- False cut-in rate: decisions fired during HOLD pauses.
- Response gap: true EOT → decision, plus a configurable TTS start offset.
- **Curve:** sweep thresholds and timeouts to plot cut-in rate against median/p90 gap, then report each system's operating point at a fixed gap (e.g., 600 ms) and at a fixed cut-in rate (e.g., 5%).
- TurnBench-style recall, FPR (qualifies at ≤ 0.15) and p50 latency.

**Barge-in:**

- Precision and recall of "stop".
- False-stop rate on backchannels and on noise/echo, reported separately.
- Time-to-stop p50/p95.
- A curve of FPR against decision latency, which shows the onset ambiguity.

**Backchannel:** 3-class macro-F1, event-level and frame-level (NAACL protocol).

**Hold:**

- Hold-intent precision and recall.
- "Stayed quiet" rate: the agent didn't speak before the customer resumed.
- False-hold rate.

**Breakdowns:**

- Codec
- SNR bucket
- Handset / earphone / speakerphone
- Digit-dictation turns
- Hold requests
- Gender and region
- Code-mix ratio
- PSTN vs simulated

**Statistics:** 95% CIs from a **speaker-clustered bootstrap**, and paired comparisons against Smart Turn v3.2.

**Compute** (`compute_bench.py`):

- Pin to one core with `psutil.Process().cpu_affinity([0])` and set ONNX Runtime `intra_op_num_threads=1`.
- Measure ms/frame (per-frame model) and ms/event (EOT-only and two-channel heads), fp32 and int8.
- **Streams/core** = 1 / (frame_cost × 100/s + event_cost × event_rate), at a measured event rate per stream. Report the CPU model.
- Run the benchmark on **both machines' CPUs**: the laptop i5-11300H and the GPU box's CPU. Two data points are more credible than one. GPU inference numbers are reported separately, because platform companies care about CPU.

**English check:** run the EOT head on the **TurnBench dev** set to show English doesn't regress catastrophically.

**Output:** `python -m pehleaap.eval.run --config configs/eval.yaml` writes `runs/<id>/report.html` (tables, curves, audio gallery of errors) and `results.json`.

---

## 7. Week-by-week plan (week 1 = Oct 26)

| Week | Dates | Deliverables | Exit gate |
|---|---|---|---|
| **1** Foundation | Oct 26 - Nov 1 | See below | All three exit criteria below |
| **2** Data engine | Nov 2 - 8 | See below | Labels produced automatically for every recorded session |
| **3** First model | Nov 9 - 15 | See below | EOT v0 beats Smart Turn v3.2 on **dev** cut-in rate at matched gap (or the error analysis explains why) |
| **4** Multi-task | Nov 16 - 22 | See below | All four heads train; dev barge-in macro-F1 is above the lexicon/min-words baselines |
| **5** Integrate | Nov 23 - 29 | See below | A live web call works end to end, and the toggle switches systems mid-session |
| **6** Full eval | Nov 30 - Dec 6 | See below | Report regenerates with one command. **Tier B go/no-go** |
| **7** Demo | Dec 7 - 13 | See below | A stranger can run the demo from the README |
| **8** Release | Dec 14 - 20 | See below | Tagged v1.0. Tier B only if the week-6 gate passed |

**Week 1 (Foundation)**

- Repo scaffold (private GitHub), venv, ffmpeg full build, `pipecat-ai[silero,webrtc]`, onnxruntime, torch (CUDA for the GTX 1650).
- **GPU box setup:** clone the repo, create the venv plus `requirements-gpu.txt`, run `env_check.py` on both boxes (this confirms the real VRAM), set up OpenSSH or Tailscale, test the `sync_data.ps1` round trip, and run a smoke training job (Smart Turn v3.2 for 100 steps).
- Smart Turn v3.2 ONNX plus three community models running offline. LiveKit v1-mini attempted (time-boxed to 1 day).
- Eval harness skeleton with synthetic stereo unit tests.
- Pipecat SmallWebRTC bot skeleton with a telephony-sim processor.
- **In parallel:** consent forms, scenario and fake-PII cards, the Craig vs Zoom pilot, recruiting 40+ people with a session calendar, an email requesting HingCoS access, the smart-turn-data-v3.2 download, TurnBench dev, and PSTN provider signup.
- **Exit criteria:**
  1. Smart Turn v3.2's reported accuracy is reproduced on its own Hindi+English test slice to within about 1 pt.
  2. The pilot session aligns to ≤ 10 ms.
  3. At least 15 sessions are booked.

**Week 2 (Data engine)**

- Recording sessions (Diwali week, around Nov 8, is a good time for family sessions).
- Pipeline: IPU → events → ASR **on the GPU box** (overnight batches) → lexicon labels → digit spans.
- Gradio verify app.
- Full telephony augmentation chain with the codec-delay unit test.
- First PSTN sessions.
- Target: about 6 h recorded.

**Week 3 (First model)**

- Baselines evaluated on the data so far.
- **Tier A EOT v0:** fine-tune v3.2 **on the GPU box (RTX 4060)** with bf16, batch 128 and grad-accum ×3 for an effective 384, lr 5e-5, cosine schedule, 4 epochs. Smoke-test on the Dev box first. The data mix is telephony-augmented v3.2 en+hi, our SHIFT/HOLD clips (upweighted) and HingCoS truncation negatives.
- Human verification of about 2,000 events.
- **Freeze the speaker-disjoint split and `prereg.md`.**
- Target: about 12 h recorded.

**Week 4 (Multi-task)**

- Two-channel multi-task event head with the barge-in, hold and BC-type heads.
- Per-frame CNN-GRU trained and distilled from the event head.
- Recording finished (about 17-20 h plus PSTN 1-2 h).
- Echo/noise synthesis for the NOISE_ECHO class.

**Week 5 (Integrate)**

- Decision policy and use-case profiles; thresholds tuned on **dev**.
- ONNX export: the EOT drop-in in Smart Turn format, plus the multi-task model. Int8 quantization (QDQ, static, entropy calibration, as in Smart Turn).
- Pipecat extensions: a custom stop strategy, a barge-in **user-turn-start strategy** (modeled on the Krisp IP integration) and a hold processor.
- CPU benchmarks.
- Optional text fusion.

**Week 6 (Full eval)**

- All systems × all conditions on the locked test set, the PSTN set and TurnBench dev.
- Ablations: no agent channel, no telephony augmentation, no Hinglish real data, 16 kHz training vs band-limited, int8 vs fp32, with and without text fusion.
- Error analysis gallery.
- **Tier B gate:** Tier A must beat v3.2 on test cut-in at matched gap, and the demo must work.

**Week 7 (Demo)**

- Web-call demo: browser → SmallWebRTC → telephony sim (G.711 by default) → Sarvam STT → LLM (Claude Haiku 5.5 for low latency) → Hindi TTS (e.g., Sarvam Bulbul).
- **Baseline/ours toggle** with a live event-timeline panel showing p(EOT), the barge-in class and hold state.
- Scripted test calls covering digits, holds, "haan" and speakerphone.
- Recorded side-by-side videos plus PSTN clips.

**Week 8 (Release)**

- Tech report / blog (scoped claims, CIs, failure cases), model card and datasheet.
- Benchmark release: scorer, schema, splits, labels for everyone, audio for opt-in speakers.
- Kaggle notebooks for reproduction.
- Repo cleanup.
- "How to rerun on your call data" guide.

### Tier B (stretch, only after the week-6 gate)

- VAP stereo model, initialized from the ErikEkstedt checkpoint.
- Adapt it to 8 kHz by fine-tuning on telephony-augmented two-channel data: ours plus the TurnBench train set if licensed, plus any two-speaker audio.
- Add a backchannel head following NAACL 2025.
- Compare it with Tier A on the same harness. If it isn't finished by Dec 20, report it as "in progress".

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Recruiting or sessions slip | Recording starts in week 1, with weekly hour targets. The minimum viable dataset is 8 h / 20 speakers, with claims scoped down to match. |
| Track misalignment or clock drift breaks timing labels | Single-clock capture (Craig/Zoom/server), start and end claps, and an automatic drift check that rejects sessions over 10 ms. |
| Noisy weak labels | Verified-only test set, κ reported, and lexicon rules kept in config. |
| Speaker overfitting | Speaker-disjoint splits, clustered bootstrap, and per-speaker results in the appendix. |
| The GPU box is a single point of failure: unavailable, or shared with someone else | Checkpoint every epoch and resume from it. Keep a Kaggle notebook version of every training script, tested once in week 3, as a fallback. |
| The GPU box has only 8 GB VRAM, not 20 | Batch sizes in §3a assume 8 GB. Grad-accum and checkpointing absorb the rest, so no plan change is needed. |
| The two machines drift apart (data or package versions) | Manifests with sha256, pinned requirements, `env_check.py --verify-data` before every training run, and the git hash recorded per run. |
| 8 GB RAM | HF datasets in streaming/memory-mapped mode, sharded manifests, and no full-dataset loads. |
| LiveKit or Krisp baselines can't run standalone | Each is time-boxed to 1 day, then documented as "not evaluable openly". |
| Codec delay corrupts labels | Delay measured and compensated, with a unit test. |
| Over-claiming | Prereg file, a fixed claim template ("on N held-out speakers, simulated + 1-2 h real PSTN"), and failure cases shown. |
| Privacy | Adults only, written consent, fake PII cards, private storage, opt-in audio release, never customer calls. |
| Pipecat API drift (stop/start strategy class names change between versions) | Pin the pipecat version in week 1 and wrap the integrations in `pipecat_ext/`. |

---

## 9. Verification (how each stage is checked)

1. **Unit tests** (`pytest tests/`):
   - Event extraction on synthetic stereo with known SHIFT/HOLD/BC/INTERRUPT timings.
   - The codec chain preserves duration and alignment to within 1 ms.
   - The band-pass cutoff falls at about 3.4 kHz.
   - Drift correction recovers an injected 50 ppm drift.
2. **Harness sanity:** reproduce Smart Turn v3.2's accuracy on its own test slice to within about 1 pt before trusting any new number.
3. **Training sanity:** overfit 64 clips to about 100%. Dev curves are logged locally to CSV (W&B is optional; `train.py` otherwise requires `WANDB_API_KEY`).
4. **ONNX parity:** the PyTorch and ONNX fp32 logits differ by < 1e-4, and the int8 accuracy drop is ≤ 1 pt on dev. The EOT export loads in `LocalSmartTurnAnalyzerV3(smart_turn_model_path=...)` and fires in a live Pipecat call.
5. **End to end:** `python -m pehleaap.eval.run --config configs/eval.yaml` regenerates every table and plot from raw data plus the checkpoints.
6. **Live:** a scripted demo call checklist of 10 scenarios (digits, "ek minute", "haan" during agent speech, a real interruption, speakerphone echo, a silent 30 s hold). Event logs are compared against expectations, and the toggle shows the baseline failing where expected.
7. **Compute:** report single-core ms/frame, ms/event and streams/core three times, quoting the median and the CPU model.

---

## 10. First actions after approval (week 1, day 1)

1. Run `git init` in `D:\pehleaap`, then scaffold the layout in §3, add `.gitignore`, and pin requirements, including `requirements-gpu.txt` and the `scripts/` helpers.
2. Install the ffmpeg full build and verify AMR-NB. Create the venv and install torch (CUDA), onnxruntime, pipecat-ai, silero-vad, transformers, datasets, gradio, soundfile, librosa and psutil.
2b. **The user runs this on the GPU box:** clone the repo, `python scripts\env_check.py`, and paste the output back. That confirms the real VRAM, RAM and CPU. Then set up OpenSSH Server (or Tailscale) so the Dev box can launch jobs there.
3. Download the Smart Turn v3.2 ONNX and the v3.2 test set (Hindi+English slices). Write `baselines/smart_turn.py` and `eval/` with tests. Reproduce the published accuracy.
4. Draft `docs/consent_*.md`, `recording_protocol.md`, scenario cards and fake-PII cards, ready for the user to review before any recording.
5. Prepare the Craig and Zoom pilot instructions for the user to run with one pair, plus `audio/align.py` to check that recording.

