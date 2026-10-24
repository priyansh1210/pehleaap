# LiveKit turn-detector v1-mini on Smart Turn's test set (GPU box, Oct 24 2026)

LiveKit's local audio end-of-turn model, scored by `scripts/eval_livekit_v1mini.py` (own venv,
`requirements-livekit.txt`) on the same clips and phone-line conditions as Jobs 1 and 2. Command:
`eval_livekit_v1mini.py --workers 12` (8 min, CPU only), plus a clean-only rerun with `--pad-ms 200`.

**License.** The LiveKit Model License only allows the model to be used with the LiveKit Agents framework and
forbids using its outputs to improve or develop other models. The script drives it through livekit-agents'
public `inference.TurnDetector` API only (not inside a full AgentSession; the project owner accepted that
reading), and these predictions are comparison numbers, never training data or labels.

**Protocol.** LiveKit scores the last 1.2 s of 16 kHz audio once its VAD has heard 200 ms of silence, and
treats the turn as finished unless P(end of turn) is below a per-language threshold (Hindi 0.305, English 0.36).
Each clip is fed as it is, the same protocol as Smart Turn (Pipecat also runs it after 200 ms of silence).
Parallel streams give bit-identical results to a single stream.

Scope: this is Smart Turn's own test set (in-distribution for Smart Turn, not for LiveKit), all Hindi clips are
synthetic TTS, and the phone line is narrowband + codec only.

## Summary: LiveKit v1-mini vs Smart Turn v3.2 (phone subset: all Hindi + 25% of English)

| Lang | Condition | LiveKit acc | LiveKit cut-in | LiveKit AUC | Smart Turn acc | Smart Turn cut-in | Smart Turn AUC |
|---|---|---|---|---|---|---|---|
| hin | clean | 55.14 | 93.7% | 0.708 | 90.26 | 17.7% | 0.982 |
| hin | G.711 μ-law | 58.10 | 85.9% | 0.703 | 86.37 | 27.5% | 0.971 |
| hin | AMR-NB 7.95 | 59.74 | 82.6% | 0.694 | 82.79 | 35.3% | 0.962 |
| hin | AMR-NB 4.75 | 59.27 | 83.7% | 0.697 | 80.53 | 40.0% | 0.951 |
| eng | clean | 56.24 | 83.2% | 0.725 | 92.96 | 9.4% | 0.979 |
| eng | AMR-NB 4.75 | 61.17 | 64.7% | 0.690 | 90.76 | 13.7% | 0.969 |

Sensitivity, clean, all clips: as is, AUC Hindi 0.708 / English 0.712 and cut-in 93.7% / 83.0%. With 200 ms of
silence appended, AUC 0.699 / 0.648, cut-in 22.8% / 9.6% but missed ends 59.8% / 76.6%.

Findings:
- **v1-mini barely separates finished from unfinished turns on this set:** AUC about 0.70 in every condition,
  against 0.95-0.98 for Smart Turn. At LiveKit's thresholds it calls 94% of unfinished Hindi turns finished.
  In a LiveKit agent that means it seldom extends the endpointing delay, so the agent mostly falls back to its
  plain silence timeout.
- **Trailing silence swings its scores but not its ranking.** With 200 ms of silence it calls most turns
  unfinished instead, and AUC stays at 0.65-0.70. The clip ending is not the cause.
- **Its accuracy "improves" on phone lines (+1.4 to +4.9 points)** only because codecs push its scores down, past
  a threshold set far too low for this data. AUC falls slightly (0.708 → 0.69-0.71 on Hindi).
- **Reading:** a weak baseline on short-clip classification, with a home-ground caveat for Smart Turn. It stays in the
  benchmark as the LiveKit reference point and should be rechecked on our own two-track recordings, where the
  test is out-of-distribution for every model.

---

## Full run report

## 1. Clean, all clips

| Lang | n | Acc | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|
| eng | 7820 | 56.45 | 0.830 | 0.025 | 0.712 |
| hin | 1284 | 55.14 | 0.937 | 0.000 | 0.708 |

## 2. Phone-line stress test (all Hindi + 25% hash sample of English)

- `g711_mulaw`: Band-limited to 8 kHz, then this codec; nothing else
- `g711_alaw`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_4.75`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_7.95`: Band-limited to 8 kHz, then this codec; nothing else
- `amrnb_12.2`: Band-limited to 8 kHz, then this codec; nothing else
- `gsm_fr`: Band-limited to 8 kHz, then this codec; nothing else
- `opus_8k`: Band-limited to 8 kHz, then this codec; nothing else

| Lang | Condition | n | Acc | Δ acc vs clean | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|---|---|
| eng | clean | 1947 | 56.24 | +0.00 | 0.832 | 0.027 | 0.725 |
| eng | g711_mulaw | 1947 | 59.58 | +3.34 | 0.692 | 0.105 | 0.695 |
| eng | g711_alaw | 1947 | 60.04 | +3.80 | 0.687 | 0.101 | 0.698 |
| eng | amrnb_4.75 | 1947 | 61.17 | +4.93 | 0.647 | 0.119 | 0.690 |
| eng | amrnb_7.95 | 1947 | 60.45 | +4.21 | 0.621 | 0.160 | 0.683 |
| eng | amrnb_12.2 | 1947 | 60.97 | +4.73 | 0.608 | 0.164 | 0.692 |
| eng | gsm_fr | 1947 | 59.12 | +2.88 | 0.699 | 0.107 | 0.682 |
| eng | opus_8k | 1947 | 59.27 | +3.03 | 0.681 | 0.123 | 0.687 |
| hin | clean | 1284 | 55.14 | +0.00 | 0.937 | 0.000 | 0.708 |
| hin | g711_mulaw | 1284 | 58.10 | +2.96 | 0.859 | 0.015 | 0.703 |
| hin | g711_alaw | 1284 | 58.02 | +2.88 | 0.862 | 0.013 | 0.708 |
| hin | amrnb_4.75 | 1284 | 59.27 | +4.13 | 0.837 | 0.012 | 0.697 |
| hin | amrnb_7.95 | 1284 | 59.74 | +4.60 | 0.826 | 0.013 | 0.694 |
| hin | amrnb_12.2 | 1284 | 59.42 | +4.28 | 0.823 | 0.022 | 0.700 |
| hin | gsm_fr | 1284 | 56.54 | +1.40 | 0.893 | 0.013 | 0.691 |
| hin | opus_8k | 1284 | 58.41 | +3.27 | 0.859 | 0.009 | 0.706 |

## Sensitivity run: clean, 200 ms of silence appended

| Lang | n | Acc | Cut-in rate | Missed-end rate | AUC |
|---|---|---|---|---|---|
| eng | 7820 | 57.56 | 0.096 | 0.766 | 0.648 |
| hin | 1284 | 57.94 | 0.228 | 0.598 | 0.699 |