# Why the mixed phone chain differed between our machines: Opus (GPU box, Nov 4 2026)

Since Job 5 (Oct 25), fp32 models scored 0 to 4 clips differently on the "mixed phone chain" on the Laptop and the GPU
box. The four fixed conditions matched exactly (`week2_fleurs_one_machine_report.md`).

**Cause: the ffmpeg build.** The GPU box runs ffmpeg 9.0.1 (winget, on PATH). The Laptop runs 9.0.2 (the runbook's
build). The two builds encode Opus differently. The mixed chain is the only condition that uses Opus. It is one of the
seven codecs it draws from.

**Codec check.** `scripts/codec_parity.py` ran all seven codecs on the first 20 FLEURS Hindi test clips with each
build, on the GPU box:
- G.711 μ-law and A-law, AMR-NB 4.75 / 7.95 / 12.2 and GSM come out bit-identical.
- Opus differs: largest sample difference 0.13, correlation 0.993, same delay (1 sample). The audio isn't shifted, so no
  label moves.
- Each build gives identical output when run twice.

**Score check.** The same FLEURS run with ffmpeg 9.0.2 on the GPU box (clean + mixed,
`week2_fleurs_ffmpeg902_results.json`) reproduces the Laptop's mixed numbers at full precision. So for fp32 models the
ffmpeg build is the whole difference; the CPU plays no part.

| mixed chain | GPU box, ffmpeg 9.0.1 (README) | GPU box, ffmpeg 9.0.2 | Laptop, ffmpeg 9.0.2 |
|---|---|---|---|
| deveshu | 70.36 | 70.55 | 70.55 |
| Smart Turn fp32 | 68.83 | 68.55 | 68.55 |
| abhishek, cut-in rate | 47.6% | 47.3% | 47.3% |
| codewithmoin | 75.05 | 75.43 | 75.43 |
| eot_v1 | 82.79 | 82.89 | not run |

Clean audio is identical with both builds for every model.

**What to do.**
- Results stay comparable within one ffmpeg build. The README's table is from 9.0.1 on the GPU box.
- Keep 9.0.1 first on the GPU box's PATH so its earlier results reproduce. 9.0.2 is also unzipped in `~\tools` there,
  but it comes later in `ffmpeg_bin`'s lookup order, so it is only used through `PEHLEAAP_FFMPEG`.
- int8 models (chhotu) also differ between the two CPUs on every condition, a separate effect (likely onnxruntime's
  int8 kernels; not checked).
- Check a new ffmpeg build with:
  ```
  python scripts/codec_parity.py --ffmpeg <old ffmpeg.exe> --out runs/codecs_old.npz
  python scripts/codec_parity.py --ffmpeg <new ffmpeg.exe> --out runs/codecs_new.npz
  python scripts/codec_parity.py --compare runs/codecs_old.npz runs/codecs_new.npz
  ```

Re-score command (GPU box, `--threads 20`, 20 min):

```
$env:PEHLEAAP_FFMPEG = "$HOME\tools\ffmpeg-9.0.2-full_build-shared\bin\ffmpeg.exe"
.venv\Scripts\python scripts\eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin
    --models deveshu_fp32 abhishek_tiny_attn codewithmoin_8s
    --onnx eot_v1=weights\eot_v1\model.onnx smart_turn_v3.2_fp32=weights\pipecat-ai__smart-turn-v3\smart-turn-v3.2-gpu.onnx
    --conditions clean mixed --threads 20 --out fleurs_mixed_ffmpeg902
```
