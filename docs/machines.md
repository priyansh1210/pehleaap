# Machines

Output of `python scripts/env_check.py --markdown` on each machine. Re-run after driver or package changes.

## Dev box (laptop)

| Item | Value |
|---|---|
| host | PRIYANSH |
| os | Windows 10 (10.0.22621) |
| python | 3.11.9 |
| cpu | 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz |
| cpu_cores | 4 physical / 8 logical |
| ram_gb | 7.8 |
| disk_free_gb | 452 |
| gpu (nvidia-smi) | NVIDIA GeForce GTX 1650, 4096 MiB, 592.82 |
| torch | 2.14.1+cu126 |
| cuda (torch build) | 12.6 |
| gpu dedicated VRAM (GB) | 4.0 |
| gpu compute capability | 7.5 |
| bf16 native | False |
| numpy | 2.4.6 |
| onnxruntime | 1.31.0 |
| transformers | 5.20.2 |
| datasets | 5.1.0 |
| soxr | 1.1.0 |
| soundfile | 0.14.0 |
| pipecat | not installed |
| ffmpeg | ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers |
| ffmpeg codecs | libopencore_amrnb, libgsm, libopus, pcm_mulaw, pcm_alaw |
| git commit | no commits |

## GPU box (RTX 4060)

Pending: on that machine, clone the repo, create the venv, then run `python scripts\env_check.py --markdown` and paste the table here. The `gpu dedicated VRAM (GB)` row is the number training batch sizes are based on.
