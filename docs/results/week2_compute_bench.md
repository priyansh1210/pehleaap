# CPU cost per decision, one core (GPU box, Nov 3 2026)

`scripts/compute_bench.py` pins itself to one core, runs every model single-threaded on the same 8 s of audio, and reports
the median of 3 rounds of 50 decisions. A decision is what a live agent pays at each pause: Whisper features of the last
8 s, then one ONNX run. Silero VAD runs on every 32 ms frame of the call on top of that.

**Streams per core** = 1000 / (VAD ms per second of audio + ms per decision x decisions per second), with an *assumed*
20 decisions (pauses) per minute. Measure the real pause rate on the recorded sessions and pass `--decisions-per-min`.

Intel Core Ultra 9 285 (24 cores), Windows 11, onnxruntime 1.31.0, idle machine. Silero VAD at 16 kHz: 2.3 ms per
second of audio on either core.

| Model | Core 0: ms per decision | model only | streams per core | Core 23: ms per decision | model only | streams per core |
|---|---|---|---|---|---|---|
| Smart Turn v3.2 int8 (Pipecat's default) | 31.3 | 28.9 | 78 | 29.3 | 27.0 | 83 |
| Smart Turn v3.2 fp32 | 57.4 | 55.0 | 47 | 58.5 | 56.4 | 46 |
| **eot_v1** (fp32) | 56.5 | 54.0 | 47 | 56.0 | 53.4 | 48 |

- eot_v1 costs the same as Smart Turn's fp32 file, and about 1.8x Smart Turn's int8 file. An int8 eot_v1 would close
  that gap (`docs/results/week1_eot_v1_int8.md`: 2.3x faster than fp32), once it ships with thresholds of its own.
- The model run is almost the whole cost; Whisper's features add about 2.5 ms.
- One core of this CPU carries about 47 calls with eot_v1, or about 80 with Smart Turn int8, at 20 pauses a minute. That is
  an upper bound: it counts VAD and the turn model only, not Pipecat's own per-frame work, STT, the LLM or TTS.
- Second CPU still to do: run the same command on the Laptop (i5-11300H) and add its row here.

Command (once per core):

```
.venv\Scripts\python scripts\compute_bench.py --models smart_turn_v3.2_int8 weights\pipecat-ai__smart-turn-v3\smart-turn-v3.2-gpu.onnx eot_v1 --core 0
```
