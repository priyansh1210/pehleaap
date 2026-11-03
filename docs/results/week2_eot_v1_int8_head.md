# eot_v1 int8 with the last layers kept in float (GPU box, Nov 4 2026)

**Question.** int8 eot_v1 calls too many unfinished Hindi turns finished at 0.5 (`week1_eot_v1_int8.md`: clean Hindi cut-ins
6.5% -> 15.8%, phone Hindi 8.7% -> 25.0%). Does keeping the end of the network in float32 remove that shift?

**Answer: no.** Keeping the attention pool and classifier in float changes no Hindi decision. Keeping the last encoder
layer's feed-forward block too moves phone Hindi by under 1 point and costs 18% more time. The shift comes from the rest of
the encoder. int8 eot_v1 still needs the Hindi thresholds from `week1_eot_v1_int8.md` (0.75 on phone audio), and the
release stays fp32.

## Method

`scripts/quantize_eot.py --run eot_v1 --config configs/train_eot_v1.yaml --keep-last-fp32 N --out <name>`, with the
same recipe as Oct 25 (QDQ, entropy calibration on the same 1,024 clean training clips, seed 0, Conv/MatMul/Gemm only).
`--keep-last-fp32 N` leaves the last N of the graph's 39 Conv/MatMul/Gemm nodes in float: 5 = attention pool (2) +
classifier (3); 7 adds the last encoder layer's two feed-forward matmuls.

This fresh clone has only the released `model.onnx`, not `best.pt`, so the script now rebuilds the weights from it. The
frozen Whisper position table is folded into a constant in our export, so it is taken from Pipecat's file. That
reproduces Oct 25 exactly:
- `model_pipecat_graph.onnx`: sha256 d30e81b4…, same as Oct 25.
- N = 0: `model_int8.onnx` sha256 fe3c6e0a…, same file as Oct 25, with identical dev numbers.

## Dev split, threshold 0.5 (`week2_eot_v1_int8_head_quantize.json`, `week2_eot_v1_int8_head_ffn_quantize.json`)

Accuracy (cut-in rate). fp32 and N = 0 are the Oct 25 numbers (`week1_eot_v1_int8_quantize.json`).

| Dev set | n | fp32 | int8 (N = 0) | pool + classifier in float (N = 5) | + last FFN in float (N = 7) |
|---|---|---|---|---|---|
| clean / Hindi TTS | 344 | 93.9 (6.5%) | 89.8 (15.8%) | 89.8 (15.8%) | 89.8 (15.2%) |
| phone / Hindi TTS | 344 | 91.6 (8.7%) | 85.8 (25.0%) | 85.8 (25.0%) | 86.6 (23.4%) |
| clean / English | 1967 | 96.6 (4.2%) | 95.7 (5.8%) | 95.5 (6.0%) | 95.7 (5.3%) |
| phone / English | 1967 | 93.0 (8.7%) | 91.1 (12.8%) | 91.2 (12.6%) | 91.4 (11.8%) |
| File size | | 32.4 MB | 8.7 MB | 9.3 MB | 12.8 MB |

## CPU latency (batch 1, median of 100 runs, all files in one quiet run on the GPU box)

| | 1 thread | 4 threads |
|---|---|---|
| Smart Turn int8 (Pipecat's file) | 28.5 ms | 10.7 ms |
| eot_v1 fp32 | 62.1 ms | 18.4 ms |
| eot_v1 int8 (N = 0) | 29.5 ms | 10.5 ms |
| N = 5 | 30.4 ms | 10.5 ms |
| N = 7 | 35.0 ms | 11.7 ms |

The latency lines in each run's JSON aren't comparable with each other. The N = 0 run shared the CPU with another job
(fp32 93 ms there), so this table times all five files together afterwards.

## Not tried

- Keeping early layers (the two input convs, the first encoder layer) in float, or a per-layer sensitivity search.
- Calibrating on more Hindi. The calibration clips are 15% Hindi, the share in training. Calibrating on phone-line audio
  was worse on Oct 25.
- A FLEURS or TTS test-set run for N = 5 and N = 7. Dev shows no gain worth one.
