# Is eot_v1's lead on real voices significant? Paired tests against every community model (GPU box, Oct 25 2026)

Job 6. `week1_fleurs_more_baselines_report.md` (Laptop) showed eot_v1 ahead of every model on FLEURS real Hindi by 6.0-7.2
points over the next best, but gave p-values only for codewithmoin (`week1_eot_v1_report.md`). This run scores eot_v1 and the
three community models that come closest on the same clips, and tests each pair.

Command (GPU box, 24 threads, 19 min):
`scripts/eval_clip_baselines.py --data-root data\external\fleurs-hi-eot --langs hin --models codewithmoin_8s deveshu_fp32
abhishek_tiny_attn --onnx eot_v1=runs\eot_v1\model.onnx --conditions clean g711_mulaw amrnb_7.95 noise10_amr7.95 mixed --out
fleurs_v1_vs_community`. Files: `week1_fleurs_significance_results.json` (all metrics), `week1_fleurs_significance_mcnemar.json`
(every test below).

Method: the same as Job 5. Each model makes a decision on each clip at its own model-card threshold (eot_v1, codewithmoin and
abhishek 0.5, deveshu 0.63). On the same 1,046 clips, count the clips only eot_v1 gets right (b) and only the other model gets
right (c). Then an exact two-sided McNemar test: binomial test of b against b + c at 0.5. The accuracies computed from
`predictions.csv` were checked against the eval's own `results.json` for every row.

## Results (FLEURS Hindi, n = 1046 per condition; accuracy %)

| Condition | eot_v1 | vs codewithmoin | vs deveshu | vs abhishek |
|---|---|---|---|---|
| clean | 88.8 | 81.6: **+7.3**, 123 vs 47, p = 5e-9 | 80.9: **+7.9**, 134 vs 51, p = 9e-10 | 79.5: **+9.4**, 151 vs 53, p = 4e-12 |
| G.711 μ-law | 86.7 | 80.7: **+6.0**, 117 vs 54, p = 2e-6 | 79.9: **+6.8**, 125 vs 54, p = 1e-7 | 78.3: **+8.4**, 150 vs 62, p = 1e-9 |
| AMR-NB 7.95 | 86.9 | 80.5: **+6.4**, 122 vs 55, p = 5e-7 | 77.5: **+9.4**, 133 vs 35, p = 1e-14 | 78.4: **+8.5**, 140 vs 51, p = 9e-11 |
| noise 10 dB + AMR 7.95 | 83.4 | 76.3: **+7.1**, 127 vs 53, p = 3e-8 | 71.1: **+12.2**, 185 vs 57, p = 6e-17 | 76.8: **+6.6**, 157 vs 88, p = 1e-5 |
| mixed chain | 82.8 | 75.1: **+7.7**, 141 vs 60, p = 1e-8 | 70.4: **+12.4**, 197 vs 67, p = 5e-16 | 76.8: **+6.0**, 148 vs 85, p = 4e-5 |

"123 vs 47" = clips only eot_v1 gets right vs clips only the other model gets right.

- **eot_v1 beats each of the three in every condition**, and all 15 tests are significant. The weakest is abhishek on the mixed chain
  (p = 4e-5). With a Bonferroni correction for all 15 tests, the largest p is 7e-4.
- **The closest rival is abhishek under noise** (+6.6 at 10 dB, +6.0 on the mixed chain). There eot_v1 still wins nearly 2 to 1 on the clips where they
  disagree (157 vs 88, 148 vs 85). It is the most noise-robust community model, as the Laptop report found.
- **Lead over the next-best model in each condition** (codewithmoin on clean and codec-only audio, abhishek under noise and on the mixed chain): **+6.0 to
  +7.3 points, p ≤ 4e-5.** The README's "6.0-7.2 points" comes from rounded table values. From the unrounded accuracies, the clean
  gap is 7.27.

## Reproduction check against the Laptop and Job 5

- **The four fixed conditions** (clean, G.711, AMR-NB 7.95, noise 10 dB + AMR): deveshu and abhishek match the Laptop's
  `week1_fleurs_more_baselines_results.json` exactly (accuracy, cut-in, missed-end, AUC, precision, recall; largest difference
  2e-6 in deveshu's clean AUC). codewithmoin and eot_v1 match Job 5's GPU-box run exactly.
- **The mixed chain differs slightly between machines, for every model:** deveshu 70.36 here against 70.55 on the Laptop (2 clips),
  and abhishek's cut-in rate 47.6% against 47.3% (1 clip). This is not new: codewithmoin's mixed score was 75.43 on the Laptop
  (`week1_fleurs_eot_results.json`) and 75.05 on the GPU box. On one machine the mixed chain repeats exactly (eot_v1 and codewithmoin are
  identical across two GPU-box runs). So the mixed chain's audio is not bit-identical across machines. The likely cause is a
  codec stage or ffmpeg build that differs between the two machines; that hasn't been checked. All comparisons in this
  report use one machine's audio for every model.

## Read with care

- FLEURS is read speech, and its splits have no speaker IDs (`week1_eot_v1_report.md`). These tests say eot_v1 is better on *these*
  clips. They don't show it is better on conversation.
- The tests treat clips as independent. Several clips come from the same reader (and each recording gives up to two clips), so
  the true uncertainty is somewhat larger than these p-values suggest. The margins (p ≤ 4e-5 everywhere) leave room for that.
