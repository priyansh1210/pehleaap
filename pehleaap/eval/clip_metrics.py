"""Clip-level binary metrics, defined exactly as in pipecat-ai/smart-turn `benchmark.py`.

Smart Turn reports "FPR" and "FNR" as percentages of *all* samples (FP / total, FN / total),
so accuracy + FPR + FNR = 100. We keep those under `fp_pct_total` / `fn_pct_total` and also
report the conventional rates (FP / negatives, FN / positives), which are what our cut-in
analysis uses.
"""

from collections import defaultdict

import numpy as np


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) > threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": total,
        "accuracy": 100.0 * (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_pct_total": 100.0 * fp / total if total else 0.0,
        "fn_pct_total": 100.0 * fn / total if total else 0.0,
        # Conventional rates. FP here = "turn complete" predicted on an unfinished turn = cut-in.
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def grouped_metrics(labels, probs, groups, threshold: float = 0.5) -> dict[str, dict]:
    """binary_metrics per value of `groups` (e.g. language, dataset, synthetic flag)."""
    index = defaultdict(list)
    for i, g in enumerate(groups):
        index[str(g)].append(i)
    labels, probs = np.asarray(labels), np.asarray(probs)
    return {g: binary_metrics(labels[idx], probs[idx], threshold) for g, idx in sorted(index.items())}
