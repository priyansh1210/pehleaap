"""Decision-threshold calibration, cross-fitted so that no clip's threshold is tuned on that clip.

Jobs 1-2 showed that much of Smart Turn's phone-line drop is a shifted score distribution: the
ranking (AUC) barely moves, but more unfinished turns score above the card's 0.5. Re-picking the
threshold recovers part of that. Picking it on the clips being scored overstates the gain, so here
the clips are split in two by a hash of their id, each half's threshold is tuned on the other half,
and every clip is scored with a threshold it never influenced.

A threshold can be shared at different levels (`VARIANTS`): what a deployment can know decides
which levels are fair. It knows it is on a phone line; it may know the caller's language; it
rarely knows the codec, so per-condition thresholds are an upper bound, not a recipe.
"""

import hashlib

import numpy as np
import pandas as pd

# name -> columns that share one threshold (None: the model card's threshold)
VARIANTS: dict[str, list[str] | None] = {
    "card": None,
    "channel": ["channel"],
    "channel+language": ["channel", "language"],
    "condition+language": ["condition", "language"],
}


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold t maximizing the accuracy of `probs > t` (the middle one if several tie)."""
    labels, probs = np.asarray(labels).astype(int), np.asarray(probs, dtype=float)
    order = np.argsort(probs, kind="stable")
    ps, ys = probs[order], labels[order]
    n = len(ps)
    # Cutting before sorted index k: the first k clips are called unfinished, the rest finished.
    neg_below = np.concatenate([[0], np.cumsum(ys == 0)])
    pos_above = int((ys == 1).sum()) - np.concatenate([[0], np.cumsum(ys == 1)])
    acc = neg_below + pos_above
    k = np.arange(n + 1)
    valid = (k == 0) | (k == n) | np.concatenate([[True], ps[1:] > ps[:-1], [True]])
    best = np.flatnonzero(valid & (acc == acc[valid].max()))
    kk = int(best[len(best) // 2])
    if kk == 0:
        return float(ps[0] - 1e-6)
    if kk == n:
        return float(ps[-1])
    return float((ps[kk - 1] + ps[kk]) / 2)


def fold_of(clip_id: str) -> int:
    return int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16) % 2


def cross_fit_thresholds(df: pd.DataFrame, prob: str, by: list[str]) -> np.ndarray:
    """Per-row threshold: tuned on the rows of the same group (`by` columns) in the other fold."""
    thresholds = np.full(len(df), np.nan)
    folds = df["fold"].to_numpy()
    for _, idx in df.groupby(by, sort=False).indices.items():
        for f in (0, 1):
            tune, score = idx[folds[idx] != f], idx[folds[idx] == f]
            if len(score):
                src = tune if len(tune) else idx  # a group that lives in one fold only can't be cross-fitted
                thresholds[score] = best_threshold(df["label"].to_numpy()[src], df[prob].to_numpy()[src])
    return thresholds


def with_channel_and_fold(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(channel=np.where(df["condition"] == "clean", "clean", "phone"), fold=df["id"].map(fold_of))


def calibrate(df: pd.DataFrame, model: str, card_threshold: float, shipped: dict[str, float] | None = None) -> pd.DataFrame:
    """Accuracy, cut-in and missed-end rate per (language, condition) under each variant of `VARIANTS`.

    `shipped` adds a "shipped" variant: fixed thresholds per "channel/language" (e.g. "phone/hin") that
    came with the model, tuned on its own dev data (train_eot.py's `dev_thresholds`), so nothing here is
    tuned on these clips. Missing keys fall back to the card threshold.
    """
    df = with_channel_and_fold(df)
    prob, labels = df[f"p_{model}"].to_numpy(), df["label"].to_numpy().astype(int)
    rows = []
    variants = dict(VARIANTS) | ({"shipped": "shipped"} if shipped else {})
    for name, by in variants.items():
        if by == "shipped":
            keys = df["channel"] + "/" + df["language"]
            t = keys.map(lambda k: shipped.get(k) if shipped.get(k) is not None else card_threshold).to_numpy(dtype=float)
        else:
            t = np.full(len(df), card_threshold) if by is None else cross_fit_thresholds(df, f"p_{model}", by)
        pred = (prob > t).astype(int)
        for (lang, cond), idx in df.groupby(["language", "condition"], sort=False).indices.items():
            y, p = labels[idx], pred[idx]
            neg, pos = y == 0, y == 1
            rows.append({
                "model": model, "language": lang, "condition": cond, "variant": name, "n": len(idx),
                "accuracy": 100.0 * float((p == y).mean()),
                "cut_in": float((p[neg] == 1).mean()) if neg.any() else float("nan"),
                "missed_end": float((p[pos] == 0).mean()) if pos.any() else float("nan"),
                "threshold_median": float(np.median(t[idx])),
            })
    return pd.DataFrame(rows)


def deployment_thresholds(df: pd.DataFrame, model: str) -> dict[str, float]:
    """The per-(channel, language) thresholds tuned on all the clips: what a deployment would ship."""
    df = with_channel_and_fold(df)
    return {f"{ch}/{lang}": best_threshold(g["label"], g[f"p_{model}"]) for (ch, lang), g in df.groupby(["channel", "language"])}
