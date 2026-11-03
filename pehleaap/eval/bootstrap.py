"""Cluster bootstrap: confidence intervals that resample speakers, not events (plan §6, docs/prereg.md).

Pauses from one speaker are not independent: a slow talker or a noisy line moves all of them together.
Resampling events treats 300 pauses from 10 people as 300 independent samples and makes the interval far
too narrow. So each draw picks clusters (speakers) with replacement and keeps every event of each pick.
"""

from typing import Callable

import numpy as np


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters, n_boot: int = 10_000, seed: int = 0,
                      level: float = 0.95) -> dict:
    """`stat(idx)` on the full data and on `n_boot` cluster resamples; `idx` indexes the events, `clusters` gives
    each event's cluster id. Returns the estimate, a percentile interval and the share of draws above zero."""
    clusters = np.asarray(clusters)
    ids, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    draws = np.array([stat(np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])) for _ in range(n_boot)])
    ok = draws[~np.isnan(draws)]
    lo, hi = np.percentile(ok, [50 * (1 - level), 50 * (1 + level)]) if len(ok) else (np.nan, np.nan)
    return {"estimate": float(stat(np.arange(len(clusters)))), "ci_low": float(lo), "ci_high": float(hi), "level": level,
            "p_above_zero": float((ok > 0).mean()) if len(ok) else float("nan"), "n_clusters": int(len(ids)),
            "n_events": int(len(clusters)), "n_boot": n_boot, "n_nan_draws": int(len(draws) - len(ok)), "seed": seed}
