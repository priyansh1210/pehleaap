"""Event-level, causal end-of-turn evaluation.

Every customer pause in a recording is one event:

* HOLD  - the customer resumes speaking after `pause_dur_s`. Any decision to respond
          before that is a false cut-in.
* SHIFT - the customer has finished. The response gap is the time from pause start to
          the decision (plus TTS start latency). If the system never fires before the
          customer speaks again (`pause_dur_s`), the turn is missed.

A system is queried only at fixed checkpoints after the pause starts (e.g. 150, 300, 500 ms),
using audio up to that instant only, so the evaluation is causal (TurnBench rule). The
expensive part, model probabilities at every checkpoint, is computed once; thresholds and
timeouts are then swept cheaply to trace the cut-in vs response-gap curve.
"""

from dataclasses import dataclass, field

import numpy as np

SHIFT = "shift"
HOLD = "hold"


@dataclass
class PauseEvent:
    event_id: str
    kind: str  # SHIFT or HOLD
    pause_dur_s: float  # silence until the customer next speaks
    groups: dict = field(default_factory=dict)  # breakdown keys: speaker, codec, device, ...


def fire_times(
    probs: np.ndarray,
    checkpoints_s: np.ndarray,
    pause_durs_s: np.ndarray,
    threshold: float,
    timeout_s: float | None,
) -> np.ndarray:
    """Decision time per event (s after pause start), or NaN if the system never fires.

    probs: (N, C) probability of turn complete at each checkpoint (ignored when None).
    A checkpoint only counts if it falls inside the pause (the model can't be asked after
    the customer resumes). The silence timeout fires if no checkpoint did first.
    """
    n = len(pause_durs_s)
    out = np.full(n, np.nan)
    if probs is not None:
        valid = checkpoints_s[None, :] < pause_durs_s[:, None]
        hit = (probs >= threshold) & valid
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        out[any_hit] = checkpoints_s[first[any_hit]]
    if timeout_s is not None:
        timed_out = np.isnan(out) & (timeout_s < pause_durs_s)
        out[timed_out] = timeout_s
    return out


def operating_point(
    events: list[PauseEvent],
    probs: np.ndarray | None,
    checkpoints_s,
    threshold: float,
    timeout_s: float | None,
    tts_offset_s: float = 0.0,
) -> dict:
    kinds = np.array([e.kind for e in events])
    durs = np.array([e.pause_dur_s for e in events], dtype=float)
    t = fire_times(probs, np.asarray(checkpoints_s, dtype=float), durs, threshold, timeout_s)

    hold, shift = kinds == HOLD, kinds == SHIFT
    cut_ins = hold & ~np.isnan(t)
    gaps = t[shift & ~np.isnan(t)] + tts_offset_s
    n_shift = int(shift.sum())
    return {
        "threshold": threshold,
        "timeout_s": timeout_s,
        "n_hold": int(hold.sum()),
        "n_shift": n_shift,
        "cut_in_rate": float(cut_ins.sum() / hold.sum()) if hold.any() else float("nan"),
        "miss_rate": float((n_shift - len(gaps)) / n_shift) if n_shift else float("nan"),
        "median_gap_s": float(np.median(gaps)) if len(gaps) else float("nan"),
        "p90_gap_s": float(np.percentile(gaps, 90)) if len(gaps) else float("nan"),
    }


def threshold_curve(events, probs, checkpoints_s, thresholds, timeout_s, tts_offset_s=0.0) -> list[dict]:
    """Model systems: sweep the probability threshold at a fixed hard timeout."""
    return [operating_point(events, probs, checkpoints_s, th, timeout_s, tts_offset_s) for th in thresholds]


def timeout_curve(events, timeouts_s, tts_offset_s=0.0) -> list[dict]:
    """Silence-timeout baseline: no model, sweep the timeout."""
    return [operating_point(events, None, [], 1.0, to, tts_offset_s) for to in timeouts_s]


def cut_in_at_gap(curve: list[dict], target_gap_s: float) -> float:
    """Cut-in rate at a matched median response gap, by linear interpolation along the curve."""
    pts = sorted(
        (p["median_gap_s"], p["cut_in_rate"]) for p in curve if not np.isnan(p["median_gap_s"])
    )
    if not pts:
        return float("nan")
    gaps, rates = zip(*pts)
    return float(np.interp(target_gap_s, gaps, rates))
