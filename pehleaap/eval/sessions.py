"""End-of-turn models on recorded conversations: the project's main metric (plan §6).

Every labelled pause of the evaluated channel is one event: SHIFT (they had finished) or HOLD (they
carried on). Each model is asked at fixed checkpoints after the pause starts, given only the last 8 s
of audio up to that instant, as a live bot would be (the demo asks 0.2 s after Silero VAD hears
silence). Before that the whole track goes through the phone-line condition: one call, one line.
`eot_events` then turns the probabilities into cut-in rate against response gap, the same way for
every model and for the plain silence-timeout baseline.

Pause starts come from the labels, i.e. VAD on the clean recording, not from a VAD running on the
degraded line; that's the same for every model, so the comparison stays fair.
"""

import numpy as np

from pehleaap.labeling.schema import Event
from pehleaap.telephony.codecs import CODECS, narrowband_many
from pehleaap.telephony.conditions import CHAIN, chain_params
from pehleaap.telephony.pipeline import AugmentConfig, phone_channel

CHECKPOINTS_S = (0.2, 0.3, 0.5, 0.8, 1.2, 2.0)  # seconds after the pause starts
WINDOW_S = 8.0  # what the models read


def condition_track(x: np.ndarray, sr: int, cond: str, track_id: str, bank=None, cfg: AugmentConfig | None = None) -> np.ndarray:
    """A whole track through a named condition (`telephony.conditions`), same length and timing."""
    if cond == "clean":
        return x
    if cond in CODECS:
        return narrowband_many([x], cond, sr)[0]
    if cond in CHAIN:
        return phone_channel(x, chain_params({"id": track_id}, cond, bank.names, cfg or AugmentConfig()), sr, bank=bank)
    raise ValueError(f"unknown condition {cond!r}")


def checkpoint_windows(x: np.ndarray, sr: int, events: list[Event], checkpoints_s=CHECKPOINTS_S):
    """(event index, checkpoint index, the audio a model sees then) for every checkpoint inside its pause."""
    for i, e in enumerate(events):
        for j, c in enumerate(checkpoints_s):
            if e.pause_s is not None and c >= e.pause_s:
                continue  # they're speaking again: a live bot would no longer be asking
            end = int(round((e.t_event + c) * sr))
            yield i, j, x[max(0, end - int(WINDOW_S * sr)) : end]


def checkpoint_probs(model, x: np.ndarray, sr: int, events: list[Event], checkpoints_s=CHECKPOINTS_S) -> np.ndarray:
    """(n_events, n_checkpoints) P(turn complete); NaN where the pause had already ended."""
    probs = np.full((len(events), len(checkpoints_s)), np.nan)
    found = list(checkpoint_windows(x, sr, events, checkpoints_s))
    if found:
        p = model.predict_batch([w for _, _, w in found])
        for (i, j, _), v in zip(found, p):
            probs[i, j] = v
    return probs
