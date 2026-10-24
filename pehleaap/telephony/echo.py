"""Room reverb and speakerphone echo (plan §4.3).

On a speakerphone call the agent's voice comes out of the customer's loudspeaker, through the
room, into their mic. The phone's echo canceller removes most of it; the residual comes back
to the agent one network round trip later. That residual is what makes a voice agent hear
itself and stop talking (a false barge-in), so the NOISE_ECHO class is trained on it.

Echo path here: agent track -> telephone band (it reached the phone over the network) ->
loudspeaker soft clipping -> room impulse response -> round-trip delay -> residual level set
by the echo canceller's ERLE.

Impulse responses are synthetic (exponentially decaying noise behind a direct path at sample
0), so convolving with one adds reverb but no delay: speech stays where its labels say.
"""

import numpy as np
from scipy.signal import fftconvolve

from pehleaap.telephony.channel import bandpass
from pehleaap.telephony.noise import NOMINAL_LEVEL_DB, active_level_db


def synthetic_rir(sr: int, rt60_s: float, drr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Direct path (1.0 at sample 0) plus a noise tail decaying 60 dB in `rt60_s`, `drr_db` below it in energy."""
    n = max(int(min(rt60_s, 1.5) * sr), 2)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * 10 ** (-3 * t / rt60_s)
    tail[0] = 0.0
    tail *= np.sqrt(10 ** (-drr_db / 10) / (np.sum(tail**2) + 1e-12))
    tail[0] = 1.0
    return tail.astype(np.float32)


def reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an impulse response whose direct path is at sample 0; same length, no delay."""
    return fftconvolve(x, rir)[: len(x)].astype(np.float32)


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    """tanh loudspeaker overdrive: near 0.5 it barely bends the peaks, 3 distorts audibly. Keeps the peak level."""
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return x.astype(np.float32)
    return (np.tanh(drive * x / peak) / np.tanh(drive) * peak).astype(np.float32)


def speakerphone_echo(
    agent: np.ndarray,
    sr: int,
    level_db: float | None,
    delay_s: float,
    erle_db: float,
    drive: float,
    rt60_s: float,
    drr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The residual echo of `agent` in the customer's mic signal (same length as `agent`).

    Before the echo canceller the echo sits at `level_db` (the customer's own active speech
    level: a handheld speakerphone couples about as loud as the talker); the canceller then
    takes `erle_db` off it.
    """
    ref = soft_clip(bandpass(agent, sr), drive)
    wet = reverb(ref, synthetic_rir(sr, rt60_s, drr_db, rng))
    d = int(round(delay_s * sr))
    wet = np.concatenate([np.zeros(d, dtype=np.float32), wet])[: len(agent)]
    wet_level = active_level_db(wet, sr)
    if wet_level is None:  # the agent is silent while it could echo
        return np.zeros(len(agent), dtype=np.float32)
    target = (NOMINAL_LEVEL_DB if level_db is None else level_db) - erle_db
    return (wet * 10 ** ((target - wet_level) / 20)).astype(np.float32)
