"""NOISE_ECHO: find "speech" on one channel that is really the other speaker's voice echoing back.

On a speakerphone the phone plays the agent out loud and its own mic picks that up, so the customer's
track carries a delayed, reverberant copy of the agent. VAD hears it as speech, and the event rules
would turn it into backchannels and interruptions nobody made. The echo usually runs straight into
the customer's reply (the agent's last words echo while the customer starts answering), so VAD
merges the two into one IPU: deciding per IPU is not enough, so this works on 10 ms frames.

1. The call's echo delay and gain: in 1 s windows of own speech, the 10 ms log-loudness of echo
   follows the other speaker's at a fixed delay. Windows correlating >= `echo_min_corr` at their
   best delay (0-0.8 s) vote; at least `echo_min_windows` agreeing votes mean the call has echo.
   Simulated (telephony/echo.py vs independent Smart Turn clips, 2026-10-25): echo windows correlate
   0.85-0.99 with the delay found within 8 ms, independent speech stays <= 0.68.
2. The predicted echo loudness in every frame: the other track at that delay plus the gain, each
   frame fading 1 dB per 10 ms afterwards (room reverb).
3. Own frames more than `echo_margin_db` louder than that prediction are own speech, and so are
   frames above it whose loudness over 0.3 s no longer follows the other speaker (correlation below
   `echo_doubletalk_corr`: someone is talking over the echo). The rest of the IPU, where echo was
   predicted, is echo. Speech frames become the new IPUs, echo runs of at least `echo_min_s` become
   NOISE_ECHO segments.
On a stitched 2-min Hindi TTS session with echo 3 dB below the customer (0.35 s delay): 99.8% of
the customer's real speech frames kept, 0.8% of echo frames left as speech, and every reply start
within 0.13 s of its echo-free position. Speakerphone barge-ins (customer cutting into a 7 s agent
turn): 5 of 6 found as INTERRUPT with the echo 15 dB below the customer, but only 2 of 6 with it
3 dB below (AEC off), where talking over the echo is hard to tell from echo by loudness. Every
NOISE_ECHO event goes to human verification.
"""

import numpy as np

from pehleaap.labeling.ipu import IPU, merge_segments

HOP_S = 0.01


def loudness(x: np.ndarray, sr: int) -> np.ndarray:
    """log10 power of the 300-3400 Hz band in 10 ms frames."""
    from pehleaap.telephony.channel import bandpass

    y = bandpass(x.astype(np.float32), sr)
    hop = int(HOP_S * sr)
    n = len(y) // hop
    return np.log10((y[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(axis=1) + 1e-10)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def _activity(ipus: list[IPU], n: int) -> np.ndarray:
    act = np.zeros(n, dtype=bool)
    for u in ipus:
        act[int(u.start / HOP_S) : int(np.ceil(u.end / HOP_S))] = True
    return act


def echo_delay(own: np.ndarray, other: np.ndarray, own_act: np.ndarray, other_act: np.ndarray, cfg) -> tuple[int, float] | None:
    """(delay in frames, gain in log10 power) of the other speaker's echo in `own`, or None if there is none."""
    win, hop = int(cfg.echo_window_s / HOP_S), int(cfg.echo_window_s / HOP_S) // 4
    max_lag = int(round(cfg.echo_max_delay_s / HOP_S))
    votes = []
    for s in range(max_lag, len(own) - win, hop):
        if own_act[s : s + win].mean() < 0.9:
            continue
        c, best, best_k = own[s : s + win], -1.0, 0
        for k in range(max_lag + 1):
            if other_act[s - k : s - k + win].mean() >= cfg.echo_min_other_active:
                r = _corr(c, other[s - k : s - k + win])
                if r > best:
                    best, best_k = r, k
        if best >= cfg.echo_min_corr:
            votes.append((best_k, float(np.median(c - other[s - best_k : s - best_k + win]))))
    if not votes:
        return None
    lag = int(np.median([k for k, _ in votes]))
    agree = [g for k, g in votes if abs(k - lag) <= round(cfg.echo_lag_spread_s / HOP_S)]
    return (lag, float(np.median(agree))) if len(agree) >= cfg.echo_min_windows else None


def predicted_echo(other: np.ndarray, lag: int, gain: float, tail_db_per_10ms: float) -> np.ndarray:
    """Expected log-loudness of the echo in each own frame: the other track `lag` frames earlier, plus reverb."""
    shifted = np.concatenate([np.full(lag, other.min()), other[: len(other) - lag]])
    decay, run = tail_db_per_10ms / 10.0, -np.inf
    pred = np.empty_like(shifted)
    for t, v in enumerate(shifted):
        run = max(v, run - decay)
        pred[t] = run
    return pred + gain


def _local_corr(a: np.ndarray, b: np.ndarray, width: int) -> np.ndarray:
    """Pearson correlation of a and b in a window of `width` frames centred on each frame."""
    from numpy.lib.stride_tricks import sliding_window_view

    half = width // 2
    pa, pb = np.pad(a, half, mode="edge"), np.pad(b, half, mode="edge")
    wa, wb = sliding_window_view(pa, 2 * half + 1), sliding_window_view(pb, 2 * half + 1)
    da, db = wa - wa.mean(axis=1, keepdims=True), wb - wb.mean(axis=1, keepdims=True)
    den = np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    return np.where(den > 1e-12, (da * db).sum(axis=1) / np.maximum(den, 1e-12), 0.0)[: len(a)]


def _segments(mask: np.ndarray) -> np.ndarray:
    """[[start_s, end_s], ...] of the runs of True frames."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges.reshape(-1, 2) * HOP_S if len(edges) else np.zeros((0, 2))


def split_echo(own_audio: np.ndarray, other_audio: np.ndarray, sr: int, own_ipus: list[IPU], other_ipus: list[IPU], cfg):
    """(own speech IPUs, echo segments, echo delay in s or None). Without echo the IPUs come back unchanged."""
    own, other = loudness(own_audio, sr), loudness(other_audio, sr)
    n = min(len(own), len(other))
    own, other = own[:n], other[:n]
    own_act, other_act = _activity(own_ipus, n), _activity(other_ipus, n)
    found = echo_delay(own, other, own_act, other_act, cfg)
    if found is None:
        return own_ipus, [], None
    lag, gain = found
    pred = predicted_echo(other, lag, gain, cfg.echo_tail_db_per_10ms)
    floor = np.percentile(own, 10)
    # Echo only where the other speaker was really talking `lag` earlier (plus a reverb tail), never on their noise floor.
    talking = np.concatenate([np.zeros(lag, dtype=bool), other_act[: n - lag]])
    tail = int(cfg.echo_tail_s / HOP_S)
    talking = np.convolve(talking.astype(np.int8), np.ones(tail + 1, dtype=np.int8))[:n] > 0  # extend each run by the tail
    echo_expected = talking & (pred > floor + cfg.echo_margin_db / 10)
    louder = own > pred + cfg.echo_margin_db / 10
    if cfg.echo_doubletalk_corr is not None:  # also own speech: above the prediction while no longer following the other
        local = _local_corr(own, np.concatenate([np.full(lag, other.min()), other[: n - lag]]), int(cfg.echo_local_s / HOP_S))
        louder |= (own > pred) & (local < cfg.echo_doubletalk_corr)
    speech = own_act & ~(echo_expected & ~louder)
    echo = own_act & ~speech
    speech_ipus = merge_segments(_segments(speech), cfg.merge_gap_s, cfg.min_ipu_s)
    echo_ipus = merge_segments(_segments(echo), cfg.merge_gap_s, cfg.echo_min_s)
    for u in echo_ipus:
        i, j = int(u.start / HOP_S), int(u.end / HOP_S)
        u.echo_lag_s = round(lag * HOP_S, 3)
        u.echo_corr = round(_corr(own[i:j], other[i - lag : j - lag]), 3) if i - lag >= 0 else None
    return speech_ipus, echo_ipus, round(lag * HOP_S, 3)
