"""The phone itself and the network around the codec (plan §4.3): AGC, clipping, band-pass,
packet loss with concealment, and jitter.

Every function keeps the length and timing of its input, so turn labels carry over unchanged:
the band-pass is zero-phase, the AGC only changes gain, and lost packets are filled in place.

Jitter: a fixed jitter buffer plays packets on schedule, so a packet delayed past the buffer's
depth arrives too late and is concealed like a lost one. Adaptive buffers that stretch silences
instead are not modelled, since they would move every later turn boundary.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, sr: int, lo_hz: float = 300.0, hi_hz: float = 3400.0) -> np.ndarray:
    """Telephone band, zero-phase (forward-backward Butterworth), so it adds no delay."""
    sos = butter(4, [lo_hz, hi_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def agc(
    x: np.ndarray,
    sr: int,
    target_dbfs: float = -20.0,
    max_gain_db: float = 15.0,
    attack_s: float = 0.01,
    release_s: float = 1.0,
    frame_s: float = 0.01,
) -> np.ndarray:
    """Causal automatic gain control, as in a phone's mic path.

    Tracks the signal power with a fast attack and slow release and steers it towards
    `target_dbfs`. In pauses the gain creeps up (to at most `max_gain_db`) and lifts the
    background noise, as real phones do; that pumping is one cause of false barge-ins.
    """
    hop = int(frame_s * sr)
    n = -(-len(x) // hop)
    padded = np.zeros(n * hop, dtype=np.float64)
    padded[: len(x)] = x
    power = (padded.reshape(n, hop) ** 2).mean(axis=1)
    a_att, a_rel = 1 - np.exp(-frame_s / attack_s), 1 - np.exp(-frame_s / release_s)
    env = np.empty(n)
    e = max(power[0], 10 ** (target_dbfs / 10))  # start at the target, not from silence
    for i, p in enumerate(power):
        e += (a_att if p > e else a_rel) * (p - e)
        env[i] = e
    gain_db = np.minimum(target_dbfs - 10 * np.log10(env + 1e-12), max_gain_db)
    # Per-sample gain: interpolate between frame centres.
    centres = (np.arange(n) + 0.5) * hop
    gain = 10 ** (np.interp(np.arange(len(x)), centres, gain_db) / 20)
    return (x * gain).astype(np.float32)


def hard_clip(x: np.ndarray, level: float) -> np.ndarray:
    """Clip at `level` (fraction of full scale), as an overdriven mic preamp or ADC does."""
    return np.clip(x, -level, level).astype(np.float32)


def gilbert_loss(n_packets: int, rate: float, mean_burst: float, rng: np.random.Generator) -> np.ndarray:
    """Bursty packet loss (two-state Gilbert model): True = lost. Long-run loss is `rate`, bursts average `mean_burst` packets."""
    if rate <= 0 or n_packets == 0:
        return np.zeros(n_packets, dtype=bool)
    p_recover = 1.0 / max(mean_burst, 1.0)
    p_fail = min(rate * p_recover / (1 - rate), 1.0)
    lost = np.empty(n_packets, dtype=bool)
    state = rng.random() < rate
    u = rng.random(n_packets)
    for i in range(n_packets):
        lost[i] = state
        state = u[i] >= p_recover if state else u[i] < p_fail
    return lost


def late_packets(n_packets: int, mean_jitter_ms: float, buffer_ms: float, rng: np.random.Generator, packet_ms: float = 20.0) -> np.ndarray:
    """Packets that miss a fixed jitter buffer: True = late (concealed like a lost packet).

    Each packet's extra network delay is exponential with mean `mean_jitter_ms`, but packets
    arrive in order, so one can't come sooner than `packet_ms` after the one before it: a
    delay spike (a cell link stalling) makes the packets queued behind it late too.
    """
    raw = rng.exponential(mean_jitter_ms, n_packets) if mean_jitter_ms > 0 else np.zeros(n_packets)
    delay = np.empty(n_packets)
    d = 0.0
    for i in range(n_packets):
        d = max(raw[i], d - packet_ms)
        delay[i] = d
    return delay > buffer_ms


def _pitch_period(hist: np.ndarray, sr: int) -> int:
    """Pitch period (samples) of the end of `hist`, by normalised autocorrelation over 66-400 Hz (10 ms if too little history)."""
    lo, hi, win = int(sr / 400), int(sr / 66), int(0.02 * sr)
    if len(hist) < hi + win:
        return int(0.01 * sr)
    ref = hist[-win:]
    best, best_score = int(0.01 * sr), -np.inf
    for lag in range(lo, hi + 1):
        past = hist[len(hist) - win - lag : len(hist) - lag]
        score = np.dot(ref, past) / (np.sqrt(np.dot(past, past)) + 1e-9)
        if score > best_score:
            best, best_score = lag, score
    return best


def conceal(x: np.ndarray, lost: np.ndarray, sr: int, packet_s: float = 0.02) -> np.ndarray:
    """Fill lost packets in place (packet loss concealment, after ITU-T G.711 Appendix I).

    A lost stretch repeats the last pitch period before it, at full level for 10 ms, then
    fading out by 60 ms (longer losses go silent). The first good packet after a loss is
    cross-faded from the concealment over 4 ms, so there's no click.
    """
    y = x.astype(np.float32).copy()
    hop = int(packet_s * sr)
    fade_start, fade_end, ola = int(0.01 * sr), int(0.06 * sr), int(0.004 * sr)
    i = 0
    while i < len(lost):
        if not lost[i]:
            i += 1
            continue
        j = i
        while j < len(lost) and lost[j]:
            j += 1
        s, e = i * hop, min(j * hop, len(y))
        hist = y[:s]
        if s == 0 or not np.any(hist):
            y[s:e] = 0.0
        else:
            period = hist[-_pitch_period(hist, sr) :]
            m = e - s + ola
            k = np.arange(m)
            gain = np.clip(1 - (k - fade_start) / (fade_end - fade_start), 0, 1)
            fill = np.resize(period, m) * gain
            y[s:e] = fill[: e - s]
            tail = min(ola, len(y) - e)
            if tail > 0:
                w = np.linspace(1, 0, tail, endpoint=False)
                y[e : e + tail] = w * fill[e - s : e - s + tail] + (1 - w) * y[e : e + tail]
        i = j
    return y
