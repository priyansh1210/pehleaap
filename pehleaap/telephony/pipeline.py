"""The full phone-line augmentation chain for the customer channel (plan §4.3).

    room reverb (optional) -> noise at an SNR bucket -> speakerphone echo of the agent track (optional)
    -> AGC -> clipping (optional) -> 300-3400 Hz band-pass -> 8 kHz -> codec(s) -> packet loss
    + jitter, concealed -> back to the input rate

Every stage keeps length and timing (the echo is the agent's voice delayed on purpose, not the
customer's), so the labels of the clean track hold for the augmented one. The agent channel
is the bot's own audio and is never augmented.

`sample_params` draws a `ChannelParams` from `AugmentConfig`; the params plus their seed fix
the output exactly, so an augmented example can be rebuilt from its metadata.
`phone_channel_many` runs the codec stage for many clips in one ffmpeg run per codec.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from pehleaap.audio.io import resample
from pehleaap.telephony.channel import agc, bandpass, conceal, gilbert_loss, hard_clip, late_packets
from pehleaap.telephony.codecs import CODECS, SR, apply_codec_many, fit_length
from pehleaap.telephony.echo import reverb, speakerphone_echo, synthetic_rir
from pehleaap.telephony.noise import NoiseBank, active_level_db, mix_at_snr

CLEAN = "clean"


@dataclass(frozen=True)
class AugmentConfig:
    # Ranges are (low, high), drawn uniformly. Weights need not sum to 1.
    # Room: the customer's own speech is reverberant with this probability (speakerphone: always).
    p_reverb: float = 0.3
    rt60_s: tuple[float, float] = (0.15, 0.6)
    drr_db: tuple[float, float] = (0.0, 12.0)  # direct-to-reverberant ratio of the customer's speech

    # Noise: SNR bucket (dB below the speech's active level), then +-snr_spread_db around it.
    snr_buckets: tuple[str, ...] = (CLEAN, "20", "10", "5", "0")
    snr_weights: tuple[float, ...] = (0.3, 0.25, 0.2, 0.15, 0.1)
    snr_spread_db: float = 2.5

    # Speakerphone echo (only when an agent track is given): residual after the phone's echo canceller.
    p_speakerphone: float = 0.2  # plan §4.1: about 20% of sessions
    echo_delay_s: tuple[float, float] = (0.1, 0.5)  # network round trip on Indian mobile networks
    erle_db: tuple[float, float] = (10.0, 30.0)
    echo_drive: tuple[float, float] = (0.5, 3.0)  # loudspeaker overdrive (tanh)
    echo_drr_db: tuple[float, float] = (3.0, 12.0)  # loudspeaker and mic share a device: strong direct path

    # Phone mic path.
    p_agc: float = 0.7
    agc_target_dbfs: tuple[float, float] = (-24.0, -16.0)
    agc_max_gain_db: tuple[float, float] = (6.0, 20.0)
    p_clip: float = 0.1
    clip_level: tuple[float, float] = (0.2, 0.6)  # fraction of full scale
    band_hz: tuple[float, float] = (300.0, 3400.0)

    # Codec; a mobile leg is often transcoded to G.711 at the operator's gateway (tandem coding).
    codecs: tuple[str, ...] = ("g711_mulaw", "g711_alaw", "amrnb_12.2", "amrnb_7.95", "amrnb_4.75", "gsm_fr", "opus_8k")
    codec_weights: tuple[float, ...] = (0.25, 0.1, 0.2, 0.15, 0.1, 0.1, 0.1)
    p_tandem_g711: float = 0.5
    tandem_codec: str = "g711_mulaw"

    # Network: 20 ms packets, bursty loss, and late packets that miss a fixed jitter buffer.
    packet_s: float = 0.02
    loss_rate: tuple[float, float] = (0.0, 0.05)
    loss_burst: tuple[float, float] = (1.0, 3.0)  # mean burst length, packets
    p_jitter: float = 0.3
    jitter_ms: tuple[float, float] = (10.0, 40.0)  # mean extra delay
    # Buffer depth as a multiple of jitter_ms: 3.5x leaves 3-7% of packets late, 6x under 1%.
    jitter_buffer_x: tuple[float, float] = (3.5, 6.0)

    def __post_init__(self):
        if len(self.snr_buckets) != len(self.snr_weights) or len(self.codecs) != len(self.codec_weights):
            raise ValueError("snr_buckets/codecs and their weights must have the same length")
        unknown = (set(self.codecs) | {self.tandem_codec}) - set(CODECS)
        if unknown:
            raise ValueError(f"unknown codecs: {sorted(unknown)}")


def load_config(path: str | Path | None = None) -> AugmentConfig:
    """Defaults, overridden by a YAML file (unknown keys raise TypeError; lists become tuples)."""
    if path is None:
        return AugmentConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AugmentConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


@dataclass(frozen=True)
class ChannelParams:
    """One draw of the chain. None switches a stage off."""

    seed: int
    codec: tuple[str, ...] = ("g711_mulaw",)  # applied in order
    snr_bucket: str = CLEAN
    snr_db: float | None = None
    noise: str | None = None
    rt60_s: float | None = None  # the room; reverb on the customer's speech needs drr_db too
    drr_db: float | None = None
    echo_delay_s: float | None = None
    erle_db: float | None = None
    echo_drive: float = 1.0
    echo_drr_db: float = 6.0
    agc_target_dbfs: float | None = None
    agc_max_gain_db: float = 15.0
    clip_level: float | None = None
    band_hz: tuple[float, float] = (300.0, 3400.0)
    packet_s: float = 0.02
    loss_rate: float = 0.0
    loss_burst: float = 1.0
    jitter_ms: float | None = None
    jitter_buffer_ms: float = 60.0

    @property
    def speakerphone(self) -> bool:
        return self.echo_delay_s is not None

    def label_fields(self) -> dict:
        """The augmentation fields of the event schema (`labeling/schema.py`)."""
        return {"codec": "+".join(self.codec), "snr_bucket": self.snr_bucket}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


def sample_params(cfg: AugmentConfig, rng: np.random.Generator, noises: list[str], speakerphone: bool | None = None) -> ChannelParams:
    """Draw a chain. `speakerphone`: True/False forces echo on/off (False for clips without an agent track); None draws it."""
    def u(lo_hi):
        return float(rng.uniform(*lo_hi))

    def pick(options, weights):
        w = np.asarray(weights, dtype=float)
        return options[int(rng.choice(len(options), p=w / w.sum()))]

    speaker = bool(rng.random() < cfg.p_speakerphone) if speakerphone is None else speakerphone
    bucket = pick(cfg.snr_buckets, cfg.snr_weights)
    codec = pick(cfg.codecs, cfg.codec_weights)
    chain = (codec, cfg.tandem_codec) if not codec.startswith("g711") and rng.random() < cfg.p_tandem_g711 else (codec,)
    reverberant = speaker or rng.random() < cfg.p_reverb
    p = dict(
        seed=int(rng.integers(2**31)),
        codec=chain,
        snr_bucket=bucket,
        rt60_s=u(cfg.rt60_s) if reverberant else None,
        drr_db=u(cfg.drr_db) if reverberant else None,
        band_hz=tuple(cfg.band_hz),
        packet_s=cfg.packet_s,
        loss_rate=u(cfg.loss_rate),
        loss_burst=u(cfg.loss_burst),
    )
    if bucket != CLEAN:
        p.update(snr_db=float(bucket) + u((-cfg.snr_spread_db, cfg.snr_spread_db)), noise=noises[int(rng.integers(len(noises)))])
    if speaker:
        p.update(echo_delay_s=u(cfg.echo_delay_s), erle_db=u(cfg.erle_db), echo_drive=u(cfg.echo_drive), echo_drr_db=u(cfg.echo_drr_db))
    if rng.random() < cfg.p_agc:
        p.update(agc_target_dbfs=u(cfg.agc_target_dbfs), agc_max_gain_db=u(cfg.agc_max_gain_db))
    if rng.random() < cfg.p_clip:
        p.update(clip_level=u(cfg.clip_level))
    if rng.random() < cfg.p_jitter:
        jitter = u(cfg.jitter_ms)
        p.update(jitter_ms=jitter, jitter_buffer_ms=jitter * u(cfg.jitter_buffer_x))
    return ChannelParams(**p)


# Each stage draws from its own stream of the params' seed, so switching one stage off
# (an ablation) leaves the others' randomness unchanged.
_RIR, _NOISE, _ECHO, _LOSS, _JITTER = range(5)


def _stage_rng(p: ChannelParams, stage: int) -> np.random.Generator:
    return np.random.default_rng([p.seed, stage])


def _before_codec(x: np.ndarray, sr: int, p: ChannelParams, agent: np.ndarray | None, bank: NoiseBank | None) -> np.ndarray:
    """Acoustic and phone stages at the input rate, then down to 8 kHz."""
    y = x.astype(np.float32)
    if p.drr_db is not None:
        y = reverb(y, synthetic_rir(sr, p.rt60_s, p.drr_db, _stage_rng(p, _RIR)))
    level = active_level_db(y, sr)
    if p.snr_db is not None:
        if bank is None or bank.sr != sr:
            raise ValueError(f"noise {p.noise!r} needs a NoiseBank at {sr} Hz")
        y = mix_at_snr(y, bank.segment(p.noise, len(y), _stage_rng(p, _NOISE)), p.snr_db, level)
    if p.speakerphone:
        if agent is None or len(agent) != len(x):
            raise ValueError("speakerphone params need the agent track, same length as the customer's")
        rt60 = p.rt60_s if p.rt60_s is not None else 0.3
        y = y + speakerphone_echo(agent, sr, level, p.echo_delay_s, p.erle_db, p.echo_drive, rt60, p.echo_drr_db, _stage_rng(p, _ECHO))
    if p.agc_target_dbfs is not None:
        y = agc(y, sr, p.agc_target_dbfs, p.agc_max_gain_db)
    if p.clip_level is not None:
        y = hard_clip(y, p.clip_level)
    y = bandpass(np.clip(y, -1.0, 1.0), sr, *p.band_hz)
    return resample(y, sr, SR)


def _after_codec(y8: np.ndarray, p: ChannelParams) -> np.ndarray:
    """Packet loss and late packets at 8 kHz, concealed."""
    n_packets = -(-len(y8) // int(p.packet_s * SR))
    lost = gilbert_loss(n_packets, p.loss_rate, p.loss_burst, _stage_rng(p, _LOSS))
    if p.jitter_ms is not None:
        lost |= late_packets(n_packets, p.jitter_ms, p.jitter_buffer_ms, _stage_rng(p, _JITTER), p.packet_s * 1000)
    return conceal(y8, lost, SR, p.packet_s) if lost.any() else y8


def phone_channel_many(
    clips: list[np.ndarray],
    params: list[ChannelParams],
    sr: int = 16000,
    agents: list[np.ndarray | None] | None = None,
    bank: NoiseBank | None = None,
    out_sr: int | None = None,
) -> list[np.ndarray]:
    """The chain on many customer clips (each with its own params), same lengths, at `out_sr` (default `sr`).

    Clips that share a codec chain go through ffmpeg together.
    """
    out_sr = out_sr or sr
    agents = agents or [None] * len(clips)
    coded = [_before_codec(x, sr, p, a, bank) for x, p, a in zip(clips, params, agents)]
    for chain in {p.codec for p in params}:
        idx = [i for i, p in enumerate(params) if p.codec == chain]
        group = [coded[i] for i in idx]
        for codec in chain:
            group = apply_codec_many(group, codec)
        for i, y in zip(idx, group):
            coded[i] = y
    out = []
    for x, p, y8 in zip(clips, params, coded):
        y = _after_codec(y8, p)
        n = round(len(x) * out_sr / sr)
        out.append(fit_length(resample(y, SR, out_sr), n))
    return out


def phone_channel(
    x: np.ndarray, params: ChannelParams, sr: int = 16000, agent: np.ndarray | None = None, bank: NoiseBank | None = None, out_sr: int | None = None
) -> np.ndarray:
    """The chain on one customer track or clip."""
    return phone_channel_many([x], [params], sr, [agent], bank, out_sr)[0]
