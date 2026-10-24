"""Named phone-line test conditions, shared by the clip evaluation (scripts/eval_clip_baselines.py) and
the session evaluation (scripts/eval_sessions.py), so both measure under exactly the same conditions.

A condition is "clean" (the original 16 kHz audio), a codec name from `codecs.CODECS` (band-limiting
plus that codec, nothing else), or a full-chain condition in `CHAIN`. A chain condition's noise file,
offset and random draws come from a seed hashed from the clip's (or track's) id, the same in every
condition, so for instance the SNR sweep compares each clip with itself.
"""

import hashlib

import numpy as np

from pehleaap.telephony.codecs import CODECS
from pehleaap.telephony.pipeline import AugmentConfig, ChannelParams, sample_params

# name -> (description, ChannelParams fields; None = a random draw from configs/augment.yaml). Unlisted
# fields keep their defaults: G.711 mu-law, no noise, no loss.
CHAIN = {
    "tandem_amr4.75": ("AMR-NB 4.75 kbit/s, then G.711 mu-law (transcoding at the operator's gateway)", {"codec": ("amrnb_4.75", "g711_mulaw")}),
    "loss3_g711": ("G.711 mu-law, 3% of 20 ms packets lost in bursts of about 2, concealed", {"loss_rate": 0.03, "loss_burst": 2.0}),
    "agc_clip_g711": (
        "Phone AGC towards -16 dBFS (up to +20 dB), clipped at 0.25 of full scale, G.711 mu-law",
        {"agc_target_dbfs": -16.0, "agc_max_gain_db": 20.0, "clip_level": 0.25},
    ),
    "noise20_amr7.95": ("MUSAN noise at 20 dB SNR, AMR-NB 7.95", {"codec": ("amrnb_7.95",), "snr_bucket": "20", "snr_db": 20.0}),
    "noise10_amr7.95": ("MUSAN noise at 10 dB SNR, AMR-NB 7.95", {"codec": ("amrnb_7.95",), "snr_bucket": "10", "snr_db": 10.0}),
    "noise5_amr7.95": ("MUSAN noise at 5 dB SNR, AMR-NB 7.95", {"codec": ("amrnb_7.95",), "snr_bucket": "5", "snr_db": 5.0}),
    "mixed": ("One random chain per clip from configs/augment.yaml (no speakerphone echo: the clips have no agent track)", None),
}


def known(cond: str) -> bool:
    return cond == "clean" or cond in CODECS or cond in CHAIN


def describe(cond: str) -> str:
    if cond in CHAIN:
        return CHAIN[cond][0]
    return "Band-limited to 8 kHz, then this codec; nothing else" if cond in CODECS else "Original 16 kHz audio"


def chain_params(meta: dict, cond: str, noises: list[str], cfg: AugmentConfig) -> ChannelParams:
    seed = int(hashlib.md5(meta["id"].encode()).hexdigest()[:8], 16)
    fields = CHAIN[cond][1]
    if fields is None:
        return sample_params(cfg, np.random.default_rng(seed), noises, speakerphone=False)
    noise = noises[seed % len(noises)] if "snr_db" in fields else None
    return ChannelParams(seed=seed, noise=noise, **fields)
