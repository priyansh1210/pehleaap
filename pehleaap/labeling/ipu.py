"""Inter-pausal units (IPUs): one speaker's stretches of speech, from VAD segments.

Gaps shorter than `merge_gap_s` are joined and segments shorter than `min_ipu_s` dropped, so
every pause between two IPUs of one speaker is at least `merge_gap_s` long.
"""

from dataclasses import dataclass

import numpy as np

from pehleaap.audio.vad import speech_segments
from pehleaap.labeling.config import LabelConfig


@dataclass
class IPU:
    start: float
    end: float
    text: str | None = None  # None until ASR has run; "" if ASR heard no words in it
    n_words: int | None = None
    echo_lag_s: float | None = None  # set when labeling/echo.py finds it is the other speaker's echo, not speech
    echo_corr: float | None = None  # ... and how closely its loudness follows theirs at that delay

    @property
    def dur(self) -> float:
        return self.end - self.start


def merge_segments(segs: np.ndarray, merge_gap_s: float, min_dur_s: float) -> list[IPU]:
    ipus: list[IPU] = []
    for start, end in segs:
        if ipus and start - ipus[-1].end < merge_gap_s:
            ipus[-1].end = max(ipus[-1].end, end)
        else:
            ipus.append(IPU(float(start), float(end)))
    return [u for u in ipus if u.dur >= min_dur_s]


def ipus_from_probs(probs: np.ndarray, hop: float, cfg: LabelConfig) -> list[IPU]:
    segs = speech_segments(probs, hop, cfg.vad_threshold, cfg.vad_neg_threshold)
    return merge_segments(segs, cfg.merge_gap_s, cfg.min_ipu_s)


def attach_words(ipus: list[IPU], words: list[dict]) -> None:
    """Give each IPU the ASR words ({"start", "end", "word"}) whose midpoint falls inside it.

    Marks the channel as transcribed: IPUs with no words get text "" (ASR heard nothing).
    """
    mids = np.array([(w["start"] + w["end"]) / 2 for w in words])
    for u in ipus:
        inside = [words[i]["word"].strip() for i in np.flatnonzero((mids >= u.start) & (mids < u.end))]
        u.text = " ".join(w for w in inside if w)
        u.n_words = len(u.text.split())
