"""Labeling thresholds (plan §4.2). `configs/labeling.yaml` holds the frozen copy used for the benchmark."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LabelConfig:
    # VAD -> IPUs: Silero on the clean 16 kHz source, then merge short gaps and drop blips.
    vad_sr: int = 16000
    vad_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    merge_gap_s: float = 0.2  # gaps shorter than this are inside one IPU
    min_ipu_s: float = 0.1

    # SHIFT / HOLD: a speaker's pause, labeled by who speaks next (backchannels ignored).
    min_pause_s: float = 0.2
    # The other speaker starting this close before a pause counts as an early turn start
    # (SHIFT with negative gap), not as an interruption.
    terminal_overlap_s: float = 0.3

    # BC: short listener IPU while the other speaker holds the floor.
    bc_max_s: float = 1.0  # with a transcript matching the BC lexicon
    bc_max_notext_s: float = 0.6  # without a transcript
    bc_context_s: float = 1.5  # the other speaker spoke within this long before the BC
    bc_continue_s: float = 1.5  # ... and continues within this long after it

    # INTERRUPT: onset inside the other speaker's IPU, long enough to be a real attempt.
    interrupt_min_s: float = 1.0
    interrupt_min_words: int = 3
    yield_within_s: float = 2.0  # the other speaker stops within this long: yielded
    competitive_overlap_s: float = 1.0  # otherwise, overlap at least this long: competitive

    # HOLD_REQ: hold phrase in the transcript, then the speaker stays quiet this long.
    hold_req_min_silence_s: float = 3.0

    # Digit spans: at least this many consecutive number words (or digits) in one IPU.
    digit_min_run: int = 3

    # NOISE_ECHO (labeling/echo.py): frames whose loudness the other speaker's delayed voice explains are their echo.
    echo_max_delay_s: float = 0.8  # network round trip + room
    echo_window_s: float = 1.0  # windows that vote for the call's echo delay (shorter ones match speech by chance)
    echo_min_corr: float = 0.8  # simulated: echo 0.85-0.99, independent speech <= 0.68 in 1 s windows
    echo_min_windows: int = 3  # agreeing votes needed to say the call has echo at all
    echo_lag_spread_s: float = 0.02  # votes this close to the median delay agree
    echo_min_other_active: float = 0.5  # the other speaker must be talking for this share of the delayed window
    echo_margin_db: float = 6.0  # own frames this much louder than the predicted echo are own speech
    echo_tail_db_per_10ms: float = 1.0  # reverb: a predicted echo frame fades this fast (an RT60 of 0.6 s)
    echo_tail_s: float = 0.3  # echo is expected only while the other speaker talks (delayed) and this long after
    echo_min_s: float = 0.3  # shortest echo run reported as a NOISE_ECHO event
    # Double-talk: above the predicted echo while the 0.3 s loudness no longer follows the other speaker is own speech too
    # (simulated: safe up to 0.7, spurious events from 0.8; None turns it off).
    echo_doubletalk_corr: float | None = 0.6
    echo_local_s: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> LabelConfig:
    """Defaults, overridden by a YAML file (unknown keys raise TypeError)."""
    if path is None:
        return LabelConfig()
    return LabelConfig(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))
