"""Event label schema (one JSON object per line) and the bridge to the event-level EOT metrics."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pehleaap.eval.eot_events import HOLD, SHIFT, PauseEvent

# SHIFT: the speaker paused and the other speaker took the turn (EOT = 1).
# HOLD:  the speaker paused and resumed first; the other's backchannels don't count (EOT = 0).
BC = "bc"  # short listener response while the other speaker holds the floor
INTERRUPT = "interrupt"  # the speaker started a real attempt to talk inside the other's IPU
HOLD_REQ = "hold_req"  # "ek minute": a hold phrase, then the speaker stays quiet for a while
NOISE_ECHO = "noise_echo"  # "speech" on this channel that is really the other speaker's voice echoing back (speakerphone)
TYPES = (SHIFT, HOLD, BC, INTERRUPT, HOLD_REQ, NOISE_ECHO)


@dataclass
class Event:
    channel: str  # whose behaviour the event describes: "agent" or "customer"
    type: str
    t_event: float  # s from session start: pause start (shift/hold/hold_req) or IPU onset (bc/interrupt)
    session: str = ""
    speaker: str | None = None
    # shift/hold/hold_req: from t_event until this channel next speaks, not counting its own
    # backchannels (or until the recording ends).
    pause_s: float | None = None
    # shift/hold: t_event -> next non-BC speech by anyone. For a shift that is the other speaker's
    # response offset, negative if they started just before the pause.
    gap_s: float | None = None
    ipu_s: float | None = None  # bc/interrupt/hold_req: duration of the event's IPU
    overlap_s: float | None = None  # bc/interrupt: overlap with the other channel's speech
    yield_s: float | None = None  # interrupt: the other speaker stopped this long after onset (None: they didn't)
    bc_type: str | None = None  # continuer / assessment; None without a transcript
    text: str | None = None  # transcript of the event's IPU (shift/hold: the IPU before the pause)
    digit_span: bool | None = None  # that IPU is digit dictation; None without a transcript
    verified: bool = False
    device: str | None = None
    scenario: str | None = None
    codec: str = "clean"
    snr_bucket: str = "clean"
    echo_lag_s: float | None = None  # noise_echo: the echo's delay behind the other speaker
    echo_corr: float | None = None  # noise_echo: how closely its loudness follows the other speaker's (0-1)

    @property
    def event_id(self) -> str:
        return f"{self.session}_{self.channel}_{self.type}_{round(self.t_event * 1000):07d}"


def write_events(path: str | Path, events: list[Event]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps({"event_id": e.event_id, **asdict(e)}, ensure_ascii=False) + "\n")


def read_events(path: str | Path) -> list[Event]:
    names = {f.name for f in fields(Event)}
    with open(path, encoding="utf-8") as f:
        return [Event(**{k: v for k, v in json.loads(line).items() if k in names}) for line in f if line.strip()]


def to_pause_events(events: list[Event], channel: str = "customer") -> list[PauseEvent]:
    """SHIFT/HOLD events of one channel, as input to `eval.eot_events` (cut-in rate vs response gap)."""
    return [
        PauseEvent(
            e.event_id,
            e.type,
            e.pause_s,
            {"session": e.session, "speaker": e.speaker, "device": e.device, "digit_span": e.digit_span},
        )
        for e in events
        if e.channel == channel and e.type in (SHIFT, HOLD)
    ]
