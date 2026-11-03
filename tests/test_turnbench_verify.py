import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from turnbench_verify import majority, verdict  # noqa: E402


def seg(start, end, label="Normal Turn"):
    return {"start_s": start, "end_s": end, "label": label, "text": ""}


def test_verdict_hold_shift_and_no_verdict():
    own = [seg(0.0, 10.0), seg(12.0, 15.0)]
    # A pause inside the annotated turn is a hold, whatever the listener does.
    assert verdict(4.0, 0.6, own, [seg(4.1, 4.5, "Continuer Backchannel")]) == "hold"
    # The turn ends at the pause and the other person's turn comes first: a shift.
    assert verdict(10.1, 1.9, own, [seg(10.5, 11.8)]) == "shift"
    # Only a backchannel before the speaker carries on: a hold.
    assert verdict(10.1, 1.9, own, [seg(10.5, 11.0, "Reaction Backchannel")]) == "hold"
    # The speech before the pause was laughter, not a turn: no verdict.
    assert verdict(20.0, 1.0, own + [seg(19.0, 20.0, "Laughter")], []) is None


def test_majority_needs_two_votes():
    assert majority(["shift", "shift", "hold"]) == "shift"
    assert majority(["shift", "hold", None]) is None
    assert majority(["hold", "hold"]) == "hold"
