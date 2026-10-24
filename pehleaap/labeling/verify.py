"""Human verification of the automatic labels (plan §4.2 step 4).

Which events a person checks: every BC, INTERRUPT, HOLD_REQ and NOISE_ECHO (the rules for them are
the least certain), every event of a session listed as PSTN (the real-phone test set), and a fixed 10% of
SHIFT/HOLD. A further 10%, picked by hash, is shown to a second annotator too, for Cohen's kappa.
Both picks hash the event id, so they don't change as sessions are added.

Each annotator's verdicts go to data/labels/<session>/verify/<annotator>.jsonl, one line per click;
the last line per event wins, so changing your mind is just another click. `apply_verdicts` turns
them into verified events: kept as labelled, retyped, or dropped ("none"). The test set uses only
verified events.
"""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from pehleaap.labeling.schema import BC, HOLD, HOLD_REQ, INTERRUPT, NOISE_ECHO, SHIFT, TYPES, Event

NONE = "none"  # not a turn-taking event at all
LABELS = TYPES + (NONE,)  # what an annotator can say an event really is (noise_echo also covers TV, traffic, coughs)
CORRECT, WRONG, UNSURE = "correct", "wrong", "unsure"
ALWAYS = (BC, INTERRUPT, HOLD_REQ, NOISE_ECHO)


def _hash01(text: str) -> float:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 16**8


def needs_review(e: Event, pstn_sessions: set[str] = frozenset(), shift_hold_fraction: float = 0.1) -> bool:
    if e.type in ALWAYS or e.session in pstn_sessions:
        return True
    return e.type in (SHIFT, HOLD) and _hash01(e.event_id) < shift_hold_fraction


def double_annotated(e: Event, fraction: float = 0.1) -> bool:
    """The events a second annotator labels too (a hash independent of `needs_review`'s)."""
    return _hash01("kappa:" + e.event_id) < fraction


def verdict_path(labels_dir: Path, session: str, annotator: str) -> Path:
    return Path(labels_dir) / session / "verify" / f"{annotator}.jsonl"


def save_verdict(path: Path, event_id: str, verdict: str, label: str, note: str = "") -> dict:
    """Append one decision. `label` is the event's type as the annotator hears it (the original type if `verdict` is correct)."""
    if verdict not in (CORRECT, WRONG, UNSURE) or label not in LABELS:
        raise ValueError(f"bad verdict {verdict!r} or label {label!r}")
    row = {"event_id": event_id, "verdict": verdict, "label": label, "note": note, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_verdicts(path: Path) -> dict[str, dict]:
    """event_id -> that annotator's latest decision."""
    if not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["event_id"]] = row
    return out


def apply_verdicts(events: list[Event], verdicts: dict[str, dict]) -> list[Event]:
    """Verified events: correct -> kept, wrong -> retyped (or dropped if "none"), unsure or unchecked -> left out."""
    out = []
    for e in events:
        v = verdicts.get(e.event_id)
        if v is None or v["verdict"] == UNSURE or v["label"] == NONE:
            continue
        out.append(replace(e, type=v["label"], verified=True))
    return out


def cohens_kappa(a: dict[str, dict], b: dict[str, dict]) -> tuple[float, int]:
    """Agreement on the label of events both annotators decided (unsure left out); (kappa, n)."""
    both = [k for k in a.keys() & b.keys() if a[k]["verdict"] != UNSURE and b[k]["verdict"] != UNSURE]
    if not both:
        return float("nan"), 0
    la, lb = [a[k]["label"] for k in both], [b[k]["label"] for k in both]
    labels = sorted(set(la) | set(lb))
    n = len(both)
    observed = sum(x == y for x, y in zip(la, lb)) / n
    expected = sum((la.count(c) / n) * (lb.count(c) / n) for c in labels)
    return (1.0 if observed == 1 else (observed - expected) / (1 - expected)), n
