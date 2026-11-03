"""Relabel our automatic TurnBench pauses with TurnBench's human annotators (majority of three).

label_session.py finds each pause and calls it SHIFT or HOLD from VAD and duration rules. Here every
annotator gives each pause a verdict from their turn segments, and a pause is kept with the verdict
at least two annotators agree on:
- HOLD: the pause lies inside one of the speaker's annotated turns, or the speaker starts a new turn
  before the other person does;
- SHIFT: the speaker's turn ends at the pause and the other person's next turn comes first;
- no verdict otherwise (e.g. the speech before the pause was a backchannel or laughter, not a turn).
Backchannels, laughter, fillers and channel bleed never count as taking the turn.

Writes events_verified.jsonl next to each events.jsonl (eval_sessions.py prefers it) and prints how
often the automatic labels agree with the human majority.

    python scripts/turnbench_verify.py "runs/turnbench_labels/TB*"
    python scripts/eval_sessions.py "runs/turnbench_labels/TB*" --channels agent customer --models ...
"""

import argparse
import glob
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.labeling.schema import HOLD, SHIFT, read_events, write_events  # noqa: E402

# Labels for speech that holds or takes the floor; every other TurnBench label (backchannels, laughter, fillers,
# non-floor-taking interruptions, overlap, noise, channel bleed) does not.
FLOOR = {"Normal Turn", "Strong Floor Hold", "Floor-taking Cooperative Interruption",
         "Floor-taking Competitive Interruption", "Bounded Response"}
# shortcut: one fixed tolerance between our VAD pause start and the annotators' segment ends; tune it on the
# agreement table if many pauses get no verdict.
TOL_S = 0.4


def verdict(t: float, pause_s: float, own: list[dict], other: list[dict], tol: float = TOL_S) -> str | None:
    """One annotator's SHIFT / HOLD / None for a pause from t to t + pause_s on the speaker whose segments are `own`."""
    own = sorted((e for e in own if e["label"] in FLOOR), key=lambda e: e["start_s"])
    at = [e for e in own if e["start_s"] <= t + tol and e["end_s"] >= t - tol]
    if not at:
        return None
    seg = max(at, key=lambda e: e["end_s"])
    if seg["end_s"] >= t + pause_s - tol:
        return HOLD  # the annotator's turn runs through the pause
    next_own = min((e["start_s"] for e in own if e["start_s"] > seg["end_s"]), default=float("inf"))
    next_other = min((e["start_s"] for e in other if e["label"] in FLOOR and e["end_s"] > seg["end_s"]), default=float("inf"))
    if next_other == next_own == float("inf"):
        return None
    return SHIFT if next_other < next_own else HOLD


def majority(votes: list[str | None]) -> str | None:
    top = Counter(v for v in votes if v is not None).most_common(1)
    return top[0][0] if top and top[0][1] >= 2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+", help="label_session.py output folders (globs allowed)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dirs = sorted({Path(p) for pat in args.labels for p in (glob.glob(pat) or [pat]) if (Path(p) / "events.jsonl").exists()})
    if not dirs:
        raise SystemExit("no label folders with events.jsonl found")
    table = Counter()
    for d in dirs:
        tracks = json.loads((d / "summary.json").read_text(encoding="utf-8"))["tracks"]
        tb = json.loads((Path(tracks["agent"]).parent / "turnbench.json").read_text(encoding="utf-8"))
        ann = {role: tb["annotations"][spk] for spk, role in tb["roles"].items()}
        kept = []
        for e in read_events(d / "events.jsonl"):
            if e.type not in (SHIFT, HOLD) or e.pause_s is None:
                continue
            other = "customer" if e.channel == "agent" else "agent"
            votes = [verdict(e.t_event, e.pause_s, ann[e.channel][a], ann[other][a]) for a in "abc" if ann[e.channel][a]]
            human = majority(votes)
            table[(e.type, human)] += 1
            if human is not None:
                kept.append(replace(e, type=human, verified=True))
        write_events(d / "events_verified.jsonl", kept)

    total = sum(table.values())
    print(f"{len(dirs)} sessions, {total} automatic SHIFT/HOLD pauses")
    print("automatic -> human majority: " + ", ".join(f"{a} -> {h or 'no verdict'}: {n}" for (a, h), n in sorted(table.items(), key=str)))
    agree = table[(SHIFT, SHIFT)] + table[(HOLD, HOLD)]
    judged = sum(n for (_, h), n in table.items() if h is not None)
    print(f"kept {judged} ({judged / total:.0%}); automatic label agrees with the humans on {agree / judged:.1%} of them")


if __name__ == "__main__":
    main()
