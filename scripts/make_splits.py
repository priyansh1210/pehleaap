"""Propose, freeze and check the speaker-disjoint train/dev/test splits (plan §4.5; pehleaap/data/splits.py).

    python scripts/make_splits.py              # propose from data/raw/sessions.csv: prints, writes nothing
    python scripts/make_splits.py --freeze     # write benchmark/splits/ + lock.json (end of week 3, before any test-set run)
    python scripts/make_splits.py              # once frozen: check the lock, list sessions not in any split
    python scripts/make_splits.py --add-new    # once frozen: add those sessions to train, if their speakers are new

Minutes per session come from data/labels/<session>/summary.json when the session is labelled (25 otherwise);
genders from data/raw/speakers.csv (used only for the balance check, never printed per person). Commit
benchmark/splits/ after freezing: it holds session ids only.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.data.splits import SPLITS, SPLITS_DIR, Session, add_to_train, assign, freeze, load, speaker_groups  # noqa: E402
from pehleaap.paths import DATA_DIR  # noqa: E402


def read_sessions(data_root: Path) -> list[Session]:
    raw = data_root / "raw"
    with open(raw / "sessions.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("session_id") or "").strip()]
    genders = {}
    if (raw / "speakers.csv").exists():
        with open(raw / "speakers.csv", encoding="utf-8", newline="") as f:
            genders = {r["speaker_id"].strip(): (r.get("gender") or "").strip() for r in csv.DictReader(f) if r.get("speaker_id")}
    out = []
    for r in rows:
        sid = r["session_id"].strip()
        summary = data_root / "labels" / sid / "summary.json"
        minutes = json.loads(summary.read_text(encoding="utf-8"))["duration_s"] / 60 if summary.exists() else 25.0
        spk = (r["spk_agent"].strip(), r["spk_customer"].strip())
        out.append(Session(sid, spk, (r["device_agent"].strip(), r["device_customer"].strip()), minutes, (genders.get(spk[0], ""), genders.get(spk[1], ""))))
    return out


def report(sessions: list[Session], assignment: dict[str, str]) -> None:
    total = sum(s.minutes for s in sessions) or 1.0
    groups = speaker_groups(sessions)
    print(f"{len(sessions)} sessions, {total:.0f} min, {len({p for s in sessions for p in s.speakers})} speakers in {len(groups)} speaker groups")
    for split in SPLITS:
        ss = [s for s in sessions if assignment.get(s.session_id) == split]
        mins = sum(s.minutes for s in ss)
        spk = {p for s in ss for p in s.speakers}
        g = Counter(x for s in ss for x in s.genders if x)
        d = Counter(x for s in ss for x in s.devices)
        print(f"  {split:<5} {len(ss):3d} sessions {mins:6.0f} min ({mins / total:4.0%})  {len(spk):3d} speakers  "
              f"gender {dict(g)}  devices {dict(d)}")
    seen: dict[str, str] = {}
    for s in sessions:
        for p in s.speakers:
            other = seen.setdefault(p, assignment.get(s.session_id, ""))
            if other and assignment.get(s.session_id) and other != assignment[s.session_id]:
                raise SystemExit(f"BUG: speaker {p} is in both {other} and {assignment[s.session_id]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--add-new", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_DIR))
    ap.add_argument("--splits-dir", default=str(SPLITS_DIR))
    args = ap.parse_args()

    sessions, splits_dir = read_sessions(Path(args.data_root)), Path(args.splits_dir)
    frozen = load(splits_dir)
    if frozen is None:
        assignment = assign(sessions)
        report(sessions, assignment)
        if args.freeze:
            freeze(assignment, splits_dir, time.strftime("%Y-%m-%d"))
            print(f"frozen: {splits_dir} (commit it; from now on new sessions can only join train)")
        else:
            print("proposal only: run with --freeze to fix it (end of week 3, before the first test-set run)")
        return
    if args.freeze:
        raise SystemExit(f"already frozen ({splits_dir / 'lock.json'}); the test set must not change")
    by_id = {s.session_id: s for s in sessions}
    new = [s for s in sessions if s.session_id not in frozen]
    print(f"splits frozen and unchanged ({splits_dir}); {len(new)} sessions recorded since")
    if args.add_new and new:
        added, held_back = add_to_train(new, frozen, by_id, splits_dir)
        print(f"added to train: {added or 'none'}")
        if held_back:
            print(f"held back (a speaker is in dev or test, so they can't be trained on): {held_back}")
        frozen = load(splits_dir)
    report(sessions, frozen)


if __name__ == "__main__":
    main()
