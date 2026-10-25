"""Turn our labelled recordings into end-of-turn training clips (plan §4.4, week 3: "our SHIFT/HOLD clips").

Every SHIFT (they had finished) and HOLD (they paused and carried on) of each channel becomes one clip:
that speaker's audio up to the pause start plus `--tail-s` (0.2 s, what a live bot has heard when it
decides), labelled finished or not. Clips go into the training clip store (train_eot.py) by the
session's frozen split (scripts/make_splits.py): train sessions to training, dev sessions to the dev
set used to pick the best checkpoint. **Test sessions are refused**, and so is everything before the
splits are frozen, so no test speaker can leak into training.

Verified labels (events_verified.jsonl, from verify_app.py --summary) are used where a session has them,
the automatic ones otherwise (fine for training; only the test set needs verified labels). A HOLD whose
pause is shorter than the tail is skipped: the clip would already contain the resumed speech.

    python scripts/sessions_to_store.py "data/labels/S0*"
    python scripts/sessions_to_store.py "data/labels/S0*" --store data/external/eot-train --channels customer
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import load_audio  # noqa: E402
from pehleaap.data.splits import SPLITS_DIR, load  # noqa: E402
from pehleaap.labeling.schema import HOLD, SHIFT, read_events  # noqa: E402
from pehleaap.paths import EXTERNAL_DIR  # noqa: E402
from pehleaap.train.eot_data import EotStore  # noqa: E402

SR = 16000


def session_clips(label_dir: Path, split: str, channels: tuple[str, ...], tail_s: float) -> tuple[list[dict], list, bool]:
    """(metas, clips, verified) for one labelled session."""
    verified = (label_dir / "events_verified.jsonl").exists()
    events = read_events(label_dir / ("events_verified.jsonl" if verified else "events.jsonl"))
    tracks = json.loads((label_dir / "summary.json").read_text(encoding="utf-8"))["tracks"]
    metas, clips = [], []
    for channel in channels:
        ev = [e for e in events if e.channel == channel and e.type in (SHIFT, HOLD) and e.pause_s is not None]
        ev = [e for e in ev if e.type == SHIFT or e.pause_s >= tail_s]
        if not ev:
            continue
        x, _ = load_audio(tracks[channel], sr=SR)
        for e in ev:
            metas.append({"id": f"{label_dir.name}_{e.event_id}", "language": "hin", "endpoint_bool": e.type == SHIFT,
                          "synthetic": False, "dataset": f"pehleaap_{label_dir.name}", "split": split})
            clips.append(x[: int(round((e.t_event + tail_s) * SR))])
    return metas, clips, verified


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="label folders (data/labels/<session>; globs allowed)")
    ap.add_argument("--store", default=str(EXTERNAL_DIR / "eot-train"))
    ap.add_argument("--channels", nargs="+", default=["agent", "customer"], help="whose pauses become clips (both are real people)")
    ap.add_argument("--tail-s", type=float, default=0.2)
    ap.add_argument("--splits-dir", default=str(SPLITS_DIR))
    args = ap.parse_args()

    splits = load(Path(args.splits_dir))
    if splits is None:
        raise SystemExit("the splits aren't frozen yet (scripts/make_splits.py --freeze): without them a test speaker could end up in training")
    dirs = sorted({Path(p) for pat in args.sessions for p in (glob.glob(pat) or [pat]) if (Path(p) / "summary.json").exists()})
    if not dirs:
        raise SystemExit("no label folders with summary.json found")
    store = EotStore(args.store)
    for d in dirs:
        split = splits.get(d.name)
        if split not in ("train", "dev"):
            print(f"{d.name}: {'TEST session: never used for training' if split == 'test' else 'not in the frozen splits (make_splits.py --add-new)'}, skipped")
            continue
        shard = f"pehleaap_{d.name}"
        if shard in store.done_shards():
            print(f"{d.name}: already in the store")
            continue
        metas, clips, verified = session_clips(d, split, tuple(args.channels), args.tail_s)
        if not metas:
            print(f"{d.name}: no SHIFT/HOLD events")
            continue
        store.append(metas, clips, shard)
        n_fin = sum(m["endpoint_bool"] for m in metas)
        print(f"{d.name} ({split}, {'verified' if verified else 'automatic'} labels): {n_fin} finished + {len(metas) - n_fin} holds")
    print(f"store {args.store}: {len(store.index)} clips")


if __name__ == "__main__":
    main()
