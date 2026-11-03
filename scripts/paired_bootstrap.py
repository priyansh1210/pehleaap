"""Paired model comparison on an eval_sessions.py run, with a speaker-clustered bootstrap (docs/prereg.md).

For each condition and matched response gap: cut-in rate of model A minus model B (negative = A cuts in less),
with a 95% interval from resampling the evaluated speakers (every pause of a drawn speaker comes along, for
both models, so the comparison stays paired). Speakers come from the label folders' events.

    python scripts/paired_bootstrap.py runs/turnbench_dev_eval "runs/turnbench_labels/TB*" --a eot_v1 --b smart_turn_v3.2_int8
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.eval.bootstrap import cluster_bootstrap  # noqa: E402
from pehleaap.eval.eot_events import PauseEvent, cut_in_at_gap, threshold_curve  # noqa: E402
from pehleaap.eval.sessions import CHECKPOINTS_S  # noqa: E402
from pehleaap.labeling.schema import read_events  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)  # the same sweep as eval_sessions.py


def matched_cut_in(kinds, durs, probs, gap_s: float, timeout_s: float) -> float:
    events = [PauseEvent(str(i), k, d) for i, (k, d) in enumerate(zip(kinds, durs))]
    return cut_in_at_gap(threshold_curve(events, probs, CHECKPOINTS_S, THRESHOLDS, timeout_s), gap_s)


def speakers(label_dirs: list[Path]) -> dict[str, str]:
    """event_id -> speaker, from each folder's verified events where it has them (as eval_sessions.py reads them)."""
    out = {}
    for d in label_dirs:
        f = d / "events_verified.jsonl" if (d / "events_verified.jsonl").exists() else d / "events.jsonl"
        out.update({e.event_id: e.speaker or f"{e.session}_{e.channel}" for e in read_events(f)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="eval_sessions.py output folder (probs.csv)")
    ap.add_argument("labels", nargs="+", help="the label folders it scored (globs allowed)")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--timeout-s", type=float, default=3.0, help="as in eval_sessions.py")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(Path(args.run) / "probs.csv")
    spk = speakers(sorted({Path(p) for pat in args.labels for p in (glob.glob(pat) or [pat]) if Path(p).is_dir()}))
    pcols = [f"p@{c}" for c in CHECKPOINTS_S]
    results = []
    for cond, sub in df.groupby("condition"):
        a = sub[sub.model == args.a].set_index("event_id")
        b = sub[sub.model == args.b].set_index("event_id").loc[a.index]  # same pauses, same order: a paired comparison
        if missing := [i for i in a.index if i not in spk]:
            raise SystemExit(f"{len(missing)} events have no speaker in the label folders, e.g. {missing[0]}")
        kinds, durs, clusters = a["kind"].to_numpy(), a["pause_s"].to_numpy(float), np.array([spk[i] for i in a.index])
        pa, pb = a[pcols].to_numpy(float), b[pcols].to_numpy(float)
        for gap in args.gaps:
            def stat(idx, gap=gap):
                return (matched_cut_in(kinds[idx], durs[idx], pa[idx], gap, args.timeout_s)
                        - matched_cut_in(kinds[idx], durs[idx], pb[idx], gap, args.timeout_s))
            r = cluster_bootstrap(stat, clusters, args.n_boot, args.seed)
            results.append({"condition": cond, "gap_s": gap, "a": args.a, "b": args.b,
                            "cut_in_a": matched_cut_in(kinds, durs, pa, gap, args.timeout_s),
                            "cut_in_b": matched_cut_in(kinds, durs, pb, gap, args.timeout_s), **r})
            x = results[-1]
            print(f"{cond} @ {gap:g} s gap: {args.a} {x['cut_in_a']:.1%} vs {args.b} {x['cut_in_b']:.1%}, difference "
                  f"{x['estimate']:+.1%} [95% CI {x['ci_low']:+.1%}, {x['ci_high']:+.1%}], "
                  f"{x['n_clusters']} speakers, {x['n_events']} pauses", flush=True)
    def short(name: str) -> str:  # a registry name as it is ("smart_turn_v3.2_int8"), a file path by its stem
        return Path(name).stem if name.endswith(".onnx") else name

    out = Path(args.run) / f"bootstrap_{short(args.a)}_vs_{short(args.b)}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
