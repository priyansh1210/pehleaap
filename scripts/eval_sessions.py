"""End-of-turn models on recorded, labelled conversations: cut-in rate against response gap (plan §6).

For every labelled pause of the evaluated channel (default: the customer) each model is asked, at
fixed checkpoints after the pause starts, whether the turn is over, given only the audio up to that
moment through the chosen phone-line condition (`pehleaap/eval/sessions.py`). Then for each model:

- at its own threshold: cut-in rate (it answered during a pause where the speaker carried on), median
  and 90th-percentile response gap on real turn ends, and missed turn ends (never answered in time);
- swept over thresholds: the cut-in rate at a matched median response gap (0.5 s and 0.8 s), the
  fair comparison, since any model can trade cut-ins for speed;
- and the plain silence-timeout bot as the floor every model must beat.

Uses verified labels (events_verified.jsonl, from verify_app.py --summary) where a session has them,
the automatic labels otherwise (marked "unverified" in the report). Heavy for long sessions and many
models/conditions: run it on the GPU box.

    python scripts/eval_sessions.py "data/labels/S0*" --models smart_turn_v3.2_int8 eot_v0 --conditions clean g711_mulaw mixed
    python scripts/eval_sessions.py runs/tts_labels/TTS01 --channels customer agent --out sessions_smoke
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import load_audio  # noqa: E402
from pehleaap.baselines.turn_models import SAMPLE_RATE, load_any  # noqa: E402
from pehleaap.eval.eot_events import cut_in_at_gap, operating_point, threshold_curve, timeout_curve  # noqa: E402
from pehleaap.eval.sessions import CHECKPOINTS_S, checkpoint_probs, condition_track  # noqa: E402
from pehleaap.labeling.schema import HOLD, SHIFT, read_events, to_pause_events  # noqa: E402
from pehleaap.paths import REPO_ROOT, RUNS_DIR  # noqa: E402
from pehleaap.telephony.conditions import CHAIN, describe, known  # noqa: E402
from pehleaap.telephony.noise import NoiseBank  # noqa: E402
from pehleaap.telephony.pipeline import load_config  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TARGET_GAPS_S = (0.5, 0.8)
TIMEOUTS_S = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)


def load_session(d: Path, channels: list[str]) -> tuple[list, dict, bool]:
    """(the evaluated channels' SHIFT/HOLD events, track paths, whether the labels were checked by a person)."""
    verified = (d / "events_verified.jsonl").exists()
    events = read_events(d / ("events_verified.jsonl" if verified else "events.jsonl"))
    events = [e for e in events if e.channel in channels and e.type in (SHIFT, HOLD) and e.pause_s is not None]
    return events, json.loads((d / "summary.json").read_text(encoding="utf-8"))["tracks"], verified


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="label folders (globs allowed), each with summary.json + events")
    ap.add_argument("--models", nargs="+", default=["smart_turn_v3.2_int8", "eot_v0"], help="registry names or weights/<name>")
    ap.add_argument("--conditions", nargs="+", default=["clean", "g711_mulaw", "amrnb_7.95"])
    ap.add_argument("--channels", nargs="+", default=["customer"], help="whose pauses to evaluate")
    ap.add_argument("--timeout-s", type=float, default=3.0, help="the bot's hard silence timeout (Pipecat's fallback is 3 s)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="sessions_eval")
    args = ap.parse_args()

    if unknown := [c for c in args.conditions if not known(c)]:
        raise SystemExit(f"unknown conditions {unknown}")
    dirs = sorted({Path(p) for pat in args.sessions for p in (glob.glob(pat) or [pat]) if (Path(p) / "summary.json").exists()})
    if not dirs:
        raise SystemExit("no label folders with summary.json found")
    bank = NoiseBank(sr=SAMPLE_RATE, part="test") if any(c in CHAIN for c in args.conditions) else None
    if bank is not None and not bank.files:
        raise SystemExit("chain conditions use MUSAN noise: run scripts/get_noise.py first")
    cfg = load_config(REPO_ROOT / "configs" / "augment.yaml")
    models = {name: load_any(name, threads=args.threads) for name in args.models}

    rows, all_events, unverified, t0 = [], [], [], time.time()
    for d in dirs:
        events, tracks, verified = load_session(d, args.channels)
        if not verified:
            unverified.append(d.name)
        all_events += events
        for channel in args.channels:
            ev = [e for e in events if e.channel == channel]
            if not ev:
                continue
            x, _ = load_audio(tracks[channel], sr=SAMPLE_RATE)
            for cond in args.conditions:
                y = condition_track(x, SAMPLE_RATE, cond, f"{d.name}_{channel}", bank, cfg)
                for name, (model, _) in models.items():
                    p = checkpoint_probs(model, y, SAMPLE_RATE, ev)
                    rows += [{"event_id": e.event_id, "session": e.session, "channel": channel, "kind": e.type, "pause_s": e.pause_s,
                              "condition": cond, "model": name, **{f"p@{c}": v for c, v in zip(CHECKPOINTS_S, p[k])}}
                             for k, e in enumerate(ev)]
        print(f"{d.name}: {len(events)} pauses ({'verified' if verified else 'UNVERIFIED'}), {time.time() - t0:.0f}s elapsed", flush=True)

    out = RUNS_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "probs.csv", index=False)
    pcols = [f"p@{c}" for c in CHECKPOINTS_S]
    pause_events = {pe.event_id: pe for ch in args.channels for pe in to_pause_events(all_events, ch)}

    n_hold = sum(e.type == HOLD for e in all_events)
    lines = [
        "# End-of-turn models on recorded conversations", "",
        f"{len(dirs)} sessions, {len(all_events)} pauses of {', '.join(args.channels)} ({len(all_events) - n_hold} turn ends, {n_hold} holds). "
        f"Models asked {', '.join(f'{c:g}' for c in CHECKPOINTS_S)} s after each pause starts, last 8 s of audio; hard timeout {args.timeout_s:g} s.",
        f"Labels: {'all verified' if not unverified else 'UNVERIFIED (automatic) for ' + ', '.join(unverified)}.",
        "Cut-in = answered during a pause where the speaker carried on. Gap = pause start to answer, on real turn ends. "
        "Missed = never answered before the speaker went on.", "",
    ]
    results = {"sessions": [d.name for d in dirs], "unverified": unverified, "checkpoints_s": CHECKPOINTS_S, "conditions": {}}
    timeout_rows = timeout_curve([pause_events[e.event_id] for e in all_events], TIMEOUTS_S)
    for cond in args.conditions:
        lines += [f"## {cond}: {describe(cond)}", "",
                  "| Model | Thr | Cut-in | Median gap | p90 gap | Missed | Cut-in @ 0.5 s gap | Cut-in @ 0.8 s gap |",
                  "|---|---|---|---|---|---|---|---|"]
        res = {}
        for name, (_, thr) in models.items():
            sub = df[(df.condition == cond) & (df.model == name)]
            evs = [pause_events[i] for i in sub.event_id]
            probs = sub[pcols].to_numpy(dtype=float)
            op = operating_point(evs, probs, CHECKPOINTS_S, thr, args.timeout_s)
            curve = threshold_curve(evs, probs, CHECKPOINTS_S, THRESHOLDS, args.timeout_s)
            matched = {g: cut_in_at_gap(curve, g) for g in TARGET_GAPS_S}
            res[name] = {"at_threshold": op, "matched": matched, "curve": curve}
            lines.append(f"| {name} | {thr:g} | {op['cut_in_rate']:.1%} | {op['median_gap_s']:.2f} s | {op['p90_gap_s']:.2f} s | "
                         f"{op['miss_rate']:.1%} | {matched[0.5]:.1%} | {matched[0.8]:.1%} |")
        timeout_matched = {g: cut_in_at_gap(timeout_rows, g) for g in TARGET_GAPS_S}
        lines.append(f"| silence timeout | - | - | - | - | - | {timeout_matched[0.5]:.1%} | {timeout_matched[0.8]:.1%} |")
        lines.append("")
        results["conditions"][cond] = {"models": res, "silence_timeout": {"matched": timeout_matched, "curve": timeout_rows}}
    lines += ["Silence timeout: no model, the bot answers after a fixed silence (swept "
              f"{', '.join(f'{t:g}' for t in TIMEOUTS_S)} s); the same for every condition, since VAD runs on the clean recording."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
