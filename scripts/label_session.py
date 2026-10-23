"""Label two-track sessions: Silero VAD -> IPUs -> SHIFT/HOLD/BC/INTERRUPT/HOLD_REQ events (plan §4.2).

Each session folder holds one track per role, named <session>_<speaker>_<role>.<ext> as in
docs/recording_protocol.md §5. Devices come from data/raw/sessions.csv when it has the session.
Writes data/labels/<session>/ipus.jsonl, events.jsonl and summary.json, and prints the counts.
Until ASR has run, only duration rules apply (no BC lexicon, HOLD_REQ or digit spans).

    python scripts/label_session.py data/raw/sessions/S017
    python scripts/label_session.py "data/raw/sessions/S0*"          # several (globs are expanded here)
"""

import argparse
import csv
import glob
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import load_audio  # noqa: E402
from pehleaap.audio.vad import SileroVAD, hop_s  # noqa: E402
from pehleaap.eval.eot_events import timeout_curve  # noqa: E402
from pehleaap.labeling.config import load_config  # noqa: E402
from pehleaap.labeling.events import extract_events, overlap  # noqa: E402
from pehleaap.labeling.ipu import ipus_from_probs  # noqa: E402
from pehleaap.labeling.schema import SHIFT, TYPES, to_pause_events, write_events  # noqa: E402
from pehleaap.paths import DATA_DIR, REPO_ROOT  # noqa: E402

ROLES = ("agent", "customer")
AUDIO_EXTS = {".flac", ".wav", ".m4a", ".ogg", ".opus", ".mp3", ".aac", ".webm"}
TIMEOUTS_S = (0.5, 0.8, 1.2)  # silence-timeout reference points for the summary


def find_tracks(session_dir: Path) -> dict[str, Path]:
    tracks = {}
    for role in ROLES:
        found = [p for p in session_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS and p.stem.endswith(f"_{role}")]
        if len(found) != 1:
            raise SystemExit(f"{session_dir}: expected one *_{role} audio track, found {[p.name for p in found]}")
        tracks[role] = found[0]
    return tracks


def session_devices(session: str, sessions_csv: Path) -> dict[str, str | None]:
    if sessions_csv.exists():
        with open(sessions_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("session_id") == session:
                    return {role: row.get(f"device_{role}") or None for role in ROLES}
    return {role: None for role in ROLES}


def label_session(session_dir: Path, out_root: Path, cfg, vad: SileroVAD) -> dict:
    session = session_dir.name
    tracks = find_tracks(session_dir)
    speakers = {role: (p.stem.split("_")[1] if p.stem.count("_") >= 2 else None) for role, p in tracks.items()}
    devices = session_devices(session, DATA_DIR / "raw" / "sessions.csv")

    audio = {role: load_audio(p, sr=cfg.vad_sr)[0] for role, p in tracks.items()}
    n = max(len(x) for x in audio.values())
    batch = np.stack([np.pad(audio[r], (0, n - len(audio[r]))) for r in ROLES])
    t = time.time()
    probs = vad.probs(batch, cfg.vad_sr)
    vad_s = time.time() - t

    ipus = {role: ipus_from_probs(probs[i], hop_s(cfg.vad_sr), cfg) for i, role in enumerate(ROLES)}
    end_s = n / cfg.vad_sr
    events = [
        replace(e, session=session, speaker=speakers[e.channel], device=devices[e.channel])
        for e in extract_events(ipus, end_s, cfg)
    ]

    out = out_root / session
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ipus.jsonl", "w", encoding="utf-8") as f:
        for role in ROLES:
            for u in ipus[role]:
                f.write(json.dumps({"channel": role, **asdict(u)}, ensure_ascii=False) + "\n")
    write_events(out / "events.jsonl", events)

    customer_shifts = [e.gap_s for e in events if e.channel == "customer" and e.type == SHIFT]
    summary = {
        "session": session,
        "tracks": {role: str(p) for role, p in tracks.items()},
        "duration_s": round(end_s, 2),
        "vad_seconds": round(vad_s, 1),
        "speech_s": {role: round(sum(u.dur for u in ipus[role]), 1) for role in ROLES},
        "overlap_s": round(sum(overlap(u, ipus["customer"]) for u in ipus["agent"]), 1),
        "n_ipus": {role: len(ipus[role]) for role in ROLES},
        "counts": {role: {ty: sum(e.channel == role and e.type == ty for e in events) for ty in TYPES} for role in ROLES},
        # How fast the human agent took the turn after the customer finished.
        "agent_response_gap_median_s": round(float(np.median(customer_shifts)), 3) if customer_shifts else None,
        # Silence-timeout reference on the customer's pauses, using these (oracle) VAD boundaries.
        "silence_timeout": [
            {k: p[k] for k in ("timeout_s", "cut_in_rate", "median_gap_s")}
            for p in timeout_curve(to_pause_events(events, "customer"), TIMEOUTS_S)
        ],
        "config": cfg.to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(s: dict) -> None:
    print(f"\n{s['session']}: {s['duration_s'] / 60:.1f} min, VAD {s['vad_seconds']} s")
    for role in ROLES:
        counts = "  ".join(f"{ty} {c}" for ty, c in s["counts"][role].items())
        print(f"  {role:<8} speech {s['speech_s'][role]:>6.1f} s  IPUs {s['n_ipus'][role]:>4}  {counts}")
    print(f"  overlap {s['overlap_s']} s; agent response gap (median) {s['agent_response_gap_median_s']} s")
    for p in s["silence_timeout"]:
        print(f"  silence timeout {p['timeout_s']:.1f} s: customer cut-in rate {p['cut_in_rate']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session folders (globs allowed)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "labeling.yaml"))
    ap.add_argument("--out", default=str(DATA_DIR / "labels"))
    args = ap.parse_args()

    dirs = sorted({Path(p) for arg in args.sessions for p in (glob.glob(arg) or [arg]) if Path(p).is_dir()})
    if not dirs:
        raise SystemExit("no session folders found")
    cfg = load_config(args.config)
    vad = SileroVAD()
    for d in dirs:
        print_summary(label_session(d, Path(args.out), cfg, vad))


if __name__ == "__main__":
    main()
