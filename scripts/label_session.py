"""Label two-track sessions: Silero VAD -> IPUs -> SHIFT/HOLD/BC/INTERRUPT/HOLD_REQ events (plan §4.2).

Each session folder holds one track per role, named <session>_<speaker>_<role>.<ext> as in
docs/recording_protocol.md §5. Devices come from data/raw/sessions.csv when it has the session.
Writes data/labels/<session>/ipus.jsonl, events.jsonl and summary.json, and prints the counts.

Without a transcript only duration rules apply (no BC lexicon, HOLD_REQ or digit spans). `--asr`
transcribes each channel's IPUs (pehleaap/labeling/asr.py; Whisper large-v3 by default, so run it on
the GPU box or Kaggle) and saves the words as data/labels/<session>/words.jsonl; later runs reuse
that file, with or without --asr, until --asr-refresh (needed after changing the VAD settings).

    python scripts/label_session.py data/raw/sessions/S017
    python scripts/label_session.py "data/raw/sessions/S0*"          # several (globs are expanded here)
    python scripts/label_session.py data/raw/sessions/S017 --asr     # transcribe, then label with text rules
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
from pehleaap.labeling.asr import AsrConfig, WhisperASR, read_words, transcribe, write_words  # noqa: E402
from pehleaap.labeling.config import load_config  # noqa: E402
from pehleaap.labeling.events import extract_events, overlap  # noqa: E402
from pehleaap.labeling.ipu import attach_words, ipus_from_probs  # noqa: E402
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


def label_session(session_dir: Path, out_root: Path, cfg, vad: SileroVAD, recognize=None, asr_cfg: AsrConfig | None = None,
                  refresh_asr: bool = False) -> dict:
    """Label one session. `recognize` (e.g. a WhisperASR) transcribes it when it has no words.jsonl yet (or `refresh_asr`)."""
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

    out = out_root / session
    out.mkdir(parents=True, exist_ok=True)
    words_path, asr_info, words = out / "words.jsonl", None, None
    if words_path.exists() and not refresh_asr:
        asr_info, words = read_words(words_path)
    elif recognize is not None:
        t = time.time()
        words = {role: transcribe(audio[role], cfg.vad_sr, ipus[role], recognize, asr_cfg) for role in ROLES}
        write_words(words_path, words, asr_cfg)
        asr_info = {**asdict(asr_cfg), "seconds": round(time.time() - t, 1), "retried_loops": getattr(recognize, "retried", None)}
    if words is not None:
        for role in ROLES:
            attach_words(ipus[role], words.get(role, []))
    events = [
        replace(e, session=session, speaker=speakers[e.channel], device=devices[e.channel])
        for e in extract_events(ipus, end_s, cfg)
    ]

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
        "asr": asr_info,  # None: duration rules only
        "n_words": {role: len(words.get(role, [])) for role in ROLES} if words is not None else None,
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
    asr = f"ASR {s['asr']['model']} ({sum(s['n_words'].values())} words)" if s["asr"] else "no transcript: duration rules only"
    print(f"\n{s['session']}: {s['duration_s'] / 60:.1f} min, VAD {s['vad_seconds']} s, {asr}")
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
    ap.add_argument("--asr", action="store_true", help="transcribe sessions that have no words.jsonl yet")
    ap.add_argument("--asr-refresh", action="store_true", help="transcribe again even if words.jsonl exists")
    ap.add_argument("--asr-model", default=AsrConfig.model)
    ap.add_argument("--asr-language", default=AsrConfig.language, help="Whisper language code, or 'auto' to detect")
    ap.add_argument("--asr-batch-size", type=int, default=AsrConfig.batch_size, help="lower it if the GPU runs out of memory")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # transcripts are Devanagari

    dirs = sorted({Path(p) for arg in args.sessions for p in (glob.glob(arg) or [arg]) if Path(p).is_dir()})
    if not dirs:
        raise SystemExit("no session folders found")
    cfg = load_config(args.config)
    vad = SileroVAD()
    asr_cfg = AsrConfig(model=args.asr_model, language=None if args.asr_language == "auto" else args.asr_language,
                        batch_size=args.asr_batch_size)
    recognize = WhisperASR(asr_cfg) if args.asr or args.asr_refresh else None
    for d in dirs:
        print_summary(label_session(d, Path(args.out), cfg, vad, recognize, asr_cfg, args.asr_refresh))


if __name__ == "__main__":
    main()
