"""Process recorded sessions the day they are recorded: checks, labels and checksums in one go (plan §4.1-4.2).

    python scripts/process_session.py data/raw/sessions/S017
    python scripts/process_session.py "data/raw/sessions/S0*"
    python scripts/process_session.py data/raw/sessions/S017 --asr     # GPU box: with transcripts (text rules)

For each session folder (docs/recording_protocol.md §5):
1. Paperwork: a row in data/raw/sessions.csv, and both speakers in data/raw/speakers.csv with a consent
   date. Without consent it stops: that audio isn't processed at all.
2. Audio: exactly one agent and one customer track; sample rate, length, speech level, clipping and
   background noise, with warnings when something will hurt the labels.
3. Clock: the sync beeps at the start and end give the drift between the tracks (at most 10 ms, or
   turn timing can't be trusted). PSTN sessions are one dual-channel recording and skip this.
4. Labels: label_session.py (VAD, speakerphone echo, turn-taking events; transcripts with --asr).
5. Checksums: data/manifests/sessions/<session>.jsonl. Commit it; the audio never goes in git.
It writes data/labels/<session>/process.json and prints the next steps (checking labels by ear,
the GPU box). Exit code 1 if any session failed a check.
"""

import argparse
import csv
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.align import measure_drift  # noqa: E402
from pehleaap.audio.io import load_audio, resample  # noqa: E402
from pehleaap.manifest import sha256_file  # noqa: E402
from pehleaap.paths import DATA_DIR  # noqa: E402

SR = 16000
ROLES = ("agent", "customer")
DEVICES = {"earphones", "handset", "speakerphone", "pstn"}
MAX_DRIFT_MS = 10.0
_label = None


def label_module():
    """scripts/label_session.py, loaded once (it isn't a package module)."""
    global _label
    if _label is None:
        spec = importlib.util.spec_from_file_location("label_session", Path(__file__).with_name("label_session.py"))
        _label = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_label)
    return _label


def read_csv(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row[key].strip(): row for row in csv.DictReader(f) if row.get(key)}


def check_paperwork(session: str, tracks: dict[str, Path], raw: Path) -> tuple[list[str], dict]:
    """(problems, the session's metadata). Reads speakers.csv only for consent; nothing personal is printed."""
    problems, meta = [], {}
    row = read_csv(raw / "sessions.csv", "session_id").get(session)
    if row is None:
        return [f"no row for {session} in {raw / 'sessions.csv'}"], meta
    speakers = read_csv(raw / "speakers.csv", "speaker_id")
    for role in ROLES:
        spk, device = (row.get(f"spk_{role}") or "").strip(), (row.get(f"device_{role}") or "").strip()
        meta[role] = {"speaker": spk, "device": device}
        if device not in DEVICES:
            problems.append(f"device_{role} is {device!r}; use one of {sorted(DEVICES)}")
        if spk and spk not in tracks[role].stem:
            problems.append(f"spk_{role} {spk} doesn't match the track name {tracks[role].name}")
        if not (speakers.get(spk, {}).get("consent_date") or "").strip():
            problems.append(f"no consent date for the {role} ({spk or 'speaker id missing'}) in speakers.csv: not processing this session")
    return problems, meta


def audio_health(x: np.ndarray, sr_file: int) -> tuple[dict, list[str]]:
    """Level, clipping and noise for one track (16 kHz mono float), with warnings."""
    hop = SR // 100
    n = len(x) // hop
    db = 10 * np.log10((x[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    speech = float(np.percentile(db, 95)) if n else -120.0
    floor = float(np.percentile(db, 10)) if n else -120.0
    stats = {"sample_rate": sr_file, "seconds": round(len(x) / SR, 1), "speech_level_dbfs": round(speech, 1),
             "noise_floor_dbfs": round(floor, 1), "clipped_pct": round(100 * float(np.mean(np.abs(x) >= 0.999)), 3)}
    warn = []
    if sr_file < SR:
        warn.append(f"recorded at {sr_file} Hz: 16 kHz or more is needed (the phone line is simulated from it)")
    if stats["clipped_pct"] > 0.1:
        warn.append(f"{stats['clipped_pct']}% of samples clipped: the mic gain was too high")
    if speech < -40:
        warn.append(f"speech is quiet ({speech:.0f} dBFS): VAD may miss soft syllables")
    if speech - floor < 20:
        warn.append(f"only {speech - floor:.0f} dB between speech and background: noisy room or noise suppression")
    return stats, warn


def write_manifest(files: list[Path], data_root: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(files):
            f.write(json.dumps({"path": p.relative_to(data_root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}) + "\n")


def process(session_dir: Path, data_root: Path, cfg, vad, recognize, asr_cfg, force: bool) -> dict:
    session, t0 = session_dir.name, time.time()
    report = {"session": session, "ok": False, "problems": [], "warnings": []}
    ls = label_module()
    try:
        tracks = ls.find_tracks(session_dir)
    except SystemExit as e:
        report["problems"].append(str(e))
        return report
    problems, meta = check_paperwork(session, tracks, data_root / "raw")
    report["problems"] += problems
    report["devices"] = {role: meta.get(role, {}).get("device") for role in ROLES}
    if any("consent" in p for p in problems):
        return report  # nothing else is touched without consent

    audio = {}
    for role, path in tracks.items():
        x, sr_file = load_audio(path)
        audio[role] = resample(x, sr_file, SR)
        stats, warn = audio_health(audio[role], sr_file)
        report[role] = stats
        report["warnings"] += [f"{role}: {w}" for w in warn]
    if abs(report["agent"]["seconds"] - report["customer"]["seconds"]) > 2.0:
        report["warnings"].append("the two tracks differ in length by more than 2 s: check they are from the same call")

    if "pstn" in report["devices"].values():
        report["clock"] = {"skipped": "PSTN: one dual-channel recording, one clock"}
    else:
        try:
            r = measure_drift(audio["agent"], audio["customer"], SR)
            report["clock"] = {"offset_start_ms": round(r.offset_start_s * 1000, 1), "offset_end_ms": round(r.offset_end_s * 1000, 1),
                               "drift_ms": round(r.drift_ms, 1), "drift_ppm": round(r.drift_ppm, 1)}
            if abs(r.drift_ms) > MAX_DRIFT_MS:
                report["problems"].append(f"clock drift {r.drift_ms:+.1f} ms between the sync beeps (max {MAX_DRIFT_MS:g}): turn timing can't be trusted")
        except ValueError as e:
            report["problems"].append(f"clock check failed: {e}")
    if report["problems"] and not force:
        return report

    labels = data_root / "labels"
    report["labels"] = ls.label_session(session_dir, labels, cfg, vad, recognize, asr_cfg)
    write_manifest(list(tracks.values()) + sorted(session_dir.glob(f"{session}_meta.json")), data_root,
                   data_root / "manifests" / "sessions" / f"{session}.jsonl")
    report["ok"] = not report["problems"]
    report["seconds"] = round(time.time() - t0, 1)
    (labels / session / "process.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return report


def print_report(r: dict) -> None:
    status = "OK" if r["ok"] else ("LABELLED DESPITE PROBLEMS (--force)" if "labels" in r else "STOPPED")
    print(f"\n=== {r['session']}: {status}")
    for p in r["problems"]:
        print(f"  problem: {p}")
    for w in r["warnings"]:
        print(f"  warning: {w}")
    for role in ROLES:
        if role in r:
            s = r[role]
            print(f"  {role:<8} {s['seconds']:.0f} s at {s['sample_rate']} Hz, speech {s['speech_level_dbfs']} dBFS, "
                  f"background {s['noise_floor_dbfs']} dBFS, clipped {s['clipped_pct']}%, device {r['devices'][role]}")
    if "clock" in r:
        c = r["clock"]
        print(f"  clock    {c['skipped']}" if "skipped" in c else
              f"  clock    drift {c['drift_ms']:+.1f} ms ({c['drift_ppm']:+.0f} ppm); offset {c['offset_start_ms']:+.1f} ms at the start")
    if "labels" in r:
        label_module().print_summary(r["labels"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session folders under data/raw/sessions (globs allowed)")
    ap.add_argument("--asr", action="store_true", help="transcribe with Whisper large-v3 (GPU box) so the text rules apply")
    ap.add_argument("--force", action="store_true", help="label even if a check failed (never without consent)")
    ap.add_argument("--data-root", default=str(DATA_DIR), help="folder holding raw/, labels/ and manifests/")
    args = ap.parse_args()

    dirs = sorted({Path(p) for pat in args.sessions for p in (glob.glob(pat) or [pat]) if Path(p).is_dir()})
    if not dirs:
        raise SystemExit("no session folders found")
    from pehleaap.audio.vad import SileroVAD
    from pehleaap.labeling.asr import AsrConfig, WhisperASR
    from pehleaap.labeling.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "labeling.yaml")
    asr_cfg = AsrConfig()
    recognize = WhisperASR(asr_cfg) if args.asr else None
    reports = [process(d, Path(args.data_root), cfg, SileroVAD(), recognize, asr_cfg, args.force) for d in dirs]
    for r in reports:
        print_report(r)

    done = [r["session"] for r in reports if "labels" in r]
    print("\nNext:")
    if done:
        print(f"  1. Commit the checksums: git add data/manifests/sessions/  ({', '.join(done)})")
        print(f"  2. Check the labels by ear: python scripts/verify_app.py --annotator <your-name> "
              f"{' '.join(str(Path(args.data_root) / 'labels' / s) for s in done)}   (add --all for the pilot)")
        if not args.asr:
            print("  3. On the GPU box: copy the session folders, then run this script there with --asr (transcripts),")
            print("     and scripts/eval_sessions.py on the labelled sessions (docs/machines.md).")
    if any(not r["ok"] for r in reports):
        print("  Fix the problems above and run again (recordings without consent are never processed).")
    sys.exit(0 if all(r["ok"] for r in reports) else 1)


if __name__ == "__main__":
    main()
