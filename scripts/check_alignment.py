"""Check that two speaker tracks from one session share a clock.

Each track must contain the same sync event (clap or beep) in its first and last `--win`
seconds. Passes if the offset changes by at most `--max-drift-ms` between the two events.

    python scripts/check_alignment.py S017_SPK031_agent.flac S017_SPK032_customer.flac
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.align import measure_drift  # noqa: E402
from pehleaap.audio.io import load_audio  # noqa: E402

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="track A (reference)")
    ap.add_argument("sig", help="track B")
    ap.add_argument("--win", type=float, default=20.0, help="seconds at start and end that contain the sync events")
    ap.add_argument("--max-lag", type=float, default=1.0, help="largest offset to search, seconds")
    ap.add_argument("--max-drift-ms", type=float, default=10.0)
    args = ap.parse_args()

    ref, _ = load_audio(args.ref, sr=SR)
    sig, _ = load_audio(args.sig, sr=SR)
    r = measure_drift(ref, sig, SR, win_s=args.win, max_lag_s=args.max_lag)

    print(f"track lengths     {len(ref) / SR:.1f}s / {len(sig) / SR:.1f}s")
    print(f"offset at start   {r.offset_start_s * 1000:+.1f} ms")
    print(f"offset at end     {r.offset_end_s * 1000:+.1f} ms")
    print(f"drift             {r.drift_ms:+.1f} ms over {r.span_s:.0f}s ({r.drift_ppm:+.0f} ppm)")
    ok = abs(r.drift_ms) <= args.max_drift_ms
    print("PASS" if ok else f"FAIL: drift above {args.max_drift_ms} ms; this tool/session does not keep one clock")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
