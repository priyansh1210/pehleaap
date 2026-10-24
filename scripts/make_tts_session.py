"""A stand-in two-track session for trying the labeling + ASR pipeline before real recordings exist.

Stitches Hindi clips from Smart Turn's test set (synthetic TTS) into agent and customer tracks that
take turns with 0.8 s gaps, named like a real session (<session>_<speaker>_<role>.wav). It goes in
runs/, not data/raw/: it is not a recording and has no consent form behind it, because it needs none.

    python scripts/make_tts_session.py                          # runs/tts_session/TTS01, 6 turns
    python scripts/label_session.py runs/tts_session/TTS01 --asr --out runs/tts_labels
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import save_audio  # noqa: E402
from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.paths import RUNS_DIR  # noqa: E402

SR = smart_turn_testset.SAMPLE_RATE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--gap-s", type=float, default=0.8)
    ap.add_argument("--out", default=str(RUNS_DIR / "tts_session" / "TTS01"))
    args = ap.parse_args()

    shards = smart_turn_testset.shard_paths()
    if not shards:
        raise SystemExit("Smart Turn's test set isn't downloaded: run scripts/eval_clip_baselines.py once, or repro_smart_turn.py")
    _, clips = smart_turn_testset.read_shard(shards[0], ["hin"])
    parts = {"agent": [], "customer": []}
    t = 0.5
    for k in range(args.turns):
        clip = clips[k][: SR * 6]
        role = "agent" if k % 2 == 0 else "customer"
        parts[role].append((int(t * SR), clip))
        print(f"{role:<8} {t:6.2f}-{t + len(clip) / SR:6.2f} s")
        t += len(clip) / SR + args.gap_s
    out = Path(args.out)
    for role, pieces in parts.items():
        x = np.zeros(int((t + 1) * SR), dtype=np.float32)
        for start, clip in pieces:
            x[start : start + len(clip)] += clip
        save_audio(out / f"{out.name}_spk1_{role}.wav", x, SR)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
