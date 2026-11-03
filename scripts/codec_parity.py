"""Do two ffmpeg builds make the same phone-line audio? Run every codec in CODECS on the same FLEURS clips with
one build and save the outputs, once per build; --compare then says which codecs came out bit-identical.

    python scripts/codec_parity.py --ffmpeg <ffmpeg.exe> --out runs/codecs_a.npz
    python scripts/codec_parity.py --compare runs/codecs_a.npz runs/codecs_b.npz

One build per process: the codec delay is cached on first use.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.paths import EXTERNAL_DIR  # noqa: E402


def compare(a: dict, b: dict) -> dict[str, str]:
    """Codec -> 'identical', or the largest sample difference, the correlation and the codec delays."""
    out = {}
    for k in sorted(k for k in a if not k.endswith("__delay")):
        x, y = a[k], b[k]
        delays = f"delay {int(a[k + '__delay'][0])} vs {int(b[k + '__delay'][0])} samples"
        if x.shape != y.shape:
            out[k] = f"different length ({len(x)} vs {len(y)}), {delays}"
        elif np.array_equal(x, y):
            out[k] = "identical" if np.array_equal(a[k + "__delay"], b[k + "__delay"]) else f"identical audio, {delays}"
        else:
            out[k] = f"different: max |d| {np.abs(x - y).max():.3g}, correlation {np.corrcoef(x, y)[0, 1]:.4f}, {delays}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", help="the build to run (set as PEHLEAAP_FFMPEG, first in the lookup order)")
    ap.add_argument("--out", help=".npz to write with --ffmpeg")
    ap.add_argument("--clips", type=int, default=20, help="first N Hindi clips of the FLEURS EOT test split")
    ap.add_argument("--compare", nargs=2, metavar="NPZ")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.compare:
        a, b = (dict(np.load(p)) for p in args.compare)
        for codec, verdict in compare(a, b).items():
            print(f"{codec:12s} {verdict}")
        return
    if not (args.ffmpeg and args.out):
        ap.error("pass --ffmpeg and --out, or --compare A B")
    os.environ["PEHLEAAP_FFMPEG"] = args.ffmpeg

    from pehleaap.data.smart_turn_testset import read_shard
    from pehleaap.paths import ffmpeg_bin
    from pehleaap.telephony.codecs import CODECS, codec_delay_samples, narrowband_many

    _, clips = read_shard(EXTERNAL_DIR / "fleurs-hi-eot" / "data" / "test.parquet", ["hin"])
    clips = [np.asarray(c, dtype=np.float32) for c in clips[: args.clips]]
    res = {}
    for codec in CODECS:
        res[codec] = np.concatenate(narrowband_many(clips, codec))
        res[f"{codec}__delay"] = np.array([codec_delay_samples(codec)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **res)
    print(f"{ffmpeg_bin()}: {len(CODECS)} codecs on {len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
