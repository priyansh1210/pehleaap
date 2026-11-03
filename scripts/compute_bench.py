"""CPU cost of end-of-turn decisions on one pinned core, and how many calls that core can carry.

A live agent pays two costs per call: Silero VAD on every 32 ms frame of the caller's audio, and one
turn-model decision at each pause VAD reports (Whisper features of the last 8 s, then one ONNX run).

    streams per core = 1000 / (VAD ms per second of audio + decision ms x decisions per second)

    python scripts/compute_bench.py                                      # Smart Turn int8 vs eot_v1
    python scripts/compute_bench.py --models eot_v1 weights/eot_v1/model_int8.onnx --core 3

Every model runs single-threaded on one core (`--core`) on the same 8 s of audio; each number is the
median of 3 rounds. Prints a markdown table and writes runs/compute_bench.json. Run it on an idle machine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.vad import SileroVAD  # noqa: E402
from pehleaap.baselines.turn_models import SAMPLE_RATE, WINDOW, load_any  # noqa: E402
from pehleaap.paths import RUNS_DIR  # noqa: E402


def median_ms(fn, runs: int, rounds: int = 3) -> float:
    """Median over `rounds` of the median time of `runs` calls, after a warm-up."""
    for _ in range(max(1, runs // 10)):
        fn()
    medians = []
    for _ in range(rounds):
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        medians.append(np.median(times))
    return 1000 * float(np.median(medians))


def streams_per_core(vad_ms_per_s: float, decision_ms: float, decisions_per_min: float) -> float:
    """Calls one core keeps up with: 1000 ms of CPU per second over the busy ms each call needs per second."""
    return 1000 / (vad_ms_per_s + decision_ms * decisions_per_min / 60)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["smart_turn_v3.2_int8", "eot_v1"],
                    help="registry names, weights/<name> folders or .onnx paths (turn_models.load_any)")
    ap.add_argument("--core", type=int, default=0, help="the one CPU core everything runs on")
    ap.add_argument("--runs", type=int, default=50, help="timed decisions per round")
    # shortcut: an assumed pause rate; measure pauses per minute on the recorded sessions once there are some.
    ap.add_argument("--decisions-per-min", type=float, default=20, help="pauses VAD reports per minute of call")
    args = ap.parse_args()

    from env_check import _cpu_name  # scripts/ is on sys.path when this runs as a script

    psutil.Process().cpu_affinity([args.core])
    audio = (0.1 * np.random.default_rng(0).standard_normal(WINDOW)).astype(np.float32)
    vad = SileroVAD(threads=1)
    vad_ms_per_s = median_ms(lambda: vad.probs(audio, SAMPLE_RATE), max(1, args.runs // 10)) / (WINDOW / SAMPLE_RATE)

    rows = []
    for name in args.models:
        model, _ = load_any(name, threads=1)
        features = model.prepare(audio)[None]
        decision = median_ms(lambda: model.predict(audio), args.runs)
        rows.append({"model": name, "decision_ms": round(decision, 2),
                     "model_ms": round(median_ms(lambda: model.predict_prepared(features), args.runs), 2),
                     "streams_per_core": round(streams_per_core(vad_ms_per_s, decision, args.decisions_per_min), 1)})

    result = {"cpu": _cpu_name(), "core": args.core, "vad_ms_per_s": round(vad_ms_per_s, 3),
              "decisions_per_min": args.decisions_per_min, "rows": rows}
    print(f"{result['cpu']}, one core (#{args.core}), 1 thread. Silero VAD (16 kHz): {vad_ms_per_s:.2f} ms per second "
          f"of audio. Streams per core assume {args.decisions_per_min:g} decisions per minute.\n")
    print("| Model | ms per decision (features + model) | model only | streams per core |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['decision_ms']:.1f} | {r['model_ms']:.1f} | {r['streams_per_core']:.0f} |")
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "compute_bench.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
