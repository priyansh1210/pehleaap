"""How much of each model's phone-line drop does re-picking its decision threshold recover?

Reads the predictions that eval_clip_baselines.py saved (runs/<run>/predictions.csv; no model is
rerun, so this takes seconds) and scores every model with four kinds of threshold
(`pehleaap/eval/calibration.py`): the model card's, one for the phone line vs clean audio, one per
phone/clean x language, and one per exact condition x language (an upper bound: a deployment rarely
knows the codec). Thresholds are cross-fitted: each half of the clips is scored with thresholds
tuned on the other half. Only the phone subset (all Hindi + 25% of English) is used, so clean and
phone rows cover the same clips.

    python scripts/calibrate_thresholds.py --runs clip_baselines clip_baselines_codecs
    python scripts/calibrate_thresholds.py --runs clip_baselines clip_baselines_codecs clip_baselines_chain eot_v0_eval --out calibration_all
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.baselines.turn_models import REGISTRY  # noqa: E402
from pehleaap.eval.calibration import VARIANTS, calibrate, deployment_thresholds  # noqa: E402
from pehleaap.paths import RUNS_DIR  # noqa: E402

KEYS = ["id", "language", "label", "condition"]


def load_predictions(runs: list[str]) -> pd.DataFrame:
    """All runs' predictions, one row per (clip, condition); a model a run didn't score stays NaN there."""
    frames = []
    for run in runs:
        path = RUNS_DIR / run / "predictions.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found: run eval_clip_baselines.py --out {run} first")
        df = pd.read_csv(path, dtype={"id": str})
        frames.append(df[df.phone_subset][KEYS + [c for c in df.columns if c.startswith("p_")]])
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=KEYS, how="outer", suffixes=("", "_dup"))
        for c in [c for c in merged.columns if c.endswith("_dup")]:
            merged[c[:-4]] = merged[c[:-4]].fillna(merged[c])  # the same model scored in two runs: keep the first
            merged = merged.drop(columns=c)
    return merged


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="eval_clip_baselines.py output folders under runs/")
    ap.add_argument("--out", default="calibration")
    args = ap.parse_args()

    df = load_predictions(args.runs)
    models = [c[2:] for c in df.columns if c.startswith("p_")]
    order = {c: i for i, c in enumerate(dict.fromkeys(["clean"] + sorted(set(df.condition) - {"clean"})))}
    lines = [
        "# Decision-threshold calibration on the phone line",
        "",
        f"Predictions from {', '.join(f'`runs/{r}`' for r in args.runs)}: phone subset (all Hindi + 25% of English), so clean and",
        "phone rows cover the same clips. Thresholds are **cross-fitted**: the clips are split in two by a hash of their id, and each",
        "half is scored with a threshold tuned (for accuracy) on the other half. Variants: **card** = the model card's threshold;",
        "**channel** = one for clean audio, one for the phone line; **+lang** = per channel and language; **cond+lang** = per exact",
        "condition and language, an upper bound since a deployment rarely knows the codec. Cut-in = share of unfinished turns called finished.",
    ]
    results = {"runs": args.runs, "variants": {k: v for k, v in VARIANTS.items()}, "models": {}}
    for model in models:
        sub = df[df[f"p_{model}"].notna()]
        card = REGISTRY[model].threshold if model in REGISTRY else 0.5
        res = calibrate(sub, model, card)
        th = deployment_thresholds(sub, model)
        results["models"][model] = {"card_threshold": card, "deployment_thresholds": th, "rows": res.to_dict(orient="records")}
        wide = res.pivot_table(index=["language", "condition"], columns="variant", values=["accuracy", "cut_in", "n"], aggfunc="first")
        lines += ["", f"## {model} (card threshold {card})", "",
                  "Thresholds tuned on all of these clips, per channel/language (reference only: these are test clips; a shipped model"
                  " takes its thresholds from dev data, as train_eot.py's summary.json does): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(th.items())), "",
                  "| Lang | Condition | n | Acc card | channel | +lang | cond+lang | Cut-in card | Cut-in +lang |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for lang, cond in sorted(wide.index, key=lambda k: (k[0], order[k[1]])):
            r = wide.loc[(lang, cond)]
            lines.append(
                f"| {lang} | {cond} | {int(r[('n', 'card')])} | {r[('accuracy', 'card')]:.2f} | {r[('accuracy', 'channel')]:.2f} | "
                f"{r[('accuracy', 'channel+language')]:.2f} | {r[('accuracy', 'condition+language')]:.2f} | "
                f"{100 * r[('cut_in', 'card')]:.1f}% | {100 * r[('cut_in', 'channel+language')]:.1f}% |"
            )
    out = RUNS_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
