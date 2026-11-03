"""A hand-labelled Hindi end-of-turn test from IndicVoices (ai4bharat/IndicVoices, CC BY 4.0): spontaneous speech
no model here trained on.

IndicVoices' segments are cut by length and pauses, not at turn ends (many stop mid-sentence, some mid-word), and
its transcripts have no punctuation, so the labels come from a person listening:

1. `cut`: one clip per speaker from the Hindi valid split's extempore and conversation segments (2-25 s): either
   the segment up to its end, or up to a pause inside it (VAD silence of at least --min-pause-s after the first
   second), both cut 0.2 s into the silence as in make_fleurs_eot.py. Writes the clips, candidates.csv and
   label.html: open it in a browser, mark each clip finished / not finished / unsure, then download labels.json.
   The page plays the clips in a shuffled order and never shows which kind of cut a clip is.
2. `build --labels labels.json`: writes data/test.parquet in Smart Turn's test-set schema (unsure clips are left
   out), so eval_clip_baselines.py scores it directly.

    python scripts/make_indicvoices_eot.py cut
    python scripts/make_indicvoices_eot.py build --labels data/external/indicvoices-hi-eot/labels.json
    python scripts/eval_clip_baselines.py --data-root data/external/indicvoices-hi-eot --langs hin --models ... --out indicvoices_eot

Evaluation only: the audio and transcripts stay in data/ (not in git); reports show only aggregates.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_fleurs_eot import SR, clips_for, flac  # noqa: E402

from pehleaap.paths import EXTERNAL_DIR  # noqa: E402

SRC = EXTERNAL_DIR / "indicvoices" / "hindi" / "valid-00000-of-00001.parquet"
OUT = EXTERNAL_DIR / "indicvoices-hi-eot"
SCENARIOS = ("Extempore", "Conversation")
LABELS = ("finished", "not_finished", "unsure")


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def pick_segments(meta: list[dict], n: int, min_s: float = 2.0, max_s: float = 25.0) -> list[dict]:
    """One segment per speaker (a fixed, hash-chosen one), for the first `n` speakers in hash order."""
    by_speaker: dict[str, list[dict]] = {}
    for m in meta:
        if m["scenario"] in SCENARIOS and min_s <= m["duration"] <= max_s:
            by_speaker.setdefault(m["speaker_id"], []).append(m)
    speakers = sorted(by_speaker, key=h)[:n]
    return [min(by_speaker[s], key=lambda m: h(s + m["path"])) for s in speakers]


def with_tail(x: np.ndarray, clip: np.ndarray, ipus: list, tail_s: float = 0.2) -> tuple[np.ndarray, float]:
    """`clip` with silence appended until `tail_s` follows its last speech, as on FLEURS and the pause clips.
    IndicVoices trims segments at the last word, so a cut at the segment end would otherwise end in speech. The
    filler is the recording's own background: the middle of its longest pause, else noise at its quietest level."""
    need = int(round((ipus[-1].end + tail_s) * SR)) - len(clip)
    if need <= 0:
        return clip, 0.0
    fade = min(80, len(clip))  # 5 ms crossfade over the join, so it doesn't click
    n = need + fade
    gaps = [(0.0, ipus[0].start)] + [(a.end, b.start) for a, b in zip(ipus[:-1], ipus[1:])]
    s, e = max(gaps, key=lambda g: g[1] - g[0])
    quiet = x[int(s * SR) : int(e * SR)]
    if len(quiet) >= n:
        fill = quiet[(len(quiet) - n) // 2 :][:n]
    else:  # shortcut: white noise at the quietest 20 ms frame's level; real background is better when there is one
        frames = x[: len(x) // 320 * 320].reshape(-1, 320)
        level = float(np.sqrt((frames**2).mean(axis=1)).min()) if len(frames) else 1e-4
        fill = np.random.default_rng(len(x)).normal(0, level, n).astype(np.float32)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    head, tail = clip[: len(clip) - fade], clip[len(clip) - fade :]
    return np.concatenate([head, tail * (1 - ramp) + fill[:fade] * ramp, fill[fade:]]), round(need / SR, 3)


def page(clips: list[dict]) -> str:
    """A self-contained labelling page: audio from clips/, labels kept in the browser and downloaded as labels.json."""
    order = sorted(clips, key=lambda c: h("order" + c["id"]))
    items = json.dumps([{"id": c["id"], "src": f"clips/{c['id']}.flac"} for c in order])
    return """<!doctype html><html><head><meta charset="utf-8"><title>IndicVoices labels</title>
<style>body{font:16px system-ui,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
button{font-size:16px;padding:10px 16px;margin:4px}#cur{font-size:20px;margin:16px 0}.done{color:#080}
audio{width:100%}</style></head><body>
<h1>Is the speaker finished?</h1>
<p>Listen to the end of each clip. <b>Finished (F)</b>: they have said what they were saying, so it would be natural for the
other person to reply now. <b>Not finished (N)</b>: they were clearly going to carry on. <b>Unsure (U)</b>: you can't tell.
Space plays again, arrow keys move. Labels stay in this browser until you press Download.</p>
<div id="cur"></div><audio id="a" controls></audio>
<div><button onclick="mark('finished')">Finished (F)</button><button onclick="mark('not_finished')">Not finished (N)</button>
<button onclick="mark('unsure')">Unsure (U)</button></div>
<p><button onclick="go(-1)">&larr; Back</button><button onclick="go(1)">Skip &rarr;</button>
<button onclick="save()">Download labels.json</button> <span id="count"></span></p>
<script>
const items=""" + items + """;let labels={};try{labels=JSON.parse(localStorage.getItem('iv_labels')||'{}')}catch(e){}
let i=items.findIndex(x=>!labels[x.id]);if(i<0)i=0;const a=document.getElementById('a');
function show(){const it=items[i];document.getElementById('cur').innerHTML='Clip '+(i+1)+' of '+items.length+
(labels[it.id]?' <span class="done">('+labels[it.id].replace('_',' ')+')</span>':'');a.src=it.src;a.play().catch(()=>{});
document.getElementById('count').textContent=Object.keys(labels).length+' labelled';}
function go(d){i=Math.max(0,Math.min(items.length-1,i+d));show();}
function mark(l){labels[items[i].id]=l;try{localStorage.setItem('iv_labels',JSON.stringify(labels))}catch(e){}go(1);}
function save(){const b=new Blob([JSON.stringify(labels,null,1)],{type:'application/json'});const u=document.createElement('a');
u.href=URL.createObjectURL(b);u.download='labels.json';u.click();}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k==='f')mark('finished');else if(k==='n')mark('not_finished');
else if(k==='u')mark('unsure');else if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);
else if(e.key===' '){e.preventDefault();a.currentTime=0;a.play();}});show();
</script></body></html>
"""


def cut(args) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf

    from pehleaap.audio.vad import SileroVAD, hop_s
    from pehleaap.labeling.config import LabelConfig
    from pehleaap.labeling.ipu import ipus_from_probs

    table = pq.read_table(args.src, columns=["audio_filepath", "speaker_id", "scenario", "duration"])
    audio = table.column("audio_filepath").combine_chunks()
    meta = [{"row": i, "path": p, **r} for i, (p, r) in enumerate(zip(
        audio.field("path").to_pylist(), table.select(["speaker_id", "scenario", "duration"]).to_pylist()))]
    segments = pick_segments(meta, args.n)
    audio = audio.field("bytes")
    vad, cfg = SileroVAD(), LabelConfig()
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    clips = []
    for m in segments:
        x, sr = sf.read(io.BytesIO(audio[m["row"]].as_py()), dtype="float32", always_2d=True)
        if sr != SR:
            raise SystemExit(f"{m['path']}: {sr} Hz, expected {SR}")
        x = x.mean(axis=1)
        found = clips_for(x, m["path"], vad, cfg, args.min_pause_s, 0.2, 1.0, max_pauses=1)
        if not found:
            continue
        want_pause = h("kind" + m["speaker_id"]) % 2 == 1
        pauses = [c for c in found if not c[0]]
        _, clip, kind = pauses[0] if (want_pause and pauses) else found[0]
        def vad_tail(y):  # silence after the last speech, as the VAD hears the clip itself
            ip = ipus_from_probs(vad.probs(y, SR), hop_s(SR), cfg)
            return len(y) / SR - (ip[-1].end if ip else 0.0)

        padded = 0.0
        if kind == "end":
            # Pad until the VAD hears the same 0.2 s on the clip as on the pause clips: speech cut off at the segment
            # boundary runs on a little into the filler (VAD hangover), so one pass falls short.
            ipus, extra, clip0 = ipus_from_probs(vad.probs(x, SR), hop_s(SR), cfg), 0.0, clip
            for _ in range(3):
                clip, padded = with_tail(x, clip0, ipus, 0.2 + extra)
                short = 0.2 - vad_tail(clip)
                if short <= 0.02:
                    break
                extra += short
        cid = f"iv_hi_{h(m['path'] + kind):032x}"[:16]  # opaque, so neither the page nor a file name gives the cut away
        (OUT / "clips" / f"{cid}.flac").write_bytes(flac(clip))
        clips.append({"id": cid, "kind": kind, "speaker_id": m["speaker_id"], "scenario": m["scenario"], "source": m["path"],
                      "seconds": round(len(clip) / SR, 2), "padded_s": padded, "tail_vad_s": round(vad_tail(clip), 3)})
    with open(OUT / "candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        w.writerows(clips)
    (OUT / "label.html").write_text(page(clips), encoding="utf-8")
    n_pause = sum(c["kind"] != "end" for c in clips)
    print(f"{len(clips)} clips from {len(clips)} speakers ({len(clips) - n_pause} segment ends, {n_pause} mid-segment pauses) -> {OUT}")
    print(f"label them: open {OUT / 'label.html'} in a browser")


def build(args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if bad := sorted({v for v in labels.values()} - set(LABELS)):
        raise SystemExit(f"unknown labels {bad}")
    with open(OUT / "candidates.csv", encoding="utf-8", newline="") as f:
        cands = list(csv.DictReader(f))
    rows = [{"id": c["id"], "language": "hin", "endpoint_bool": labels[c["id"]] == "finished", "synthetic": False,
             "dataset": "indicvoices_hi_valid", "midfiller": False, "endfiller": False,
             "audio": {"bytes": (OUT / "clips" / f"{c['id']}.flac").read_bytes(), "path": f"{c['id']}.flac"}}
            for c in cands if labels.get(c["id"]) in ("finished", "not_finished")]
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), OUT / "data" / "test.parquet")
    (OUT / "README.md").write_text(
        "# IndicVoices Hindi spontaneous speech (ai4bharat/IndicVoices, CC BY 4.0), one clip per speaker, cut at a segment end "
        "or 0.2 s into a mid-segment pause, labelled finished / not finished by a person listening\n\n"
        "Built by scripts/make_indicvoices_eot.py; see its docstring.\n", encoding="utf-8")
    n_fin = sum(r["endpoint_bool"] for r in rows)
    skipped = sum(1 for c in cands if labels.get(c["id"]) not in ("finished", "not_finished"))
    print(f"{len(rows)} clips ({n_fin} finished, {len(rows) - n_fin} not finished); {skipped} unsure or unlabelled left out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--src", default=str(SRC))
    c.add_argument("--n", type=int, default=300, help="speakers, one clip each")
    c.add_argument("--min-pause-s", type=float, default=0.25)
    b = sub.add_parser("build")
    b.add_argument("--labels", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"cut": cut, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
