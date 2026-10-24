"""Check the automatic labels by ear (plan §4.2 step 4): a local web page, one event at a time.

For each event that needs review (`pehleaap/labeling/verify.py`: every backchannel, interruption
and hold request, PSTN sessions in full, 10% of SHIFT/HOLD) it plays the moment in stereo (agent
left, customer right), draws both speakers' loudness with the event marked, shows the label and its
ASR text, and records your verdict with one key. Verdicts go to
data/labels/<session>/verify/<annotator>.jsonl, so nothing is lost if you close the page.

It serves on 127.0.0.1 only: the recordings never leave this machine.

    python scripts/verify_app.py --annotator priya                        # every labelled session
    python scripts/verify_app.py --annotator priya "data/labels/S01*"     # some sessions
    python scripts/verify_app.py --annotator ravi --double-only           # second annotator: the kappa subset
    python scripts/verify_app.py --annotator priya --all data/labels/P01  # every event (the pilot: tune the rules on it)
    python scripts/verify_app.py --summary                                # progress, label accuracy, kappa;
                                                                          # writes events_verified.jsonl
"""

import argparse
import csv
import glob
import io
import json
import sys
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import load_audio  # noqa: E402
from pehleaap.labeling.schema import read_events, write_events  # noqa: E402
from pehleaap.labeling.verify import (  # noqa: E402
    CORRECT,
    LABELS,
    UNSURE,
    apply_verdicts,
    cohens_kappa,
    double_annotated,
    load_verdicts,
    needs_review,
    save_verdict,
    verdict_path,
)
from pehleaap.paths import DATA_DIR  # noqa: E402

SR = 16000
ROLES = ("agent", "customer")
BEFORE_S, AFTER_S, MAX_S = 3.0, 2.0, 14.0  # context around an event, and the longest clip


def pstn_sessions() -> set[str]:
    """Sessions whose device is "pstn" in data/raw/sessions.csv (the real-phone test set)."""
    path = DATA_DIR / "raw" / "sessions.csv"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["session_id"] for r in csv.DictReader(f) if "pstn" in (r.get("device_agent", ""), r.get("device_customer", ""))}


def session_dirs(patterns: list[str], labels_dir: Path) -> list[Path]:
    found = {Path(p) for pat in patterns for p in glob.glob(pat)} if patterns else set(labels_dir.iterdir()) if labels_dir.is_dir() else set()
    return sorted(d for d in found if (d / "events.jsonl").exists() and (d / "summary.json").exists())


def clip_window(e) -> tuple[float, float, float]:
    """(start, end, event offset in the clip) in seconds."""
    # shift: up to the other person's reply (gap); hold / hold_req: the pause until they resume; bc / interrupt: the speech itself
    span = {"shift": e.gap_s, "hold": e.pause_s, "hold_req": e.pause_s}.get(e.type, e.ipu_s) or 0.0
    span = max(span, 0.0)  # a reply that overlaps the end of the turn has a negative gap
    start = max(0.0, e.t_event - BEFORE_S)
    return start, min(start + MAX_S, e.t_event + span + AFTER_S), e.t_event - start


class Sessions:
    def __init__(self, dirs: list[Path], annotator: str, double_only: bool = False, everything: bool = False):
        self.annotator, self.items, self.dirs, self.tracks = annotator, [], {}, {}
        pstn = pstn_sessions()
        for d in dirs:
            session = d.name
            self.dirs[session] = d
            self.tracks[session] = json.loads((d / "summary.json").read_text(encoding="utf-8"))["tracks"]
            done = load_verdicts(verdict_path(d.parent, session, annotator))
            for e in read_events(d / "events.jsonl"):
                if not (everything or needs_review(e, pstn)) or (double_only and not double_annotated(e)):
                    continue
                start, end, offset = clip_window(e)
                self.items.append({"session": session, "event_id": e.event_id, "type": e.type, "channel": e.channel,
                                   "t_event": round(e.t_event, 2), "text": e.text, "pause_s": e.pause_s, "gap_s": e.gap_s,
                                   "ipu_s": e.ipu_s, "bc_type": e.bc_type, "start": round(start, 3), "end": round(end, 3),
                                   "offset": round(offset, 3), "done": done.get(e.event_id)})

    @lru_cache(maxsize=4)
    def audio(self, session: str) -> dict[str, np.ndarray]:
        return {role: load_audio(self.tracks[session][role], sr=SR)[0] for role in ROLES}

    def clip(self, session: str, start: float, end: float) -> np.ndarray:
        a, b = int(start * SR), int(end * SR)
        tracks = self.audio(session)
        return np.stack([np.pad(tracks[r][a:b], (0, max(0, b - a - len(tracks[r][a:b])))) for r in ROLES], axis=1)


def make_handler(state: Sessions):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/items":
                self._json({"annotator": state.annotator, "labels": LABELS, "items": state.items})
            elif url.path in ("/api/audio", "/api/envelope"):
                if q.get("session") not in state.tracks:
                    return self._json({"error": "unknown session"}, 404)
                x = state.clip(q["session"], float(q["start"]), float(q["end"]))
                if url.path == "/api/audio":
                    buf = io.BytesIO()
                    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
                    return self._send(buf.getvalue(), "audio/wav")
                hop = SR // 100  # 10 ms
                n = len(x) // hop
                env = np.sqrt((x[: n * hop].reshape(n, hop, 2) ** 2).mean(axis=1))
                db = np.clip(20 * np.log10(env + 1e-6) + 60, 0, 60) / 60  # -60..0 dBFS -> 0..1
                self._json({"hop": 0.01, "agent": db[:, 0].round(3).tolist(), "customer": db[:, 1].round(3).tolist()})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/verdict":
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            item = next((i for i in state.items if i["event_id"] == body.get("event_id")), None)
            if item is None:
                return self._json({"error": "unknown event"}, 404)
            try:
                row = save_verdict(verdict_path(state.dirs[item["session"]].parent, item["session"], state.annotator), item["event_id"],
                                   body.get("verdict"), body.get("label"), body.get("note", ""))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            item["done"] = row
            self._json(row)

    return Handler


def summary(dirs: list[Path]) -> None:
    """Progress per annotator, how often each automatic label was right, kappa, and events_verified.jsonl per session."""
    pstn = pstn_sessions()
    totals: dict[str, dict] = {}
    pairs: dict[tuple[str, str], list] = {}
    for d in dirs:
        events = read_events(d / "events.jsonl")
        review = [e for e in events if needs_review(e, pstn)]
        files = sorted((d / "verify").glob("*.jsonl")) if (d / "verify").is_dir() else []
        by_annotator = {f.stem: load_verdicts(f) for f in files}
        for name, v in by_annotator.items():
            t = totals.setdefault(name, {"done": 0, "todo": 0, "right": {}, "seen": {}})
            t["done"] += sum(e.event_id in v for e in review)
            t["todo"] += len(review)
            for e in review:
                if e.event_id in v and v[e.event_id]["verdict"] != UNSURE:
                    t["seen"][e.type] = t["seen"].get(e.type, 0) + 1
                    t["right"][e.type] = t["right"].get(e.type, 0) + (v[e.event_id]["verdict"] == CORRECT)
        names = sorted(by_annotator)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs.setdefault((a, b), []).append((by_annotator[a], by_annotator[b]))
        if names:  # the first annotator (alphabetically) is the reference for the verified labels
            verified = apply_verdicts(events, by_annotator[names[0]])
            write_events(d / "events_verified.jsonl", verified)
            print(f"{d.name}: {len(verified)} verified events (from {names[0]}) -> {d / 'events_verified.jsonl'}")
    for name, t in totals.items():
        acc = ", ".join(f"{ty} {t['right'][ty]}/{n}" for ty, n in sorted(t["seen"].items()))
        print(f"{name}: {t['done']} of {t['todo']} checked; automatic label right: {acc or '-'}")
    for (a, b), runs in pairs.items():
        merged_a = {k: v for va, _ in runs for k, v in va.items()}
        merged_b = {k: v for _, vb in runs for k, v in vb.items()}
        kappa, n = cohens_kappa(merged_a, merged_b)
        print(f"kappa {a} vs {b}: {kappa:.3f} on {n} events")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="label folders (globs allowed); default: every folder in --labels")
    ap.add_argument("--labels", default=str(DATA_DIR / "labels"))
    ap.add_argument("--annotator", help="your short name (no spaces); verdicts are saved under it")
    ap.add_argument("--double-only", action="store_true", help="only the 10% kappa subset (for the second annotator)")
    ap.add_argument("--all", action="store_true", help="every event, not just the review sample (e.g. the pilot session)")
    ap.add_argument("--summary", action="store_true", help="print progress and kappa, write events_verified.jsonl")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(args.labels)
    dirs = session_dirs(args.sessions, labels_dir)
    if not dirs:
        raise SystemExit(f"no labelled sessions (events.jsonl + summary.json) in {args.sessions or labels_dir}")
    if args.summary:
        return summary(dirs)
    if not args.annotator or not args.annotator.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("give --annotator: a short name of letters, digits, - or _")
    state = Sessions(dirs, args.annotator, args.double_only, args.all)
    todo = sum(i["done"] is None for i in state.items)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(state.items)} events to check in {len(dirs)} sessions ({todo} not done yet). Open {url}  (Ctrl+C to stop)")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label check</title>
<style>
  :root { --bg: #fff; --fg: #111; --muted: #555; --card: #f5f5f4; --line: #d6d3d1; --agent: #2563eb; --cust: #c2410c;
          --ok: #15803d; --bad: #b91c1c; --unsure: #a16207; --mark: #111; }
  @media (prefers-color-scheme: dark) { :root { --bg: #18181b; --fg: #f4f4f5; --muted: #a1a1aa; --card: #27272a; --line: #3f3f46;
          --agent: #60a5fa; --cust: #fb923c; --ok: #4ade80; --bad: #f87171; --unsure: #facc15; --mark: #f4f4f5; } }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; }
  main { max-width: 900px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 1.15rem; margin: 0; }
  .muted { color: var(--muted); }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 12px 0; }
  .type { font-size: 1.6rem; font-weight: 700; }
  .agent { color: var(--agent); } .customer { color: var(--cust); }
  .text { font-size: 1.1rem; margin: 8px 0; min-height: 1.4em; }
  canvas { width: 100%; height: 140px; display: block; border-radius: 6px; background: var(--bg); border: 1px solid var(--line); cursor: pointer; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  button { font: inherit; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); cursor: pointer; }
  button kbd { font-size: .8em; opacity: .75; margin-left: 4px; }
  button.ok { border-color: var(--ok); } button.bad { border-color: var(--bad); } button.unsure { border-color: var(--unsure); }
  #fix { display: none; } #fix.show { display: flex; }
  .status { font-weight: 600; } .status.correct { color: var(--ok); } .status.wrong { color: var(--bad); } .status.unsure { color: var(--unsure); }
  input[type=text] { font: inherit; width: 100%; padding: 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
  details { margin-top: 12px; } dt { font-weight: 600; margin-top: 6px; } dd { margin: 0 0 0 1em; }
</style></head>
<body><main>
<header><h1>Label check <span class="muted" id="who"></span></h1><span id="progress" class="muted"></span></header>

<div class="card">
  <div class="row" style="justify-content: space-between; margin-top: 0">
    <div><span class="type" id="type"></span> <span id="channel"></span></div>
    <div class="muted" id="where"></div>
  </div>
  <div class="text" id="text"></div>
  <div class="muted" id="facts"></div>
  <canvas id="wave" width="1700" height="280" title="Click to play from there"></canvas>
  <div class="muted" style="margin-top: 4px"><span class="agent">&#9632; agent (left ear)</span> &nbsp; <span class="customer">&#9632; customer (right ear)</span> &nbsp; black line = the event</div>
  <audio id="audio"></audio>
  <div class="row">
    <button id="play">Play / pause <kbd>Space</kbd></button>
    <button id="fromEvent">From the event <kbd>E</kbd></button>
  </div>
</div>

<div class="card">
  <div>Is the label right? <span class="status" id="status"></span></div>
  <div class="row">
    <button class="ok" data-verdict="correct">Correct <kbd>1</kbd></button>
    <button class="bad" data-verdict="wrong">Wrong, it's... <kbd>2</kbd></button>
    <button class="unsure" data-verdict="unsure">Can't tell <kbd>3</kbd></button>
  </div>
  <div class="row" id="fix"></div>
  <div class="row"><input type="text" id="note" placeholder="Note (optional), e.g. 'two people laughing'"></div>
  <div class="row">
    <button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button><button id="nextTodo">Next unchecked <kbd>N</kbd></button>
  </div>
</div>

<details><summary>What the labels mean</summary><dl>
  <dt>shift</dt><dd>The speaker paused and the other person took the turn (a real turn end).</dd>
  <dt>hold</dt><dd>The speaker paused but carried on themselves ("mera order... kal aana tha"). The other's "haan" doesn't count.</dd>
  <dt>bc (backchannel)</dt><dd>A short listener response while the other keeps talking: "haan", "hmm", "achha", "ji", "theek hai".</dd>
  <dt>interrupt</dt><dd>A real attempt to take the floor while the other is speaking, not just "haan".</dd>
  <dt>hold_req</dt><dd>"Ek minute", "ruko", "hold karo", then silence while they look something up.</dd>
  <dt>noise_echo</dt><dd>No one actually spoke: speakerphone echo of the other person, a TV, traffic, a cough.</dd>
  <dt>none</dt><dd>Not a turn-taking moment at all (e.g. the detector split one sentence oddly).</dd>
</dl></details>
</main>
<script>
const LABEL_KEYS = { shift: "S", hold: "H", bc: "B", interrupt: "I", hold_req: "Q", noise_echo: "X", none: "0" };
let items = [], labels = [], i = 0, env = null;
const $ = id => document.getElementById(id);
const audio = $("audio"), canvas = $("wave"), ctx = canvas.getContext("2d");

async function init() {
  const r = await (await fetch("/api/items")).json();
  items = r.items; labels = r.labels;
  $("who").textContent = "- " + r.annotator;
  $("fix").innerHTML = labels.map(l => `<button data-label="${l}">${l} <kbd>${LABEL_KEYS[l] || ""}</kbd></button>`).join("");
  document.querySelectorAll("#fix button").forEach(b => b.onclick = () => decide("wrong", b.dataset.label));
  document.querySelectorAll("[data-verdict]").forEach(b => b.onclick = () => verdictButton(b.dataset.verdict));
  i = Math.max(0, items.findIndex(x => !x.done));
  if (!items.length) { $("type").textContent = "Nothing to check"; return; }
  show();
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

async function show() {
  const it = items[i];
  $("type").textContent = it.type;
  $("channel").innerHTML = `by <span class="${it.channel}">${it.channel}</span>`;
  $("where").textContent = `${it.session} - ${it.t_event.toFixed(2)} s - ${i + 1} of ${items.length}`;
  $("text").textContent = it.text ? `"${it.text}"` : (it.text === "" ? "(ASR heard nothing)" : "(no transcript)");
  const facts = [];
  if (it.pause_s != null) facts.push(`pause ${it.pause_s.toFixed(2)} s`);
  if (it.gap_s != null) facts.push(`gap ${it.gap_s.toFixed(2)} s`);
  if (it.ipu_s != null) facts.push(`speech ${it.ipu_s.toFixed(2)} s`);
  if (it.bc_type) facts.push(it.bc_type);
  $("facts").textContent = facts.join(" - ");
  $("note").value = it.done ? it.done.note || "" : "";
  $("fix").classList.remove("show");
  showStatus(it);
  progress();
  const q = `session=${encodeURIComponent(it.session)}&start=${it.start}&end=${it.end}`;
  audio.src = "/api/audio?" + q;
  env = await (await fetch("/api/envelope?" + q)).json();
  draw();
  audio.currentTime = 0;
  audio.play().catch(() => {});
}

function showStatus(it) {
  const s = $("status");
  s.className = "status " + (it.done ? it.done.verdict : "");
  s.textContent = !it.done ? "" : it.done.verdict === "wrong" ? `- you said: ${it.done.label}` : `- you said: ${it.done.verdict}`;
}

function progress() {
  const done = items.filter(x => x.done).length;
  $("progress").textContent = `${done} of ${items.length} checked`;
}

function draw() {
  const w = canvas.width, h = canvas.height, it = items[i];
  ctx.clearRect(0, 0, w, h);
  if (!env) return;
  const n = env.agent.length, half = h / 2;
  for (const [key, color, top] of [["agent", cssVar("--agent"), 0], ["customer", cssVar("--cust"), half]]) {
    ctx.fillStyle = color;
    env[key].forEach((v, k) => { const bh = v * (half - 6); ctx.fillRect(k / n * w, top + half - 3 - bh, Math.max(1, w / n), bh); });
  }
  const dur = it.end - it.start;
  ctx.fillStyle = cssVar("--mark");
  ctx.fillRect(it.offset / dur * w - 2, 0, 4, h);
  ctx.fillStyle = cssVar("--muted");
  ctx.fillRect(audio.currentTime / dur * w - 1, 0, 2, h);
}

function verdictButton(v) {
  if (v === "wrong") { $("fix").classList.add("show"); return; }
  decide(v, items[i].type);
}

async function decide(verdict, label) {
  const it = items[i];
  const r = await fetch("/api/verdict", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_id: it.event_id, verdict, label, note: $("note").value }) });
  if (!r.ok) { const s = $("status"); s.className = "status wrong"; s.textContent = "- not saved: " + (await r.json()).error; return; }
  it.done = await r.json();
  next(true);
}

function next(unchecked) {
  if (unchecked) {
    const k = items.findIndex((x, j) => j > i && !x.done);
    if (k >= 0) { i = k; return show(); }
  }
  if (i < items.length - 1) { i++; show(); } else { showStatus(items[i]); progress(); }
}

$("play").onclick = () => audio.paused ? audio.play() : audio.pause();
$("fromEvent").onclick = () => { audio.currentTime = Math.max(0, items[i].offset - 0.5); audio.play(); };
$("prev").onclick = () => { if (i > 0) { i--; show(); } };
$("next").onclick = () => next(false);
$("nextTodo").onclick = () => next(true);
canvas.onclick = ev => { const r = canvas.getBoundingClientRect(); audio.currentTime = (ev.clientX - r.left) / r.width * (items[i].end - items[i].start); audio.play(); };
audio.ontimeupdate = draw;

document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (k === " ") { ev.preventDefault(); $("play").click(); }
  else if (k === "e") $("fromEvent").click();
  else if (k === "1") verdictButton("correct");
  else if (k === "2") verdictButton("wrong");
  else if (k === "3") verdictButton("unsure");
  else if (k === "n") next(true);
  else if (k === "arrowright") next(false);
  else if (k === "arrowleft") $("prev").click();
  else if ($("fix").classList.contains("show")) {
    const label = Object.keys(LABEL_KEYS).find(l => LABEL_KEYS[l].toLowerCase() === k);
    if (label) decide("wrong", label);
  }
});
init();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
