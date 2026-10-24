import importlib.util
import io
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.labeling.schema import BC, HOLD, SHIFT, Event, read_events, write_events
from pehleaap.labeling.verify import (
    NONE,
    apply_verdicts,
    cohens_kappa,
    double_annotated,
    load_verdicts,
    needs_review,
    save_verdict,
)

REPO = Path(__file__).resolve().parent.parent


def _events(session="S1", n=200):
    return [Event("customer", SHIFT if k % 2 else HOLD, 1.0 + k, session=session, pause_s=0.4, gap_s=0.3) for k in range(n)]


def test_review_sample_is_stable_and_sized():
    events = _events()
    picked = [e for e in events if needs_review(e)]
    assert 8 <= len(picked) <= 35  # ~10% of 200
    assert picked == [e for e in events if needs_review(e)]  # hash-based: the same every time
    assert needs_review(Event("customer", BC, 5.0, session="S1"))  # backchannels: always
    assert all(needs_review(e, pstn_sessions={"S1"}) for e in events)  # PSTN sessions: everything
    assert 5 <= sum(double_annotated(e) for e in events) <= 35


def test_latest_verdict_wins_and_bad_input_is_refused(tmp_path):
    path = tmp_path / "v.jsonl"
    save_verdict(path, "e1", "correct", SHIFT)
    save_verdict(path, "e1", "wrong", HOLD, note="changed my mind")
    assert load_verdicts(path)["e1"]["label"] == HOLD
    with pytest.raises(ValueError):
        save_verdict(path, "e1", "maybe", SHIFT)
    with pytest.raises(ValueError):
        save_verdict(path, "e1", "correct", "laugh")


def test_apply_verdicts_keeps_retypes_and_drops():
    events = _events(n=5)
    ids = [e.event_id for e in events]
    v = {ids[0]: {"verdict": "correct", "label": events[0].type}, ids[1]: {"verdict": "wrong", "label": BC},
         ids[2]: {"verdict": "wrong", "label": NONE}, ids[3]: {"verdict": "unsure", "label": events[3].type}}
    out = apply_verdicts(events, v)
    assert [(e.type, e.verified) for e in out] == [(events[0].type, True), (BC, True)]  # 2 dropped, 3 unsure, 4 unchecked
    assert out[1].t_event == events[1].t_event


def test_kappa():
    a = {k: {"verdict": "correct", "label": lab} for k, lab in enumerate("SSSHHB")}
    assert cohens_kappa(a, a) == (1.0, 6)
    b = dict(a) | {5: {"verdict": "wrong", "label": "S"}, 0: {"verdict": "unsure", "label": "S"}}
    kappa, n = cohens_kappa(a, b)
    assert n == 5 and 0 < kappa < 1
    assert np.isnan(cohens_kappa(a, {})[0])


def _app():
    spec = importlib.util.spec_from_file_location("verify_app", REPO / "scripts" / "verify_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_app_serves_items_audio_and_saves_verdicts(tmp_path):
    app = _app()
    raw, labels = tmp_path / "raw" / "S1", tmp_path / "labels" / "S1"
    raw.mkdir(parents=True)
    tracks = {}
    for k, role in enumerate(("agent", "customer")):
        tracks[role] = str(raw / f"S1_a_{role}.wav")
        sf.write(tracks[role], 0.1 * np.sin(np.arange(16000 * 20) * (0.05 + 0.02 * k)).astype(np.float32), 16000)
    write_events(labels / "events.jsonl", [Event("customer", BC, 4.0, session="S1", ipu_s=0.4, text="haan"), *_events(n=12)])
    (labels / "summary.json").write_text(json.dumps({"tracks": tracks}), encoding="utf-8")

    state = app.Sessions(app.session_dirs([], tmp_path / "labels"), "tester", everything=True)
    assert len(state.items) == 13 and state.items[0]["type"] == BC
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Label check" in urllib.request.urlopen(base + "/").read()
        items = json.loads(urllib.request.urlopen(base + "/api/items").read())["items"]
        it = items[0]
        q = f"session=S1&start={it['start']}&end={it['end']}"
        x, sr = sf.read(io.BytesIO(urllib.request.urlopen(base + "/api/audio?" + q).read()))
        assert sr == 16000 and x.shape == (int(it["end"] * sr) - int(it["start"] * sr), 2)
        env = json.loads(urllib.request.urlopen(base + "/api/envelope?" + q).read())
        assert len(env["agent"]) == len(env["customer"]) > 100
        req = urllib.request.Request(base + "/api/verdict", data=json.dumps({"event_id": it["event_id"], "verdict": "wrong", "label": NONE}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["label"] == NONE
    finally:
        server.shutdown()
    assert load_verdicts(labels / "verify" / "tester.jsonl")[it["event_id"]]["verdict"] == "wrong"

    app.summary([labels])  # writes events_verified.jsonl from the (only) annotator
    assert read_events(labels / "events_verified.jsonl") == []  # the one checked event was "none": dropped
