import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.data.splits import Session, add_to_train, assign, freeze, load, speaker_groups
from pehleaap.labeling.schema import HOLD, SHIFT, Event, write_events
from pehleaap.train.eot_data import EotStore

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def _sessions():
    """20 pairs plus three people who also recorded with a second partner, and one PSTN pair."""
    out = []
    for k in range(20):
        a, b = f"SPK{2 * k:03d}", f"SPK{2 * k + 1:03d}"
        out.append(Session(f"S{k:03d}", (a, b), ("earphones", "handset" if k % 3 else "speakerphone"), 25.0, ("F" if k % 2 else "M", "M")))
    out += [Session("S100", ("SPK000", "SPK010"), ("earphones", "earphones"), 25.0),  # links pairs 0 and 5
            Session("S101", ("SPK020", "SPK021"), ("earphones", "earphones"), 25.0),  # a second session of pair 10
            Session("P900", ("SPK090", "SPK091"), ("pstn", "pstn"), 20.0),
            Session("S102", ("SPK091", "SPK030"), ("earphones", "earphones"), 25.0)]  # a PSTN speaker's other session
    return out


def test_groups_follow_shared_speakers():
    groups = {frozenset(s.session_id for s in g.sessions) for g in speaker_groups(_sessions())}
    assert frozenset({"S000", "S005", "S100"}) in groups and frozenset({"P900", "S102", "S015"}) in groups


def test_assignment_is_speaker_disjoint_balanced_and_puts_pstn_in_test():
    sessions = _sessions()
    split = assign(sessions)
    where: dict[str, set] = {}
    for s in sessions:
        for p in s.speakers:
            where.setdefault(p, set()).add(split[s.session_id])
    assert all(len(v) == 1 for v in where.values())
    assert split["P900"] == split["S102"] == split["S015"] == "test"
    minutes = {k: sum(s.minutes for s in sessions if split[s.session_id] == k) for k in ("train", "dev", "test")}
    total = sum(minutes.values())
    assert 0.6 < minutes["train"] / total < 0.8 and 0.05 < minutes["dev"] / total < 0.2 and 0.15 < minutes["test"] / total < 0.3
    assert assign(sessions) == split  # deterministic


def test_freeze_lock_detects_changes_and_new_sessions_only_join_train(tmp_path):
    sessions = _sessions()
    split = assign(sessions)
    freeze(split, tmp_path, "2026-11-15")
    assert load(tmp_path) == split
    test_speaker = next(p for s in sessions if split[s.session_id] == "test" for p in s.speakers)
    new = [Session("S200", ("SPK500", "SPK501"), ("earphones", "earphones")), Session("S201", (test_speaker, "SPK502"), ("handset", "handset"))]
    by_id = {s.session_id: s for s in sessions + new}
    added, held = add_to_train(new, load(tmp_path), by_id, tmp_path)
    assert added == ["S200"] and held == ["S201"]
    assert load(tmp_path)["S200"] == "train"  # train re-locked; dev/test untouched
    (tmp_path / "test.txt").write_text("S999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the splits were frozen"):
        load(tmp_path)


def _labelled_session(labels: Path, name: str) -> Path:
    d = labels / name
    d.mkdir(parents=True)
    tracks = {}
    for role in ("agent", "customer"):
        tracks[role] = str(labels / f"{name}_{role}.wav")
        sf.write(tracks[role], (0.05 * np.random.default_rng(len(role)).standard_normal(SR * 20)).astype(np.float32), SR)
    write_events(d / "events.jsonl", [Event("customer", SHIFT, 5.0, session=name, pause_s=0.8), Event("customer", HOLD, 9.0, session=name, pause_s=0.5),
                                      Event("customer", HOLD, 12.0, session=name, pause_s=0.15), Event("agent", SHIFT, 7.0, session=name, pause_s=1.0)])
    (d / "summary.json").write_text(json.dumps({"tracks": tracks}), encoding="utf-8")
    return d


def test_sessions_to_store_takes_train_and_dev_and_refuses_test(tmp_path):
    labels, splits_dir, store_dir = tmp_path / "labels", tmp_path / "splits", tmp_path / "store"
    for name in ("S001", "S002", "S003"):
        _labelled_session(labels, name)
    script = [sys.executable, str(REPO / "scripts" / "sessions_to_store.py"), str(labels / "S00*"), "--store", str(store_dir), "--splits-dir", str(splits_dir)]
    r = subprocess.run(script, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "aren't frozen" in (r.stdout + r.stderr)  # nothing before the freeze
    freeze({"S001": "train", "S002": "dev", "S003": "test"}, splits_dir, "2026-11-15")
    r = subprocess.run(script, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "TEST session" in r.stdout
    idx = EotStore(store_dir).index
    assert set(idx.dataset) == {"pehleaap_S001", "pehleaap_S002"}
    assert dict(idx.groupby("dataset").split.first()) == {"pehleaap_S001": "train", "pehleaap_S002": "dev"}
    s1 = idx[idx.dataset == "pehleaap_S001"]
    assert sorted(s1.label) == [0, 1, 1]  # the 0.15 s hold is skipped: the clip would contain resumed speech
    # each clip ends 0.2 s into its pause; the store keeps the last 8 s (all the model reads)
    assert sorted(s1.length) == [min(int(round(t * SR)), 8 * SR) for t in (5.2, 7.2, 9.2)]
    again = subprocess.run(script, capture_output=True, text=True, encoding="utf-8")
    assert "already in the store" in again.stdout and len(EotStore(store_dir).index) == len(idx)


def test_make_splits_script_proposes_then_freezes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = "".join(f"S{k:03d},2026-11-0{1 + k % 9},craig,SPK{2 * k:03d},SPK{2 * k + 1:03d},earphones,handset\n" for k in range(12))
    (raw / "sessions.csv").write_text("session_id,date,tool,spk_agent,spk_customer,device_agent,device_customer\n" + rows, encoding="utf-8")
    cmd = [sys.executable, str(REPO / "scripts" / "make_splits.py"), "--data-root", str(tmp_path), "--splits-dir", str(tmp_path / "splits")]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and "proposal only" in r.stdout and not (tmp_path / "splits").exists()
    assert subprocess.run(cmd + ["--freeze"], capture_output=True, text=True, encoding="utf-8").returncode == 0
    r = subprocess.run(cmd + ["--freeze"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "already frozen" in (r.stdout + r.stderr)
    assert importlib.util.find_spec("pehleaap.data.splits") is not None
