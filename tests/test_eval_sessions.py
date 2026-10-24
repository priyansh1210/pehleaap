import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from pehleaap.eval.sessions import CHECKPOINTS_S, checkpoint_probs, checkpoint_windows, condition_track
from pehleaap.labeling.schema import HOLD, SHIFT, Event, write_events

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def test_windows_end_at_the_checkpoint_and_stop_when_they_speak_again():
    x = np.arange(SR * 20, dtype=np.float32)
    events = [Event("customer", HOLD, 10.0, pause_s=0.6), Event("customer", SHIFT, 3.0, pause_s=5.0)]
    found = list(checkpoint_windows(x, SR, events))
    hold = [(j, w) for i, j, w in found if i == 0]
    assert [CHECKPOINTS_S[j] for j, _ in hold] == [0.2, 0.3, 0.5]  # 0.8 s and later: they've resumed
    for j, w in hold:
        assert w[-1] == x[int(round((10.0 + CHECKPOINTS_S[j]) * SR)) - 1] and len(w) == 8 * SR
    shift_first = next(w for i, j, w in found if i == 1 and j == 0)
    assert len(shift_first) == int(3.2 * SR)  # only 3.2 s of audio exist before that checkpoint


class LengthModel:
    """P(complete) = seconds of audio / 10: lets the test see which window each probability came from."""

    def predict_batch(self, clips):
        return np.array([len(c) / SR / 10 for c in clips])


def test_probs_fill_valid_checkpoints_and_leave_the_rest_nan():
    events = [Event("customer", HOLD, 10.0, pause_s=0.6), Event("customer", SHIFT, 3.0, pause_s=5.0)]
    p = checkpoint_probs(LengthModel(), np.zeros(SR * 20, dtype=np.float32), SR, events)
    assert p.shape == (2, len(CHECKPOINTS_S))
    assert np.isnan(p[0, 3:]).all() and np.allclose(p[0, :3], 0.8)
    assert np.allclose(p[1], [(3.0 + c) / 10 for c in CHECKPOINTS_S])


def test_condition_track_keeps_length():
    x = (0.1 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32)
    assert condition_track(x, SR, "clean", "t") is x
    assert len(condition_track(x, SR, "g711_mulaw", "t")) == len(x)


def test_script_end_to_end(tmp_path):
    labels = tmp_path / "labels" / "S9"
    labels.mkdir(parents=True)
    rng = np.random.default_rng(1)
    tracks = {}
    for role in ("agent", "customer"):
        tracks[role] = str(tmp_path / f"S9_x_{role}.wav")
        sf.write(tracks[role], (0.05 * rng.standard_normal(SR * 30)).astype(np.float32), SR)
    write_events(labels / "events.jsonl", [Event("customer", SHIFT if k % 2 else HOLD, 3.0 + 4 * k, session="S9", pause_s=1.0 + k % 3)
                                           for k in range(6)])
    (labels / "summary.json").write_text(json.dumps({"tracks": tracks}), encoding="utf-8")
    out = "sessions_test_" + tmp_path.name
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "eval_sessions.py"), str(labels), "--models", "smart_turn_v3.2_int8",
                        "--conditions", "clean", "--threads", "1", "--out", out], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    run = REPO / "runs" / out
    try:
        probs = pd.read_csv(run / "probs.csv")
        assert len(probs) == 6 and set(probs.kind) == {SHIFT, HOLD}
        report = (run / "report.md").read_text(encoding="utf-8")
        assert "UNVERIFIED" in report and "smart_turn_v3.2_int8" in report and "silence timeout" in report
    finally:
        for f in run.glob("*"):
            f.unlink()
        run.rmdir()
