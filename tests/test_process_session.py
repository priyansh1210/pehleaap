import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.audio.io import resample
from pehleaap.audio.vad import SileroVAD
from pehleaap.labeling.config import LabelConfig

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def _script():
    spec = importlib.util.spec_from_file_location("process_session", REPO / "scripts" / "process_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(seed: int, offset_s: float = 0.0, drift_ppm: float = 0.0) -> np.ndarray:
    """50 s: a sync beep near each end (shared by both tracks), noise bursts as speech, a quiet floor."""
    rng = np.random.default_rng(seed)
    x = 0.001 * rng.standard_normal(SR * 50).astype(np.float32)
    t = np.arange(int(0.05 * SR)) / SR
    for at in (3.0, 46.0):
        i = int((at + offset_s) * SR)
        x[i : i + len(t)] += 0.6 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    for a in np.arange(6.0 + seed, 40.0, 7.0):
        i = int(a * SR)
        x[i : i + SR] += (0.1 * rng.standard_normal(SR) * np.hanning(SR)).astype(np.float32)
    return resample(x, SR, SR * (1 + drift_ppm * 1e-6))[: SR * 50] if drift_ppm else x


def _session(root: Path, name: str, devices=("earphones", "handset"), consent=("2026-10-20", "2026-10-21"), drift_ppm=0.0) -> Path:
    d = root / "raw" / "sessions" / name
    d.mkdir(parents=True)
    sf.write(d / f"{name}_SPK1_agent.wav", _track(1), SR)
    sf.write(d / f"{name}_SPK2_customer.wav", _track(2, offset_s=0.03, drift_ppm=drift_ppm), SR)
    with open(root / "raw" / "sessions.csv", "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("session_id,date,tool,spk_agent,spk_customer,device_agent,device_customer\n")
        f.write(f"{name},2026-10-25,craig,SPK1,SPK2,{devices[0]},{devices[1]}\n")
    (root / "raw" / "speakers.csv").write_text(
        f"speaker_id,consent_date\nSPK1,{consent[0]}\nSPK2,{consent[1]}\n", encoding="utf-8")
    return d


def test_good_session_is_checked_labelled_and_checksummed(tmp_path):
    ps = _script()
    d = _session(tmp_path, "S050")
    r = ps.process(d, tmp_path, LabelConfig(), SileroVAD(), None, None, force=False)
    assert r["ok"] and not r["problems"], r
    assert abs(r["clock"]["drift_ms"]) < 2 and r["clock"]["offset_start_ms"] == pytest.approx(30, abs=3)
    assert r["agent"]["sample_rate"] == SR and r["agent"]["clipped_pct"] == 0
    manifest = (tmp_path / "manifests" / "sessions" / "S050.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["path"] for line in manifest] == ["raw/sessions/S050/S050_SPK1_agent.wav", "raw/sessions/S050/S050_SPK2_customer.wav"]
    assert json.loads((tmp_path / "labels" / "S050" / "process.json").read_text(encoding="utf-8"))["ok"]


def test_no_consent_means_nothing_is_processed(tmp_path):
    ps = _script()
    d = _session(tmp_path, "S051", consent=("2026-10-20", ""))
    r = ps.process(d, tmp_path, LabelConfig(), SileroVAD(), None, None, force=True)  # not even with --force
    assert not r["ok"] and any("consent" in p for p in r["problems"])
    assert "agent" not in r and not (tmp_path / "labels" / "S051").exists()


def test_clock_drift_stops_labelling_unless_forced(tmp_path):
    ps = _script()
    d = _session(tmp_path, "S052", drift_ppm=800)  # ~34 ms over the 43 s between the beeps
    r = ps.process(d, tmp_path, LabelConfig(), SileroVAD(), None, None, force=False)
    assert any("drift" in p for p in r["problems"]) and "labels" not in r
    forced = ps.process(d, tmp_path, LabelConfig(), SileroVAD(), None, None, force=True)
    assert "labels" in forced and not forced["ok"]


def test_pstn_sessions_skip_the_clock_check(tmp_path):
    ps = _script()
    d = _session(tmp_path, "S053", devices=("pstn", "pstn"), drift_ppm=800)
    r = ps.process(d, tmp_path, LabelConfig(), SileroVAD(), None, None, force=False)
    assert "skipped" in r["clock"] and r["ok"]
