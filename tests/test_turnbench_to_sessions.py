import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from label_session import find_tracks  # noqa: E402
from turnbench_to_sessions import write_session  # noqa: E402


def test_write_session_makes_a_folder_label_session_accepts(tmp_path):
    event = {"start_s": 0.5, "end_s": 2.0, "label": "Normal Turn", "text": "Hi."}
    row = {"conversation_id": "7", "metadata": {"speaker_1_actor_id": "abGOCERUSBhvd2Gy", "speaker_2_actor_id": "ejILcbef9"},
           "speaker_1_audio": {"bytes": b"one"}, "speaker_2_audio": {"bytes": b"two"},
           **{f"speaker_{s}_annotation_{a}": [event] if s == 1 else [] for s in (1, 2) for a in "abc"}}
    out = write_session(row, tmp_path)
    tracks = find_tracks(out)
    assert tracks["agent"].name == "TB7_abGOCERU_agent.flac" and tracks["agent"].read_bytes() == b"one"
    assert tracks["customer"].stem.split("_")[1] == "ejILcbef" and tracks["customer"].read_bytes() == b"two"
    labels = json.loads((out / "turnbench.json").read_text(encoding="utf-8"))
    assert labels["annotations"]["speaker_1"]["b"] == [event] and labels["roles"]["speaker_2"] == "customer"
