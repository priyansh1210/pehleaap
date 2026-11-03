import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np  # noqa: E402

from make_indicvoices_eot import page, pick_segments, with_tail  # noqa: E402


def test_pick_segments_one_per_speaker_in_range():
    meta = [{"path": f"s{s}_chunk_{k}.flac", "speaker_id": f"S{s}", "scenario": sc, "duration": d}
            for s in range(5) for k, (sc, d) in enumerate([("Extempore", 5.0), ("Conversation", 1.0), ("Read", 6.0), ("Conversation", 30.0)])]
    picked = pick_segments(meta, n=3)
    assert len(picked) == 3 and len({m["speaker_id"] for m in picked}) == 3
    # Only extempore/conversation segments of 2-25 s qualify: here just chunk_0.
    assert all(m["path"].endswith("_chunk_0.flac") for m in picked)
    assert pick_segments(meta, n=3) == picked  # fixed choice


def test_with_tail_appends_the_recordings_own_pause():
    from types import SimpleNamespace as Ipu

    sr = 16000
    speech = np.sin(np.arange(sr) * 0.3).astype(np.float32)
    quiet = np.full(sr // 2, 0.001, dtype=np.float32)
    x = np.concatenate([speech, quiet, speech])  # speech 0-1 s, pause 1-1.5 s, speech 1.5-2.5 s, trimmed at the last word
    ipus = [Ipu(start=0.0, end=1.0), Ipu(start=1.5, end=2.5)]
    out, padded = with_tail(x, x, ipus)
    assert len(out) == int(2.7 * sr) and padded == 0.2
    assert np.allclose(out[-int(0.19 * sr):], 0.001)  # the filler is the recording's own pause
    assert np.array_equal(out[: len(x) - 80], x[: len(x) - 80])  # speech untouched up to the 5 ms crossfade
    already = np.concatenate([x, quiet[: int(0.2 * sr)]])
    out, padded = with_tail(already, already, ipus)
    assert out is already and padded == 0.0


def test_page_lists_every_clip_once_without_its_kind():
    clips = [{"id": f"iv_hi_{i:011x}", "kind": "end" if i % 2 else "pause@1.20"} for i in range(6)]
    html = page(clips)
    items = json.loads(html.split("const items=")[1].split(";let labels")[0])
    assert sorted(x["id"] for x in items) == sorted(c["id"] for c in clips)
    assert all(x["src"] == f"clips/{x['id']}.flac" for x in items)
    assert "pause" not in html and "end\"" not in html
