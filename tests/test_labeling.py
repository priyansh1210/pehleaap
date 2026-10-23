from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from pehleaap.labeling import lexicons
from pehleaap.labeling.config import LabelConfig, load_config
from pehleaap.labeling.events import extract_events, is_backchannel
from pehleaap.labeling.ipu import IPU, attach_words, ipus_from_probs, merge_segments
from pehleaap.labeling.schema import BC, HOLD, HOLD_REQ, INTERRUPT, SHIFT, read_events, to_pause_events, write_events

REPO = Path(__file__).resolve().parent.parent


def test_config_file_matches_defaults():
    assert load_config(REPO / "configs" / "labeling.yaml") == LabelConfig()


def test_short_gaps_merge_and_blips_drop():
    segs = np.array([[0.0, 1.0], [1.15, 2.0], [3.0, 3.05], [4.0, 5.0]])
    ipus = merge_segments(segs, merge_gap_s=0.2, min_dur_s=0.1)
    assert [(u.start, u.end) for u in ipus] == [(0.0, 2.0), (4.0, 5.0)]


def test_ipus_from_probs():
    hop = 0.032
    p = np.zeros(100)
    p[10:40] = 0.9
    p[43:60] = 0.9  # 3-frame (96 ms) gap: merged
    p[80:82] = 0.9  # 64 ms blip: dropped
    ipus = ipus_from_probs(p, hop, LabelConfig())
    assert len(ipus) == 1
    assert ipus[0].start == pytest.approx(10 * hop) and ipus[0].end == pytest.approx(60 * hop)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("haan ji", "continuer"),
        ("Hmmmm.", "continuer"),
        ("हाँ जी", "continuer"),
        ("achha theek hai", "assessment"),
        ("ठीक है", "assessment"),
        ("hai", None),
        ("mera account number", None),
        ("", None),
        (None, None),
    ],
)
def test_bc_lexicon(text, expected):
    assert lexicons.bc_type(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ek minute, card dhoondh rahi hoon", True),
        ("ruko zara", True),
        ("रुकिए, एक मिनट", True),
        ("hold on please", True),
        ("main check karke batata hoon", True),
        ("ek minute ka kaam hai", True),  # a false positive the 3 s silence rule filters out
        ("rukawat ke liye khed hai", False),
        ("mera minute plan", False),
        (None, False),
    ],
)
def test_hold_phrases(text, expected):
    assert lexicons.is_hold_request(text) is expected


@pytest.mark.parametrize(
    "text, run",
    [
        ("nau aath saat chhe", 4),
        ("mera number hai 98765 43210", 10),
        ("double nine eight seven", 4),
        ("नौ आठ सात", 3),
        ("ek minute do", 1),
        ("do teen din mein", 2),
    ],
)
def test_digit_runs(text, run):
    assert lexicons.digit_run(text) == run


def test_attach_words_by_midpoint():
    ipus = [IPU(0.0, 1.0), IPU(2.0, 3.0)]
    words = [{"start": 0.1, "end": 0.4, "word": " haan"}, {"start": 0.5, "end": 1.2, "word": " ji"}]
    attach_words(ipus, words)
    assert (ipus[0].text, ipus[0].n_words) == ("haan ji", 2)  # "ji" ends at 1.2, but its midpoint (0.85) is inside
    assert (ipus[1].text, ipus[1].n_words) == ("", 0)  # transcribed, but nothing heard


def _conversation():
    """A scripted 25 s session; every rule fires at a known time (see expected list below)."""
    customer = [
        IPU(0.0, 2.0),
        IPU(3.0, 3.4),  # "hmm" while the agent talks: BC
        IPU(7.5, 9.5),  # cuts into the agent at 7.5; the agent stops at 9.0: yielded INTERRUPT
        IPU(10.5, 12.0),
        IPU(13.0, 14.5),  # talks over the agent, who carries on: competitive INTERRUPT
        IPU(17.0, 18.0, "ek minute, card dhoondh rahi hoon", 6),
        IPU(22.0, 24.0),
    ]
    agent = [
        IPU(2.5, 6.0),
        IPU(6.6, 9.0),
        IPU(9.8, 10.2),  # BC during the customer's pause
        IPU(11.85, 16.0),  # starts 150 ms before the customer finishes: early start, not an interruption
    ]
    return {"customer": customer, "agent": agent}


def test_scripted_conversation_events():
    events = extract_events(_conversation(), end_s=25.0)
    got = [(e.t_event, e.channel, e.type) for e in events]
    assert got == [
        (2.0, "customer", SHIFT),
        (3.0, "customer", BC),
        (6.0, "agent", HOLD),
        (7.5, "customer", INTERRUPT),
        (9.5, "customer", HOLD),  # the agent's BC at 9.8 doesn't take the turn
        (9.8, "agent", BC),
        (12.0, "customer", SHIFT),
        (13.0, "customer", INTERRUPT),
        (16.0, "agent", SHIFT),
        (18.0, "customer", HOLD),
        (18.0, "customer", HOLD_REQ),
    ]
    # Pauses that end under the other speaker's talk (agent at 9.0, customer at 14.5) aren't clean pauses,
    # and the customer's last IPU has nobody speaking after it.
    by_key = {(e.t_event, e.type): e for e in events}
    shift = by_key[(2.0, SHIFT)]
    assert shift.gap_s == pytest.approx(0.5)
    assert shift.pause_s == pytest.approx(5.5)  # the customer's own "hmm" at 3.0 doesn't end the pause
    assert by_key[(12.0, SHIFT)].gap_s == pytest.approx(-0.15)
    assert by_key[(9.5, HOLD)].pause_s == pytest.approx(1.0)
    yielded = by_key[(7.5, INTERRUPT)]
    assert yielded.yield_s == pytest.approx(1.5) and yielded.overlap_s == pytest.approx(1.5)
    competitive = by_key[(13.0, INTERRUPT)]
    assert competitive.yield_s is None and competitive.overlap_s == pytest.approx(1.5)
    hold_req = by_key[(18.0, HOLD_REQ)]
    assert hold_req.pause_s == pytest.approx(4.0) and hold_req.digit_span is False
    assert by_key[(3.0, BC)].bc_type is None  # no transcript yet


def test_short_answer_is_not_a_backchannel_if_the_other_speaker_stops():
    other = [IPU(0.0, 2.0)]
    assert not is_backchannel(IPU(2.5, 2.9), other, LabelConfig())  # "haan" answering a question, then silence
    assert is_backchannel(IPU(2.5, 2.9), other + [IPU(3.5, 5.0)], LabelConfig())


def test_transcript_decides_backchannels():
    other = [IPU(0.0, 5.0)]
    cfg = LabelConfig()
    assert is_backchannel(IPU(2.0, 2.8, "haan ji", 2), other, cfg)  # 0.8 s: too long without text, fine with BC words
    assert not is_backchannel(IPU(2.0, 2.4, "kitna", 1), other, cfg)
    assert not is_backchannel(IPU(2.0, 2.8), other, cfg)


def test_hold_request_needs_the_silence():
    ipus = {"customer": [IPU(0.0, 1.0, "ek minute", 2), IPU(2.0, 3.0)], "agent": [IPU(4.0, 5.0)]}
    assert not any(e.type == HOLD_REQ for e in extract_events(ipus, end_s=6.0))


def test_jsonl_roundtrip_and_pause_events(tmp_path):
    events = extract_events(_conversation(), end_s=25.0)
    path = tmp_path / "events.jsonl"
    write_events(path, events)
    back = read_events(path)
    assert back == events
    assert {f.name for f in fields(back[0])} >= {"session", "speaker", "channel", "t_event", "type", "verified"}

    pauses = to_pause_events(events, "customer")
    assert [(p.kind, p.pause_dur_s) for p in pauses] == [
        (SHIFT, pytest.approx(5.5)),
        (HOLD, pytest.approx(1.0)),
        (SHIFT, pytest.approx(1.0)),
        (HOLD, pytest.approx(4.0)),
    ]
