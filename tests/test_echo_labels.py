import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.audio.vad import hop_s
from pehleaap.labeling.config import LabelConfig
from pehleaap.labeling.echo import split_echo
from pehleaap.labeling.ipu import IPU
from pehleaap.labeling.schema import INTERRUPT, NOISE_ECHO, SHIFT, read_events
from pehleaap.telephony.echo import speakerphone_echo

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def syllables(seconds: float, seed: int) -> np.ndarray:
    """Speech stand-in: noise bursts of random length, loudness and spacing (independent for every seed)."""
    rng = np.random.default_rng(seed)
    x, t = np.zeros(int(seconds * SR), np.float32), 0.0
    while t < seconds - 0.3:
        d = rng.uniform(0.08, 0.25)
        a, b = int(t * SR), int((t + d) * SR)
        x[a:b] = rng.uniform(0.05, 0.3) * rng.standard_normal(b - a) * np.hanning(b - a)
        t += d + rng.uniform(0.03, 0.15)
    return x


def call(erle_db: float = 3.0):
    """Agent talks 1-6 s and 12-17 s. The customer's track has the agent's echo (0.3 s later) and their own
    speech at 6.6-9 s, right after the echo of the agent's first turn, and at 18-20 s."""
    n = SR * 22
    agent, customer = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for k, (a, b) in enumerate(((1.0, 6.0), (12.0, 17.0))):
        agent[int(a * SR) : int(b * SR)] = syllables(b - a, seed=10 + k)
    for k, (a, b) in enumerate(((6.6, 9.0), (18.0, 20.0))):
        customer[int(a * SR) : int(b * SR)] = syllables(b - a, seed=20 + k)
    echo = speakerphone_echo(agent, SR, -26.0 + 10 * np.log10(np.mean(customer[customer != 0] ** 2) / 10 ** -2.6), 0.3, erle_db, 1.5, 0.3, 8.0,
                             np.random.default_rng(0))
    noise = 0.0005 * np.random.default_rng(1).standard_normal(n).astype(np.float32)
    return agent + noise, customer + echo + noise


AGENT_IPUS = [IPU(1.0, 6.0), IPU(12.0, 17.0)]
CUSTOMER_IPUS = [IPU(1.3, 9.0), IPU(12.3, 17.4), IPU(18.0, 20.0)]  # VAD merges the first echo with the reply


def test_echo_is_cut_out_and_the_reply_keeps_its_start():
    agent, customer = call()
    speech, echo, lag = split_echo(customer, agent, SR, CUSTOMER_IPUS, AGENT_IPUS, LabelConfig())
    assert lag == pytest.approx(0.3, abs=0.02)
    starts = [round(u.start, 2) for u in speech if u.dur > 1.0]
    assert starts == [pytest.approx(6.6, abs=0.15), pytest.approx(18.0, abs=0.15)]
    covered = sum(u.dur for u in echo)
    assert covered > 8.0 and all(u.echo_lag_s == lag and u.echo_corr > 0.5 for u in echo)
    assert all(u.end <= 6.6 + 0.15 or u.start >= 12.0 for u in echo)  # no echo segment inside the reply


def test_no_echo_leaves_the_ipus_alone():
    agent = np.zeros(SR * 22, np.float32)
    agent[SR : 6 * SR] = syllables(5.0, seed=10)
    customer = np.zeros(SR * 22, np.float32)
    customer[int(6.6 * SR) : 9 * SR] = syllables(2.4, seed=20)
    ipus = [IPU(6.6, 9.0)]
    speech, echo, lag = split_echo(customer, agent, SR, ipus, [IPU(1.0, 6.0)], LabelConfig())
    assert lag is None and echo == [] and speech is ipus


class FakeVAD:
    def probs(self, batch, sr):
        hop = hop_s(sr)
        p = np.zeros((2, int(batch.shape[1] / sr / hop)))
        for row, ipus in enumerate((AGENT_IPUS, CUSTOMER_IPUS)):
            for u in ipus:
                p[row, int(u.start / hop) : int(u.end / hop)] = 0.9
        return p


def test_label_session_turns_echo_into_noise_echo_not_interruptions(tmp_path):
    spec = importlib.util.spec_from_file_location("label_session", REPO / "scripts" / "label_session.py")
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    agent, customer = call()
    session = tmp_path / "raw" / "S800"
    session.mkdir(parents=True)
    sf.write(session / "S800_spk1_agent.wav", agent, SR)
    sf.write(session / "S800_spk1_customer.wav", np.clip(customer, -1, 1), SR)
    s = ls.label_session(session, tmp_path / "labels", LabelConfig(), FakeVAD())
    events = read_events(tmp_path / "labels" / "S800" / "events.jsonl")
    customer_events = [e for e in events if e.channel == "customer"]
    assert s["echo_delay_s"]["customer"] == pytest.approx(0.3, abs=0.02) and s["echo_delay_s"]["agent"] is None
    assert any(e.type == NOISE_ECHO for e in customer_events) and not any(e.type == INTERRUPT for e in customer_events)
    agent_shift = [e for e in events if e.channel == "agent" and e.type == SHIFT and abs(e.t_event - 6.0) < 0.1]
    assert agent_shift and agent_shift[0].gap_s == pytest.approx(0.6, abs=0.15)  # the reply's timing survives the echo
