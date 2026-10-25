import numpy as np
import pytest

from pehleaap.eval.clip_metrics import binary_metrics, grouped_metrics
from pehleaap.eval.eot_events import (
    HOLD,
    SHIFT,
    PauseEvent,
    cut_in_at_gap,
    fire_times,
    operating_point,
    threshold_curve,
    timeout_curve,
)


def test_binary_metrics_match_smart_turn_definitions():
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.2, 0.7, 0.1, 0.4])  # 1 TP, 1 FN, 1 FP, 2 TN
    m = binary_metrics(labels, probs)
    assert m["accuracy"] == pytest.approx(60.0)
    assert m["fp_pct_total"] == pytest.approx(20.0)
    assert m["fn_pct_total"] == pytest.approx(20.0)
    assert m["accuracy"] + m["fp_pct_total"] + m["fn_pct_total"] == pytest.approx(100.0)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["fnr"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(100 * (0.5 * 2 / 3 + 0.5 * 0.5))
    # AUC: of the 6 positive/negative pairs, the positive scores higher in 4.
    assert m["auc"] == pytest.approx(4 / 6)


def test_grouped_metrics_splits_by_group():
    out = grouped_metrics([1, 0, 1, 0], [0.9, 0.1, 0.1, 0.9], ["hin", "hin", "eng", "eng"])
    assert out["hin"]["accuracy"] == 100.0
    assert out["eng"]["accuracy"] == 0.0


CHECKPOINTS = np.array([0.15, 0.3, 0.5, 0.8, 1.2])


def test_fire_time_is_first_confident_checkpoint_inside_the_pause():
    probs = np.array(
        [
            [0.1, 0.2, 0.9, 0.9, 0.9],  # fires at 0.5
            [0.1, 0.9, 0.9, 0.9, 0.9],  # would fire at 0.3, but the pause ends at 0.25
            [0.1, 0.1, 0.1, 0.1, 0.1],  # never confident: timeout at 2.0 (pause lasts 3.0)
        ]
    )
    t = fire_times(probs, CHECKPOINTS, np.array([2.0, 0.25, 3.0]), threshold=0.5, timeout_s=2.0)
    assert t[0] == pytest.approx(0.5)
    assert np.isnan(t[1])
    assert t[2] == pytest.approx(2.0)


def test_operating_point_counts_cut_ins_and_gaps():
    events = [
        PauseEvent("h1", HOLD, 0.6),  # customer resumes at 0.6 s
        PauseEvent("h2", HOLD, 0.6),
        PauseEvent("s1", SHIFT, 5.0),
        PauseEvent("s2", SHIFT, 5.0),
    ]
    probs = np.array(
        [
            [0.1, 0.9, 0.9, 0.9, 0.9],  # hold, fires at 0.3 -> cut-in
            [0.1, 0.1, 0.1, 0.9, 0.9],  # hold, would fire at 0.8 but the customer resumed: fine
            [0.9, 0.9, 0.9, 0.9, 0.9],  # shift, gap 0.15
            [0.1, 0.1, 0.9, 0.9, 0.9],  # shift, gap 0.5
        ]
    )
    p = operating_point(events, probs, CHECKPOINTS, threshold=0.5, timeout_s=None, tts_offset_s=0.1)
    assert p["cut_in_rate"] == pytest.approx(0.5)
    assert p["miss_rate"] == pytest.approx(0.0)
    assert p["median_gap_s"] == pytest.approx((0.25 + 0.6) / 2)


def test_curves_trade_cut_ins_for_latency():
    rng = np.random.default_rng(0)
    events = [PauseEvent(f"h{i}", HOLD, rng.uniform(0.2, 1.5)) for i in range(200)]
    events += [PauseEvent(f"s{i}", SHIFT, 10.0) for i in range(200)]

    silence = timeout_curve(events, [0.2, 0.5, 1.0, 2.0])
    rates = [p["cut_in_rate"] for p in silence]
    gaps = [p["median_gap_s"] for p in silence]
    assert rates == sorted(rates, reverse=True)  # longer timeout, fewer cut-ins
    assert gaps == sorted(gaps)  # ... but slower responses

    # A perfect model: confident only on shifts.
    probs = np.array([[0.0] * 5 if e.kind == HOLD else [1.0] * 5 for e in events])
    curve = threshold_curve(events, probs, CHECKPOINTS, [0.5], timeout_s=2.0)
    assert curve[0]["cut_in_rate"] == 0.0
    assert curve[0]["median_gap_s"] == pytest.approx(0.15)
    assert cut_in_at_gap(silence, 0.5) > cut_in_at_gap(curve, 0.5)


def test_cut_in_at_gap_uses_the_best_threshold_at_each_gap():
    # Two thresholds share the 0.2 s median gap; the one with fewer cut-ins is the fair point.
    curve = [
        {"median_gap_s": 0.2, "cut_in_rate": 0.9},
        {"median_gap_s": 0.2, "cut_in_rate": 0.7},
        {"median_gap_s": 1.0, "cut_in_rate": 0.3},
        {"median_gap_s": 3.0, "cut_in_rate": 0.4},  # slower and worse: never chosen
        {"median_gap_s": float("nan"), "cut_in_rate": 0.0},  # never fired on a turn end
    ]
    assert cut_in_at_gap(curve, 0.2) == pytest.approx(0.7)
    assert cut_in_at_gap(curve, 0.6) == pytest.approx(0.5)
    assert cut_in_at_gap(curve, 3.0) == pytest.approx(0.3)
    assert cut_in_at_gap(curve, 0.1) == pytest.approx(0.7)
