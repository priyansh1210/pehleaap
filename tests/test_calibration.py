import numpy as np
import pandas as pd
import pytest

from pehleaap.eval.calibration import best_threshold, calibrate, cross_fit_thresholds, deployment_thresholds, with_channel_and_fold


def test_best_threshold_separates_and_picks_the_middle_of_ties():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_threshold(labels, probs)
    assert 0.3 < t < 0.7 and t == pytest.approx(0.5)
    assert best_threshold(np.array([1, 1]), np.array([0.2, 0.4])) < 0.2  # all finished: below the lowest score
    assert best_threshold(np.array([0, 0]), np.array([0.2, 0.4])) == 0.4  # all unfinished: nothing scores above
    # Tied scores can't be split: both clips at 0.5 fall on the same side.
    t = best_threshold(np.array([0, 1, 0]), np.array([0.5, 0.5, 0.2]))
    assert (np.array([0.5, 0.5, 0.2]) > t).tolist() in ([True, True, False], [False, False, False])


def _shifted(n=2000, shift=0.3, seed=0):
    """A phone condition whose scores are shifted up: perfect ranking, wrong threshold for 0.5."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    p_clean = np.clip(np.where(label == 1, 0.75, 0.25) + rng.normal(0, 0.1, n), 0, 1)
    rows = [{"id": f"c{i}", "language": "hin", "label": int(label[i]), "condition": cond, "p_m": float(min(p + s, 1.0))}
            for i, p in enumerate(p_clean) for cond, s in (("clean", 0.0), ("amrnb_4.75", shift))]
    return pd.DataFrame(rows)


def test_recalibration_recovers_a_pure_score_shift_without_tuning_on_itself():
    df = _shifted()
    res = calibrate(df, "m", 0.5).set_index(["condition", "variant"])
    card = res.loc[("amrnb_4.75", "card"), "accuracy"]
    tuned = res.loc[("amrnb_4.75", "channel+language"), "accuracy"]
    clean = res.loc[("clean", "card"), "accuracy"]
    assert card < clean - 15 and tuned > clean - 2  # the shift costs a lot at 0.5 and almost nothing once re-thresholded
    assert res.loc[("amrnb_4.75", "channel+language"), "cut_in"] < res.loc[("amrnb_4.75", "card"), "cut_in"]
    th = deployment_thresholds(df, "m")
    assert th["phone/hin"] == pytest.approx(0.8, abs=0.05) and th["clean/hin"] == pytest.approx(0.5, abs=0.05)


def test_shipped_thresholds_apply_per_channel_and_fall_back_to_the_card():
    df = _shifted()
    res = calibrate(df, "m", 0.5, shipped={"phone/hin": 0.8}).set_index(["condition", "variant"])
    assert res.loc[("amrnb_4.75", "shipped"), "threshold_median"] == 0.8
    assert res.loc[("clean", "shipped"), "threshold_median"] == 0.5  # no clean/hin entry: the card threshold
    assert res.loc[("amrnb_4.75", "shipped"), "accuracy"] > res.loc[("amrnb_4.75", "card"), "accuracy"] + 15
    assert "shipped" not in set(calibrate(df, "m", 0.5).variant)


def test_cross_fitting_never_uses_a_clips_own_fold():
    df = with_channel_and_fold(_shifted(n=200))
    t = cross_fit_thresholds(df, "p_m", ["condition"])
    for cond in ("clean", "amrnb_4.75"):
        for f in (0, 1):
            other = df[(df.condition == cond) & (df.fold != f)]
            mine = (df.condition == cond).to_numpy() & (df.fold == f).to_numpy()
            assert np.all(t[mine] == best_threshold(other.label, other.p_m))
