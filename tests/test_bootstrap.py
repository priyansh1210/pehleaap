import numpy as np

from pehleaap.eval.bootstrap import cluster_bootstrap


def test_cluster_bootstrap_resamples_whole_clusters():
    # Two speakers, every event of a speaker has the same value: a draw's mean can only be 0, 0.5 or 1.
    values = np.array([0.0] * 50 + [1.0] * 50)
    clusters = np.array(["a"] * 50 + ["b"] * 50)
    seen = set()
    res = cluster_bootstrap(lambda idx: seen.add(round(values[idx].mean(), 6)) or values[idx].mean(), clusters, n_boot=200)
    assert seen <= {0.0, 0.5, 1.0} and res["estimate"] == 0.5 and res["n_clusters"] == 2
    assert res["ci_low"] == 0.0 and res["ci_high"] == 1.0  # an event-level bootstrap would give about 0.4-0.6


def test_cluster_bootstrap_is_reproducible_and_counts_sign():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(20), 10)
    diff = rng.normal(0.3, 0.1, 20)[clusters]  # a clear positive per-speaker difference
    a = cluster_bootstrap(lambda idx: diff[idx].mean(), clusters, n_boot=500, seed=3)
    b = cluster_bootstrap(lambda idx: diff[idx].mean(), clusters, n_boot=500, seed=3)
    assert a == b and a["ci_low"] > 0 and a["p_above_zero"] == 1.0
