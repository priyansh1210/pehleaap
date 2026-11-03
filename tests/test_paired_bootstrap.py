import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from paired_bootstrap import matched_cut_in  # noqa: E402


def test_matched_cut_in_separates_a_perfect_model_from_an_eager_one():
    kinds = np.array(["shift"] * 4 + ["hold"] * 4)
    durs = np.full(8, 1.0)  # every pause lasts 1 s, so checkpoints 0.2-0.8 s count
    perfect = np.where(kinds[:, None] == "shift", 0.99, 0.01) * np.ones((8, 6))
    eager = np.full((8, 6), 0.99)
    assert matched_cut_in(kinds, durs, perfect, 0.2, 3.0) == 0.0
    assert matched_cut_in(kinds, durs, eager, 0.2, 3.0) == 1.0
