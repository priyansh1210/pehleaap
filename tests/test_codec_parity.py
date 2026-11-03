import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from codec_parity import compare  # noqa: E402


def test_compare_names_only_codecs_that_differ():
    d = np.array([10])
    sig = np.sin(np.linspace(0, 20, 400)).astype(np.float32)
    a = {"g711": sig, "g711__delay": d, "opus": sig, "opus__delay": d, "gsm": sig, "gsm__delay": d}
    b = {"g711": sig.copy(), "g711__delay": d, "opus": sig + 0.01, "opus__delay": d, "gsm": sig, "gsm__delay": np.array([12])}
    out = compare(a, b)
    assert out["g711"] == "identical"
    assert out["opus"].startswith("different: max |d| 0.01")
    assert out["gsm"] == "identical audio, delay 10 vs 12 samples"
