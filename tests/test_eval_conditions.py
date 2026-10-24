import importlib.util
from pathlib import Path

import numpy as np

from pehleaap.telephony.codecs import narrowband_many
from pehleaap.telephony.noise import NoiseBank
from pehleaap.telephony.pipeline import AugmentConfig

REPO = Path(__file__).resolve().parent.parent


def _eval_script():
    spec = importlib.util.spec_from_file_location("eval_clip_baselines", REPO / "scripts" / "eval_clip_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_chain_condition_runs_and_keeps_lengths():
    ev = _eval_script()
    rng = np.random.default_rng(0)
    clips = [0.1 * rng.standard_normal(n).astype(np.float32) for n in (8000, 12000)]
    metas = [{"id": "a"}, {"id": "b"}]
    bank = NoiseBank(root=None, sr=16000)
    for cond in ev.CHAIN:
        out = ev.phone_line(clips, metas, cond, bank, AugmentConfig())
        assert [len(y) for y in out] == [len(x) for x in clips], cond
        assert ev.describe(cond) == ev.CHAIN[cond][0]


def test_codec_conditions_are_unchanged_and_noise_is_paired_across_snrs():
    ev = _eval_script()
    clips = [0.1 * np.random.default_rng(1).standard_normal(8000).astype(np.float32)]
    same = ev.phone_line(clips, [{"id": "a"}], "g711_mulaw", None, AugmentConfig())
    np.testing.assert_array_equal(same[0], narrowband_many(clips, "g711_mulaw")[0])
    names = NoiseBank(root=None, sr=16000).names
    p5, p20 = (ev.chain_params({"id": "clip-7"}, c, names, AugmentConfig()) for c in ("noise5_amr7.95", "noise20_amr7.95"))
    assert (p5.seed, p5.noise) == (p20.seed, p20.noise) and (p5.snr_db, p20.snr_db) == (5.0, 20.0)
