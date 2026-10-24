import gc
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pehleaap.baselines.turn_models import WhisperMelOnnx, fetch  # noqa: E402
from pehleaap.models.smart_turn import PUBLISHED  # noqa: E402
from pehleaap.train.eot_data import DEV_CLEAN, DEV_PHONE, Augmenter, EotStore, EpochKeys  # noqa: E402
from pehleaap.train.train_eot import TrainConfig, load_config, lr_lambda, precision_for, train  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
AUGMENT = REPO / "configs" / "augment.yaml"


def _store(root: Path, n: int = 12) -> EotStore:
    rng = np.random.default_rng(0)
    metas = [{"id": f"clip-{k}", "language": "hin" if k % 3 == 0 else "eng", "endpoint_bool": k % 2 == 0, "synthetic": True, "dataset": "t"}
             for k in range(n)]
    clips = [0.1 * rng.standard_normal(int(16000 * (1.5 + k % 4))).astype(np.float32) for k in range(n)]
    store = EotStore(root)
    store.append(metas, clips, "shard-0")
    return store


def test_config_file_is_the_defaults():
    assert load_config(REPO / "configs" / "train_eot.yaml") == TrainConfig()
    assert load_config(None, ["micro_batch=8", "batch_size=16", "langs=[hin]"]).langs == ("hin",)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=100, micro_batch=64)


def test_store_round_trip_and_crash_repair(tmp_path):
    store = _store(tmp_path)
    again = EotStore(tmp_path)
    assert len(again.index) == 12 and again.done_shards() == {"shard-0"}
    x = again.clip(3)
    assert x.dtype == np.float32 and len(x) == int(16000 * 4.5)
    del again  # release its memory map (Windows won't truncate a mapped file)
    gc.collect()
    # Audio written by a crashed run without index rows is dropped by the next append.
    with open(tmp_path / "audio.int16", "ab") as f:
        f.write(b"\x01\x00" * 999)
    store.append([{"id": "late", "language": "eng", "endpoint_bool": True, "synthetic": False, "dataset": "t"}], [np.ones(16000, np.float32) * 0.5], "shard-1")
    late = EotStore(tmp_path)
    assert late.clip(12) == pytest.approx(np.full(16000, 0.5), abs=1e-4)
    assert (tmp_path / "audio.int16").stat().st_size == 2 * int(late.index.offset.iat[-1] + late.index.length.iat[-1])


def test_long_clips_keep_their_last_8_seconds(tmp_path):
    x = np.linspace(-0.5, 0.5, 16000 * 10).astype(np.float32)
    store = EotStore(tmp_path)
    store.append([{"id": "long", "language": "eng", "endpoint_bool": True, "synthetic": False, "dataset": "t"}], [x], "s")
    np.testing.assert_allclose(store.clip(0), x[-8 * 16000 :], atol=1e-4)


def test_sampler_resumes_exactly_and_repeats_hindi():
    rows, repeat = np.arange(10), np.array([2] * 3 + [1] * 7)
    full = list(EpochKeys(rows, repeat, epochs=3, batch_size=4, seed=1))
    assert len(full) == 3 * 12  # 13 keys per epoch, cut to whole batches of 4
    assert sum(1 for e, r in full if e == 0 and r < 3) >= 3  # repeated rows show up more often
    for start in (0, 5, 12, 30):
        assert list(EpochKeys(rows, repeat, epochs=3, batch_size=4, seed=1, start=start)) == full[start:]


def test_clean_features_match_the_eval_wrapper_and_draws_are_reproducible(tmp_path):
    store = _store(tmp_path)
    aug = Augmenter(store, AUGMENT, p_phone=0.5, seed=0)
    clean = aug([(DEV_CLEAN, r) for r in range(4)])
    mel = WhisperMelOnnx(fetch(*PUBLISHED))
    for r in range(4):
        np.testing.assert_allclose(clean["features"][r].numpy(), mel.prepare(store.clip(r)), atol=1e-5)
    assert not clean["phone"].any() and aug([(DEV_PHONE, 0)])["phone"].all()
    keys = [(0, r) for r in range(12)]
    a, b = aug(keys), aug(keys)
    assert torch.equal(a["features"], b["features"]) and torch.equal(a["phone"], b["phone"])
    assert 0 < int(a["phone"].sum()) < 12
    assert a["labels"].tolist() == [float(r % 2 == 0) for r in range(12)]


def test_precision_and_schedule():
    assert precision_for(torch.device("cpu")) == "fp32" and precision_for(torch.device("cpu"), "bf16") == "bf16"
    f = lr_lambda(10, 100)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(100) == pytest.approx(0.0, abs=1e-9)


def test_tiny_run_trains_resumes_and_exports(tmp_path):
    store = _store(tmp_path / "data", n=40)
    assert len(store.rows("train", ("eng", "hin"))) >= 8
    cfg = TrainConfig(data_dir=str(tmp_path / "data"), epochs=1, batch_size=4, micro_batch=2, num_workers=0, eval_every_steps=2,
                      dev_limit=None, checkpoint_minutes=1e9)
    out = tmp_path / "run"
    summary = train(cfg, out, max_steps=2, device="cpu")
    assert summary["steps"] == 2 and (out / "model.onnx").exists() and (out / "ckpt.pt").exists()
    assert train(cfg, out, max_steps=2, device="cpu")["steps"] == 2  # resumes at the end: nothing more to do
    feats = np.random.default_rng(0).uniform(-1, 1, (2, 80, 800)).astype(np.float32)
    from pehleaap.models.smart_turn import SmartTurnV3

    model = SmartTurnV3()
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["model"])
    with torch.no_grad():
        p_torch = torch.sigmoid(model.eval()(torch.from_numpy(feats))).numpy()
    np.testing.assert_allclose(WhisperMelOnnx(out / "model.onnx").predict_prepared(feats), p_torch, atol=1e-5)
