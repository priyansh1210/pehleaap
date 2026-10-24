import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pehleaap.baselines.turn_models import WhisperMelOnnx, fetch  # noqa: E402
from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.models.smart_turn import PUBLISHED, SmartTurnV3, from_published, onnx_with_state_dict, state_dict_from_onnx  # noqa: E402


@pytest.fixture(scope="module")
def onnx_path():
    return fetch(*PUBLISHED)


def _features(onnx_path, n=8):
    """Real test clips when the test set is on disk, random log-mel otherwise."""
    shards = smart_turn_testset.shard_paths()
    if not shards:
        return np.random.default_rng(0).uniform(-1, 1, (n, 80, 800)).astype(np.float32)
    _, clips = smart_turn_testset.read_shard(shards[0], ["hin"])
    mel = WhisperMelOnnx(onnx_path)
    return np.stack([mel.prepare(c) for c in clips[:n]])


def test_every_weight_is_mapped(onnx_path):
    state = state_dict_from_onnx(onnx_path)
    assert set(state) == set(SmartTurnV3().state_dict())


def test_pytorch_model_matches_the_published_onnx(onnx_path):
    feats = _features(onnx_path)
    p_onnx = WhisperMelOnnx(onnx_path).predict_prepared(feats)
    with torch.no_grad():
        p_torch = torch.sigmoid(from_published(onnx_path)(torch.from_numpy(feats))).numpy()
    np.testing.assert_allclose(p_torch, p_onnx, atol=1e-4)


def test_new_weights_go_into_the_published_graph(onnx_path, tmp_path):
    torch.manual_seed(0)
    model = SmartTurnV3().eval()  # random weights, unlike the published ones
    out = onnx_with_state_dict(onnx_path, model.state_dict(), tmp_path / "graft.onnx")
    back = state_dict_from_onnx(out)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(back[name], value)
    feats = _features(onnx_path, n=4)
    with torch.no_grad():
        p_torch = torch.sigmoid(model(torch.from_numpy(feats))).numpy()
    np.testing.assert_allclose(WhisperMelOnnx(out).predict_prepared(feats), p_torch, atol=1e-4)
