import numpy as np

from pehleaap.baselines.turn_models import WaveformOnnx, last_n_seconds


def test_short_clips_are_left_padded():
    x = np.ones(16000, dtype=np.float32)
    y = last_n_seconds(x)
    assert len(y) == 8 * 16000
    assert y[: 7 * 16000].sum() == 0  # silence first
    assert y[-16000:].sum() == 16000  # speech at the end


def test_long_clips_keep_the_last_8_seconds():
    x = np.arange(10 * 16000, dtype=np.float32)
    y = last_n_seconds(x)
    assert len(y) == 8 * 16000
    assert y[-1] == x[-1]
    assert y[0] == x[2 * 16000]


def test_waveform_standardisation_uses_real_samples_only():
    # Chhotu's contract: standardise over the real samples, then left-pad with zeros.
    model = WaveformOnnx.__new__(WaveformOnnx)
    model.standardize = True
    x = np.linspace(-1, 3, 16000, dtype=np.float32)
    y = model.prepare(x)
    assert np.all(y[: -16000] == 0)
    assert abs(y[-16000:].mean()) < 1e-5 and abs(y[-16000:].std() - 1) < 1e-3


def test_load_any_finds_registry_and_local_models(tmp_path, monkeypatch):
    import shutil

    import pytest

    from pehleaap.baselines import turn_models

    published = turn_models.fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx")
    (tmp_path / "ours").mkdir()
    shutil.copy(published, tmp_path / "ours" / "model.onnx")
    monkeypatch.setattr(turn_models, "WEIGHTS_DIR", tmp_path)
    model, threshold = turn_models.load_any("ours")
    assert threshold == 0.5 and 0.0 <= model.predict(np.zeros(16000, dtype=np.float32)) <= 1.0
    assert turn_models.load_any("deveshu_fp32")[1] == turn_models.REGISTRY["deveshu_fp32"].threshold
    with pytest.raises(FileNotFoundError):
        turn_models.load_any("missing")


def test_quantized_models_are_told_apart():
    from pehleaap.baselines.turn_models import fetch, is_quantized

    assert is_quantized(fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"))  # int8
    assert not is_quantized(fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"))  # fp32


def test_cuda_device_matches_cpu_or_says_why_not():
    import onnxruntime as ort
    import pytest

    from pehleaap.baselines.turn_models import WhisperMelOnnx, fetch

    fp32 = fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx")
    with pytest.raises(ValueError):
        WhisperMelOnnx(fp32, device="gpu")
    if "CUDAExecutionProvider" not in ort.get_available_providers():  # the CPU-only research venv
        with pytest.raises(RuntimeError, match="no CUDA provider"):
            WhisperMelOnnx(fp32, device="cuda")
        return
    feats = np.random.default_rng(0).uniform(-1, 1, (4, 80, 800)).astype(np.float32)
    gpu = WhisperMelOnnx(fp32, device="cuda")
    assert gpu.session.get_providers()[0] == "CUDAExecutionProvider"
    np.testing.assert_allclose(gpu.predict_prepared(feats), WhisperMelOnnx(fp32).predict_prepared(feats), atol=1e-4)
    int8 = WhisperMelOnnx(fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"), device="cuda")
    assert int8.session.get_providers() == ["CPUExecutionProvider"]
