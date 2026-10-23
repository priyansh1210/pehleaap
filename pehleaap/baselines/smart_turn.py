"""Pipecat Smart Turn v3.x ONNX baseline.

Preprocessing matches pipecat-ai/smart-turn `inference.py` and `audio_utils.py` exactly:
keep the last 8 s of 16 kHz audio, left-pad shorter clips with zeros, then Whisper log-mel
features of shape (80, 800). The exported ONNX graph already applies the sigmoid, so its
output is P(turn complete).
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from pehleaap.paths import WEIGHTS_DIR

SAMPLE_RATE = 16000
WINDOW_S = 8
HF_REPO = "pipecat-ai/smart-turn-v3"


def download_weights(version: str = "3.2", variant: str = "cpu") -> Path:
    """Fetch `smart-turn-v{version}-{variant}.onnx` (cpu = int8, gpu = fp32) into weights/."""
    from huggingface_hub import hf_hub_download

    filename = f"smart-turn-v{version}-{variant}.onnx"
    return Path(hf_hub_download(HF_REPO, filename, local_dir=WEIGHTS_DIR / "smart-turn"))


def last_n_seconds(audio: np.ndarray, n_seconds: int = WINDOW_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Keep the last n seconds; left-pad with zeros so speech sits at the end of the window."""
    n = n_seconds * sr
    if len(audio) >= n:
        return audio[-n:]
    return np.pad(audio, (n - len(audio), 0))


class SmartTurnONNX:
    def __init__(self, onnx_path: str | Path, threads: int = 1):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def features(self, audio_16k: np.ndarray) -> np.ndarray:
        """(80, 800) float32 log-mel features for one clip."""
        inputs = self.feature_extractor(
            last_n_seconds(audio_16k.astype(np.float32)),
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_S * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        return inputs.input_features[0].astype(np.float32)

    def predict_features(self, feats: np.ndarray) -> np.ndarray:
        """Batch (B, 80, 800) -> (B,) probabilities."""
        return self.session.run(None, {"input_features": feats})[0].reshape(-1)

    def predict(self, audio_16k: np.ndarray) -> float:
        return float(self.predict_features(self.features(audio_16k)[None])[0])

    def predict_batch(self, clips: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(clips), batch_size):
            feats = np.stack([self.features(c) for c in clips[i : i + batch_size]])
            out.append(self.predict_features(feats))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
