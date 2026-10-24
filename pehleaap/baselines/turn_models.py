"""End-of-turn clip classifiers behind one interface: `predict_batch(clips_16k) -> P(turn complete)`.

Two input styles cover every open model we compare against:

* WhisperMelOnnx - Whisper log-mel features (80, 800) computed outside the graph
  (Smart Turn v3.x and fine-tunes of it).
* WaveformOnnx   - the last 8 s of raw 16 kHz audio, with the frontend inside the graph.

Every model sees the same window: the last 8 s, left-padded with zeros so speech ends at
the end of the window. Preprocessing flags per model come from each model card and are
checked by reproducing the card's published accuracy (scripts/eval_clip_baselines.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pehleaap.paths import WEIGHTS_DIR

SAMPLE_RATE = 16000
WINDOW_S = 8
WINDOW = SAMPLE_RATE * WINDOW_S


def last_n_seconds(audio: np.ndarray, n_seconds: int = WINDOW_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Keep the last n seconds; left-pad with zeros so speech sits at the end of the window."""
    n = n_seconds * sr
    if len(audio) >= n:
        return audio[-n:]
    return np.pad(audio, (n - len(audio), 0))


def fetch(repo: str, filename: str) -> Path:
    """Download one file from a Hugging Face model repo into weights/<owner>__<name>/."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, local_dir=WEIGHTS_DIR / repo.replace("/", "__")))


def _session(onnx_path: str | Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])


class _OnnxClipModel:
    def __init__(self, onnx_path: str | Path, threads: int = 1, output_is_logit: bool = False):
        self.session = _session(onnx_path, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Some exports fix the batch dimension at 1.
        self.max_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.output_is_logit = output_is_logit

    def prepare(self, audio_16k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_prepared(self, batch: np.ndarray) -> np.ndarray:
        """Batch of prepared inputs -> (B,) P(turn complete)."""
        out = self.session.run(None, {self.input_name: batch.astype(np.float32)})[0].reshape(-1)
        return 1.0 / (1.0 + np.exp(-out)) if self.output_is_logit else out

    def predict(self, audio_16k: np.ndarray) -> float:
        return float(self.predict_prepared(self.prepare(audio_16k)[None])[0])

    def predict_batch(self, clips: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if self.max_batch:
            batch_size = min(batch_size, self.max_batch)
        out = []
        for i in range(0, len(clips), batch_size):
            out.append(self.predict_prepared(np.stack([self.prepare(c) for c in clips[i : i + batch_size]])))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class WhisperMelOnnx(_OnnxClipModel):
    """Input: Whisper log-mel (80, 800) of the last 8 s."""

    def __init__(self, onnx_path, threads: int = 1, do_normalize: bool = True, output_is_logit: bool = False):
        from transformers import WhisperFeatureExtractor

        super().__init__(onnx_path, threads, output_is_logit)
        self.do_normalize = do_normalize
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def prepare(self, audio_16k: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            last_n_seconds(audio_16k.astype(np.float32)),
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW,
            truncation=True,
            do_normalize=self.do_normalize,
        )
        return inputs.input_features[0].astype(np.float32)

    # Kept for callers that name the step explicitly.
    features = prepare


class WaveformOnnx(_OnnxClipModel):
    """Input: raw last-8 s waveform (128000,), optionally standardised over the real samples first."""

    def __init__(self, onnx_path, threads: int = 1, standardize: bool = False, output_is_logit: bool = False):
        super().__init__(onnx_path, threads, output_is_logit)
        self.standardize = standardize

    def prepare(self, audio_16k: np.ndarray) -> np.ndarray:
        x = audio_16k[-WINDOW:].astype(np.float32)
        if self.standardize:
            x = (x - x.mean()) / (x.std() + 1e-7)
        return last_n_seconds(x)


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    filename: str
    kind: str  # "mel" or "wave"
    threshold: float  # the decision threshold the model card recommends for this file
    kwargs: dict = field(default_factory=dict)
    license: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "smart_turn_v3.2_int8": ModelSpec(
        "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx", "mel", 0.5, {"do_normalize": True}, "BSD-2-Clause"
    ),
    "deveshu_fp32": ModelSpec(
        "deveshu/hinglish-turn-detector", "model_fp32.onnx", "mel", 0.63,
        {"do_normalize": False, "output_is_logit": True}, "MIT",
    ),
    "abhishek_tiny_attn": ModelSpec(
        "abhishek-040010/smart-turn-hinglish", "smart-turn-v3-hinglish-whisper-tiny-attention.onnx", "mel", 0.5,
        {"do_normalize": False, "output_is_logit": True}, "Apache-2.0",
    ),
    "codewithmoin_8s": ModelSpec(
        "codewithmoin/smart-turn-hinglish", "smart_turn_8s.onnx", "wave", 0.5, {}, "BSD-2-Clause"
    ),
    "chhotu_int8": ModelSpec(
        "Vishwas07/chhotu-turn-detection", "chhotu_int8.onnx", "wave", 0.48, {"standardize": True},
        "none stated (card: research/evaluation only)",
    ),
}


def load(name: str, threads: int = 1) -> _OnnxClipModel:
    spec = REGISTRY[name]
    path = fetch(spec.repo, spec.filename)
    cls = WhisperMelOnnx if spec.kind == "mel" else WaveformOnnx
    return cls(path, threads=threads, **spec.kwargs)


def load_any(name: str, threads: int = 1) -> tuple[_OnnxClipModel, float]:
    """(model, decision threshold) for a REGISTRY name, or for one of our own Smart-Turn-format ONNX files:
    weights/<name>/model.onnx (e.g. eot_v0, from train_eot.py) or a path ending in .onnx, at threshold 0.5.

    Our models stay out of REGISTRY so the baseline evals' default model list never changes under them.
    """
    if name in REGISTRY:
        return load(name, threads), REGISTRY[name].threshold
    path = Path(name) if name.endswith(".onnx") else WEIGHTS_DIR / name / "model.onnx"
    if not path.exists():
        raise FileNotFoundError(f"{name!r} is not a registry model and {path} doesn't exist")
    return WhisperMelOnnx(path, threads=threads), 0.5
