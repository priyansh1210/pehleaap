"""Silero VAD on numpy audio, at 16 kHz (clean source, for labeling) or 8 kHz (phone line).

Runs the ONNX model shipped in the silero-vad package directly through onnxruntime, so it
doesn't import torch, and can run several equal-length tracks as one batch. It gives one speech
probability per 32 ms frame (512 samples at 16 kHz, 256 at 8 kHz): frame i covers
[i * hop, (i + 1) * hop) seconds and sees only audio up to its end, so it is causal.
"""

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort

FRAME = {16000: 512, 8000: 256}
CONTEXT = {16000: 64, 8000: 32}  # samples of the previous frame prepended to each input, as in silero's wrapper


def model_path() -> Path:
    # find_spec locates the package without importing it (its __init__ imports torch).
    spec = importlib.util.find_spec("silero_vad")
    if spec is None:
        raise ImportError("silero-vad is not installed")
    return Path(spec.origin).parent / "data" / "silero_vad.onnx"


def hop_s(sr: int) -> float:
    return FRAME[sr] / sr


class SileroVAD:
    def __init__(self, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path()), opts, providers=["CPUExecutionProvider"])

    def probs(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Speech probability per frame for mono (n,) audio, or a batch of tracks (b, n)."""
        if sr not in FRAME:
            raise ValueError(f"Silero VAD runs at 8000 or 16000 Hz, not {sr}")
        frame, ctx = FRAME[sr], CONTEXT[sr]
        batch = np.atleast_2d(np.asarray(x, dtype=np.float32))
        b, n = batch.shape
        n_frames = -(-n // frame)
        # Left-pad with the (zero) context of frame 0; right-pad the last partial frame.
        padded = np.pad(batch, ((0, 0), (ctx, n_frames * frame - n)))
        state = np.zeros((2, b, 128), np.float32)
        sr_arr = np.array(sr, dtype=np.int64)
        out = np.empty((b, n_frames), np.float32)
        for i in range(n_frames):
            chunk = padded[:, i * frame : (i + 1) * frame + ctx]
            p, state = self.session.run(None, {"input": chunk, "state": state, "sr": sr_arr})
            out[:, i] = p[:, 0]
        return out[0] if np.ndim(x) == 1 else out


def speech_segments(probs: np.ndarray, hop: float, threshold: float = 0.5, neg_threshold: float = 0.35) -> np.ndarray:
    """Hysteresis over frame probabilities: (k, 2) array of [start_s, end_s] speech segments.

    Speech starts at the first frame with p >= threshold and lasts until the first frame with
    p < neg_threshold (Silero's own two-threshold rule, without its padding or min-duration
    filters, which `labeling.ipu` applies instead).
    """
    segs = []
    start = None
    for i, p in enumerate(probs):
        if start is None and p >= threshold:
            start = i
        elif start is not None and p < neg_threshold:
            segs.append((start * hop, i * hop))
            start = None
    if start is not None:
        segs.append((start * hop, len(probs) * hop))
    return np.array(segs, dtype=float).reshape(-1, 2)
