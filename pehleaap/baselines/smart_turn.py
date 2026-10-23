"""Pipecat Smart Turn v3.x ONNX baseline.

Preprocessing matches pipecat-ai/smart-turn `inference.py` and `audio_utils.py` exactly:
keep the last 8 s of 16 kHz audio, left-pad shorter clips with zeros, then Whisper log-mel
features of shape (80, 800) with waveform normalisation. The exported ONNX graph already
applies the sigmoid, so its output is P(turn complete).
"""

from pathlib import Path

from pehleaap.baselines.turn_models import SAMPLE_RATE, WhisperMelOnnx, fetch, last_n_seconds  # noqa: F401

HF_REPO = "pipecat-ai/smart-turn-v3"


def download_weights(version: str = "3.2", variant: str = "cpu") -> Path:
    """Fetch `smart-turn-v{version}-{variant}.onnx` (cpu = int8, gpu = fp32) into weights/."""
    return fetch(HF_REPO, f"smart-turn-v{version}-{variant}.onnx")


class SmartTurnONNX(WhisperMelOnnx):
    def __init__(self, onnx_path: str | Path, threads: int = 1):
        super().__init__(onnx_path, threads, do_normalize=True, output_is_logit=False)
