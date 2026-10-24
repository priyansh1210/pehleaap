"""Smart Turn v3 as a PyTorch model, loaded from the published v3.2 weights, for fine-tuning.

The model class is vendored from pipecat-ai/smart-turn (`train.py`, commit 4786657, BSD-2-Clause):
a Whisper-tiny encoder over 8 s of log-mel (80 x 800), attention pooling, and a small classifier.
Module names match upstream, so upstream checkpoints load into it as they are.

Pipecat publishes v3.2 only as ONNX. `from_published()` reads the weights out of the fp32 file
(`smart-turn-v3.2-gpu.onnx`); attention and feed-forward matrices are stored there unnamed and
transposed, so each is matched to its layer by the named bias added to its output. The rebuilt
model matches the ONNX file to within 4e-6 in probability on 256 real test clips (checked
2026-10-24; tests/test_smart_turn_torch.py keeps checking it).

Input features come from `baselines.turn_models.WhisperMelOnnx.prepare` (last 8 s, normalized
log-mel), the same as for the ONNX baseline.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from pehleaap.baselines.turn_models import fetch

PUBLISHED = ("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx")


def whisper_tiny_config() -> WhisperConfig:
    """openai/whisper-tiny's encoder, cut to 400 positions (8 s) as upstream does."""
    return WhisperConfig(
        d_model=384, encoder_layers=4, encoder_attention_heads=6, encoder_ffn_dim=1536, num_mel_bins=80, max_source_positions=400
    )


class SmartTurnV3(nn.Module):
    def __init__(self, config: WhisperConfig | None = None):
        super().__init__()
        config = config or whisper_tiny_config()
        self.encoder = WhisperEncoder(config)
        hidden = config.d_model
        self.pool_attention = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """(batch, 80, 800) log-mel -> (batch,) logits of P(turn complete)."""
        hidden = self.encoder(input_features=input_features).last_hidden_state
        weights = torch.softmax(self.pool_attention(hidden), dim=1)
        return self.classifier((hidden * weights).sum(dim=1)).squeeze(-1)


def state_dict_from_onnx(path: str | Path) -> dict[str, torch.Tensor]:
    """Weights of an upstream fp32 ONNX export, as a SmartTurnV3 state dict."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    state = {k.removeprefix("inner."): v for k, v in init.items() if k.startswith("inner.")}
    k_layer = 0
    for node in graph.node:
        weight = [x for x in node.input if x in init and not x.startswith("inner.")] if node.op_type == "MatMul" else []
        if not weight:
            continue
        bias = next(
            (x for c in consumers.get(node.output[0], []) if c.op_type == "Add" for x in c.input if x.startswith("inner.") and x.endswith(".bias")),
            None,
        )
        if bias is None:  # k_proj is the only projection without a bias; the layers come in order
            name = f"encoder.layers.{k_layer}.self_attn.k_proj.weight"
            k_layer += 1
        else:
            name = bias.removeprefix("inner.").removesuffix(".bias") + ".weight"
        if name in state:
            raise ValueError(f"two ONNX weights map to {name}")
        state[name] = init[weight[0]].T  # MatMul stores (in, out); nn.Linear wants (out, in)
    return {k: torch.from_numpy(np.array(v)) for k, v in state.items()}  # a writable, contiguous copy


def from_published(onnx_path: str | Path | None = None) -> SmartTurnV3:
    """Smart Turn v3.2 with its published weights (downloads the fp32 ONNX into weights/ on first use)."""
    model = SmartTurnV3()
    model.load_state_dict(state_dict_from_onnx(onnx_path or fetch(*PUBLISHED)), strict=True)
    return model.eval()
