"""Int8 quantisation of a Smart Turn-format ONNX model with upstream's recipe.

pipecat-ai/smart-turn train.py (quantize_onnx_model) made smart-turn-v3.2-cpu.onnx this way:
`quant_pre_process` (symbolic shape inference skipped), then static QDQ quantisation with uint8
activations, int8 per-channel weights and entropy calibration, on Conv/MatMul/Gemm only, calibrated
on clean training clips. Run on Pipecat's fp32 file it reproduces their published int8 file on our
dev clips (mean probability shift +0.034 vs +0.033). Quantising every op type instead shifted eot_v0's
probabilities by -0.1 to -0.2.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np


# pipecat-ai/smart-turn train.py (quantize_onnx_model) quantises only these.
UPSTREAM_OP_TYPES = ("Conv", "MatMul", "Gemm")


class _Features:
    """CalibrationDataReader over (N, 80, 800) features, one clip per call, as the deployed model runs."""

    def __init__(self, features: np.ndarray, input_name: str = "input_features"):
        self.features, self.input_name, self.i = features, input_name, 0

    def get_next(self):
        if self.i >= len(self.features):
            return None
        self.i += 1
        return {self.input_name: self.features[self.i - 1 : self.i].astype(np.float32)}

    def rewind(self):
        self.i = 0


class _Chunk:
    """At most `size` clips of `reader`, so the histogram calibrator never holds more activations than that."""

    def __init__(self, reader: _Features, size: int):
        self.reader, self.left = reader, size

    def get_next(self):
        if self.left == 0:
            return None
        self.left -= 1
        return self.reader.get_next()


@contextmanager
def _chunked_histograms(size: int):
    """onnxruntime's histogram (entropy/percentile) calibrator keeps every activation of every clip
    until the reader runs dry (about 150 MB per clip here; CalibMaxIntermediateOutputs only applies
    to MinMax). Feed it `size` clips at a time instead: the histograms merge across calls."""
    from onnxruntime.quantization.calibrate import HistogramCalibrater

    original = HistogramCalibrater.collect_data

    def collect_data(self, reader):
        while reader.i < len(reader.features):
            original(self, _Chunk(reader, size))

    HistogramCalibrater.collect_data = collect_data
    try:
        yield
    finally:
        HistogramCalibrater.collect_data = original


def quantize_eot(fp32_path: str | Path, out_path: str | Path, features: np.ndarray, method: str = "entropy",
                 keep_fp32: tuple[str, ...] = (), op_types: tuple[str, ...] | None = UPSTREAM_OP_TYPES) -> Path:
    """Static QDQ int8 copy of `fp32_path`, calibrated on `features`, written to `out_path`.
    Only `op_types` are quantised (None: every op onnxruntime can); nodes whose names start with any
    of `keep_fp32` stay in float."""
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepped = out_path.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32_path), str(prepped), skip_optimization=False, skip_symbolic_shape=True)
    exclude = [n.name for n in onnx.load(str(prepped)).graph.node if n.name.startswith(keep_fp32)] if keep_fp32 else []
    try:
        with _chunked_histograms(32):
            quantize_static(
                str(prepped), str(out_path), _Features(features),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                calibrate_method={"entropy": CalibrationMethod.Entropy, "minmax": CalibrationMethod.MinMax,
                                  "percentile": CalibrationMethod.Percentile}[method],
                nodes_to_exclude=exclude, op_types_to_quantize=list(op_types) if op_types else None,
            )
    finally:
        prepped.unlink(missing_ok=True)
    return out_path
