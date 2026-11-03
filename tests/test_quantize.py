import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from pehleaap.models.quantize import quantize_eot


def _two_matmuls(path):
    rng = np.random.default_rng(0)
    w1, w2 = (numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n) for s, n in (((4, 8), "w1"), ((8, 1), "w2")))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input_features", "w1"], ["h"], name="first"),
         helper.make_node("Relu", ["h"], ["r"], name="relu"),
         helper.make_node("MatMul", ["r", "w2"], ["logits"], name="last")],
        "g", [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1])], [w1, w2])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)


def test_keep_last_fp32_leaves_the_last_node_in_float(tmp_path):
    _two_matmuls(tmp_path / "fp32.onnx")
    feats = np.random.default_rng(1).standard_normal((16, 4)).astype(np.float32)
    out = quantize_eot(tmp_path / "fp32.onnx", tmp_path / "int8.onnx", feats, "minmax", keep_last_fp32=1)
    graph = onnx.load(str(out)).graph
    weights = {i.name: i.data_type for i in graph.initializer}
    assert weights["w2"] == TensorProto.FLOAT  # the last MatMul's weight is untouched
    assert "w1" not in weights or weights["w1"] != TensorProto.FLOAT  # the first one is quantised
    assert any(n.op_type == "DequantizeLinear" and n.output[0].startswith("w1") for n in graph.node)
