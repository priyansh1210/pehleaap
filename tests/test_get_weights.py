import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "get_weights.py"
spec = importlib.util.spec_from_file_location("get_weights", SCRIPT)
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)


def test_downloads_once_and_checks_the_checksum(tmp_path):
    src = tmp_path / "release" / "m.onnx"
    src.parent.mkdir()
    src.write_bytes(b"model bytes")
    sha = hashlib.sha256(b"model bytes").hexdigest()
    out = tmp_path / "weights" / "m" / "model.onnx"

    assert gw.fetch(src.as_uri(), out, sha) == "downloaded"
    assert out.read_bytes() == b"model bytes"
    assert gw.fetch(src.as_uri(), out, sha) == "present"

    with pytest.raises(SystemExit, match="checksum"):
        gw.fetch(src.as_uri(), tmp_path / "bad" / "model.onnx", "0" * 64)
    assert not (tmp_path / "bad" / "model.onnx").exists()
    assert not (tmp_path / "bad" / "model.onnx.part").exists()


def test_every_released_file_has_a_full_checksum():
    for name, files in gw.MODELS.items():
        for asset, filename, sha in files:
            assert asset.startswith(name) and filename.endswith(".onnx")
            assert len(sha) == 64 and int(sha, 16) >= 0, f"{asset}: sha256 missing"
