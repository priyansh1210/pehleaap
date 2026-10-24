import ast
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _runner():
    spec = importlib.util.spec_from_file_location("kaggle_job", REPO / "scripts" / "kaggle_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kernel_folder_has_commit_command_and_metadata(tmp_path, monkeypatch):
    kj = _runner()
    monkeypatch.setattr(kj, "pushed_commit", lambda: "a" * 40)
    monkeypatch.setattr(kj, "RUNS_DIR", tmp_path)
    folder = kj.build_kernel("smoke", "someone/pehleaap-smoke", ["python", "-m", "pytest"], "NvidiaTeslaT4", ["someone/pehleaap-raw"])

    job = (folder / "job.py").read_text(encoding="utf-8")
    consts = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(job).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("COMMIT", "CMD")
    }
    assert consts == {"COMMIT": "a" * 40, "CMD": ["python", "-m", "pytest"]}

    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/pehleaap-smoke" and meta["title"] == "pehleaap-smoke"  # title must slugify to the id
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["someone/pehleaap-raw"]
    assert (folder / meta["code_file"]).exists()
