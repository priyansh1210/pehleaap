"""Pipecat integration tests. Run in the demo venv: .venv-demo\\Scripts\\python -m pytest tests\\test_pipecat_ext.py"""

import asyncio
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3  # noqa: E402
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, UserStoppedSpeakingFrame  # noqa: E402

from pehleaap.pipecat_ext.telephony_sim import PhoneLineProcessor  # noqa: E402
from pehleaap.pipecat_ext.turn_analyzer import ClipModelTurnAnalyzer  # noqa: E402
from pehleaap.telephony.realtime import PhoneLine  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    """Noise with a ~4 Hz syllable envelope, float32 in [-1, 1] at 16 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000
    return (0.2 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) ** 2).astype(np.float32)


@pytest.mark.parametrize("seconds", [2.0, 9.0])  # shorter than and longer than the 8 s window
def test_smart_turn_matches_pipecats_own_analyzer(seconds):
    x = _speechlike(seconds, seed=int(seconds))
    ours = ClipModelTurnAnalyzer(model="smart_turn_v3.2_int8")._predict_endpoint(x)
    theirs = LocalSmartTurnAnalyzerV3()._predict_endpoint(x)  # same ONNX, bundled with pipecat
    assert abs(ours["probability"] - theirs["probability"]) < 1e-3
    assert ours["prediction"] == theirs["prediction"]


def _second_model() -> str:
    """A local Smart-Turn-format ONNX to switch to: eot_v0 if it's on this machine, else Pipecat's fp32 file."""
    from pehleaap.baselines.turn_models import fetch
    from pehleaap.paths import WEIGHTS_DIR

    if (WEIGHTS_DIR / "eot_v0" / "model.onnx").exists():
        return "eot_v0"
    return str(fetch("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-gpu.onnx"))


def test_switch_decides_with_the_active_model_only():
    from pehleaap.pipecat_ext.turn_analyzer import ModelSwitch, SwitchTurnAnalyzer, loaded

    other = _second_model()
    switch = ModelSwitch(["smart_turn_v3.2_int8", other], "smart_turn_v3.2_int8")
    analyzer = SwitchTurnAnalyzer(switch=switch)
    x = _speechlike(3.0, seed=7)
    for name in switch.names:
        switch.set(name)
        model, threshold = loaded(name)
        out = analyzer._predict_endpoint(x)
        assert out["probability"] == pytest.approx(model.predict(x))
        assert out["prediction"] == int(out["probability"] > threshold)
    with pytest.raises(KeyError):
        switch.set("nope")
    with pytest.raises(ValueError):
        ModelSwitch(["a"], "b")


def test_switch_page_and_routes():
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("demo_bot_routes", REPO / "demo" / "bot.py")
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)
    client = TestClient(bot.app)
    assert "Who decides" in client.get("/switch").text
    state = client.get("/switch/state").json()
    assert state["active"] in state["models"] and "smart_turn_v3.2_int8" in state["models"]
    assert client.post("/switch/smart_turn_v3.2_int8").json()["active"] == "smart_turn_v3.2_int8"
    assert bot.SWITCH.active == "smart_turn_v3.2_int8"
    assert client.post("/switch/not-a-model").status_code == 404


def test_phone_line_processor_matches_offline_phone_line():
    rng = np.random.default_rng(0)
    pcm = np.clip(rng.normal(0, 3000, 16000), -32768, 32767).astype(np.int16)
    frames = [InputAudioRawFrame(pcm[i : i + 320].tobytes(), 16000, 1) for i in range(0, len(pcm), 320)]
    proc = PhoneLineProcessor()
    for f in frames:
        proc.degrade(f)
    out = np.concatenate([np.frombuffer(f.audio, dtype=np.int16) for f in frames])
    assert all(f.num_frames == 320 for f in frames)
    assert np.max(np.abs(out.astype(np.int32) - PhoneLine(16000).process(pcm).astype(np.int32))) <= 1


def _complete_hindi_turn() -> np.ndarray:
    """First complete-turn Hindi clip (2-6 s) from the Smart Turn test set, as int16 at 16 kHz."""
    pq = pytest.importorskip("pyarrow.parquet")
    import soundfile as sf

    from pehleaap.data import smart_turn_testset

    shards = smart_turn_testset.shard_paths()
    if not shards:
        pytest.skip("Smart Turn test set not downloaded")
    for batch in pq.ParquetFile(shards[0]).iter_batches(batch_size=64, columns=["language", "endpoint_bool", "audio"]):
        for row in batch.to_pylist():
            if row["language"] == "hin" and row["endpoint_bool"]:
                x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
                if sr == 16000 and x.ndim == 1 and 2 * sr <= len(x) <= 6 * sr:
                    return x
    pytest.skip("no suitable Hindi clip in the first shard")


def test_demo_pipeline_ends_one_turn_and_dings_once():
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.tests.utils import SleepFrame, run_test
    from pipecat.turns.user_start import VADUserTurnStartStrategy
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    spec = importlib.util.spec_from_file_location("demo_bot", REPO / "demo" / "bot.py")
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)

    speech = _complete_hindi_turn()
    audio = np.concatenate([np.zeros(8000, np.int16), speech, np.zeros(16000 * 4, np.int16)])  # 4 s > 3 s fallback
    frames = [InputAudioRawFrame(audio[i : i + 320].tobytes(), 16000, 1) for i in range(0, len(audio) - 319, 320)]
    analyzer = ClipModelTurnAnalyzer(model="smart_turn_v3.2_int8")
    pipeline = Pipeline(
        [
            PhoneLineProcessor(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))),
            UserTurnProcessor(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy()],
                    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, wait_for_transcript=False)],
                )
            ),
            bot.DingOnTurnEnd(),
        ]
    )
    # Feed audio in real time (20 ms per frame) so VAD and the analyzer see realistic timing.
    paced = [f for frame in frames for f in (frame, SleepFrame(sleep=0.02))]
    down, _ = asyncio.run(run_test(pipeline, frames_to_send=paced + [SleepFrame(sleep=1.0)]))

    assert sum(isinstance(f, UserStoppedSpeakingFrame) for f in down) == 1
    dings = [f for f in down if isinstance(f, OutputAudioRawFrame)]
    assert len(dings) == 1 and dings[0].sample_rate == 16000
