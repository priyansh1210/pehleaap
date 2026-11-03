import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("compute_bench", Path(__file__).resolve().parent.parent / "scripts" / "compute_bench.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def test_streams_per_core():
    # VAD 2 ms per second of audio + a 30 ms decision 20 times a minute = 12 busy ms per second of call.
    assert abs(cb.streams_per_core(2.0, 30.0, 20) - 1000 / 12) < 1e-9
    assert cb.streams_per_core(0.0, 60.0, 60) == 1000 / 60


def test_median_ms_calls_warm_up_then_every_round():
    calls = []
    assert cb.median_ms(lambda: calls.append(1), runs=10, rounds=3) >= 0
    assert len(calls) == 1 + 30
