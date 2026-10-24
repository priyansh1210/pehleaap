"""End-of-turn models as Pipecat turn analyzers: any registry baseline or one of our own models.

Pipecat's LocalSmartTurnAnalyzerV3 hard-codes Smart Turn's interface: input name `input_features`,
Whisper-normalised features, sigmoid output and a 0.5 threshold. The community Hinglish models
differ in some or all of these, and abhishek's model even shares the input name, so it would load
there and silently return wrong probabilities. These analyzers run each model with its own
preprocessing and model-card threshold, so the demo can switch models fairly.

`SwitchTurnAnalyzer` holds several models at once; a `ModelSwitch` (flipped from the demo's /switch
page) says which one decides. At every pause the others are scored too, on a background thread
after the decision, and logged next to it, so their verdicts show without adding delay.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

from pehleaap.audio.io import resample
from pehleaap.baselines.turn_models import SAMPLE_RATE, load_any

_LOADED: dict[tuple[str, int], tuple] = {}  # (name, threads) -> (model, threshold), shared across calls


def loaded(name: str, threads: int = 1) -> tuple:
    if (name, threads) not in _LOADED:
        _LOADED[(name, threads)] = load_any(name, threads)
    return _LOADED[(name, threads)]


class ClipModelTurnAnalyzer(BaseSmartTurn):
    def __init__(self, *, model: str = "smart_turn_v3.2_int8", threshold: float | None = None, threads: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self._model, card = loaded(model, threads)
        self.threshold = card if threshold is None else threshold

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict:
        """float32 audio in [-1, 1] at the pipeline rate -> Pipecat's {prediction, probability}."""
        sr = self.sample_rate or SAMPLE_RATE
        audio = resample(audio_array, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else audio_array
        t0 = time.perf_counter()
        p = self._model.predict(audio)
        logger.info(
            f"[pehleaap] {self.model_name}: p(complete)={p:.3f} (threshold {self.threshold}), "
            f"{len(audio) / SAMPLE_RATE:.1f}s audio, {(time.perf_counter() - t0) * 1000:.0f} ms"
        )
        return {"prediction": int(p > self.threshold), "probability": p}


class ModelSwitch:
    """Which of `names` decides turn ends right now. Shared by every call in the process."""

    def __init__(self, names: list[str], active: str):
        if active not in names:
            raise ValueError(f"{active!r} is not one of {names}")
        self.names, self.active = list(names), active

    def set(self, name: str) -> None:
        if name not in self.names:
            raise KeyError(name)
        self.active = name


class SwitchTurnAnalyzer(BaseSmartTurn):
    def __init__(self, *, switch: ModelSwitch, threads: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.switch = switch
        self._models = {name: loaded(name, threads) for name in switch.names}
        self._shadow = ThreadPoolExecutor(1, thread_name_prefix="pehleaap-shadow")

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict:
        sr = self.sample_rate or SAMPLE_RATE
        audio = resample(audio_array, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else audio_array
        name = self.switch.active
        model, threshold = self._models[name]
        t0 = time.perf_counter()
        p = model.predict(audio)
        verdict = "END" if p > threshold else "wait"
        logger.info(
            f"[pehleaap] {name} decides: p(complete)={p:.3f} vs {threshold} -> {verdict} "
            f"({len(audio) / SAMPLE_RATE:.1f}s audio, {(time.perf_counter() - t0) * 1000:.0f} ms)"
        )
        others = [n for n in self._models if n != name]
        if others:
            self._shadow.submit(self._log_others, audio, others)
        return {"prediction": int(p > threshold), "probability": p}

    def _log_others(self, audio: np.ndarray, names: list[str]) -> None:
        for name in names:
            model, threshold = self._models[name]
            p = model.predict(audio)
            logger.info(f"[pehleaap]   {name} would say: p(complete)={p:.3f} vs {threshold} -> {'END' if p > threshold else 'wait'}")

    async def cleanup(self):
        await super().cleanup()
        self._shadow.shutdown(wait=False)
