"""Any end-of-turn model in `turn_models.REGISTRY` as a Pipecat turn analyzer.

Pipecat's LocalSmartTurnAnalyzerV3 hard-codes Smart Turn's interface: input name `input_features`,
Whisper-normalised features, sigmoid output and a 0.5 threshold. The community Hinglish models
differ in some or all of these, and abhishek's model even shares the input name, so it would load
there and silently return wrong probabilities. This analyzer runs each model with its own
preprocessing and model-card threshold, so the demo can switch baselines fairly.
"""

import time

import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

from pehleaap.audio.io import resample
from pehleaap.baselines.turn_models import REGISTRY, SAMPLE_RATE, load


class ClipModelTurnAnalyzer(BaseSmartTurn):
    def __init__(self, *, model: str = "smart_turn_v3.2_int8", threshold: float | None = None, threads: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self.threshold = REGISTRY[model].threshold if threshold is None else threshold
        self._model = load(model, threads=threads)

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
