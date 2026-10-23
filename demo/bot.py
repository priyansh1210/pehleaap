"""Week-1 demo skeleton: a browser call where the bot answers each finished turn with a short ding.

    .venv-demo\\Scripts\\python demo\\bot.py              # then open http://localhost:7860/client
    $env:PEHLEAAP_TURN_MODEL = "deveshu_fp32"           # any name in pehleaap.baselines.turn_models.REGISTRY
    $env:PEHLEAAP_PHONE_LINE = "0"                      # hear the model on wideband audio instead

The caller's audio goes through a simulated 8 kHz G.711 phone line, then Silero VAD, then the
chosen end-of-turn model. There is no STT, LLM or TTS yet: the ding is the bot taking the turn,
so cut-ins (ding mid-sentence) and slow replies can be heard directly, and each turn's gap from
end of speech to ding is logged. Week 5 replaces the env vars with a live baseline/ours toggle.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.pipecat_ext.telephony_sim import PhoneLineProcessor  # noqa: E402
from pehleaap.pipecat_ext.turn_analyzer import ClipModelTurnAnalyzer  # noqa: E402

SAMPLE_RATE = 16000


def ding(sr: int = SAMPLE_RATE, hz: float = 880.0, secs: float = 0.15) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    tone = 0.3 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
    return (tone * 32767).astype(np.int16).tobytes()


class DingOnTurnEnd(FrameProcessor):
    """Plays a ding when the user's turn ends and logs the gap from VAD end-of-speech to the decision."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sound = ding()
        self._speech_end = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._speech_end = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # VAD reports the stop stop_secs after speech actually ended.
            self._speech_end = time.monotonic() - frame.stop_secs
        elif isinstance(frame, UserStoppedSpeakingFrame):
            gap = f"{(time.monotonic() - self._speech_end) * 1000:.0f} ms" if self._speech_end else "n/a"
            logger.info(f"[pehleaap] turn end -> ding, gap from end of speech: {gap}")
            await self.push_frame(OutputAudioRawFrame(self._sound, SAMPLE_RATE, 1))


async def bot(runner_args: SmallWebRTCRunnerArguments):
    turn_model = os.environ.get("PEHLEAAP_TURN_MODEL", "smart_turn_v3.2_int8")
    phone_line = os.environ.get("PEHLEAAP_PHONE_LINE", "1") != "0"
    logger.info(f"[pehleaap] turn model: {turn_model}, phone line: {'G.711 8 kHz' if phone_line else 'off'}")

    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            # No STT in the skeleton, so the turn model alone decides.
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=ClipModelTurnAnalyzer(model=turn_model), wait_for_transcript=False)],
        )
    )
    pipeline = Pipeline(
        [
            transport.input(),
            PhoneLineProcessor(enabled=phone_line),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))),
            turns,
            DingOnTurnEnd(),
            transport.output(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    # The runner's banner uses box-drawing characters; Windows pipes and log files default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
