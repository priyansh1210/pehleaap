"""Demo: a browser call where the bot answers each finished turn with a short ding.

    .venv-demo\\Scripts\\python demo\\bot.py     # then open http://localhost:7860/client to talk,
                                              # and http://localhost:7860/switch in a second tab
    $env:PEHLEAAP_TURN_MODELS = "smart_turn_v3.2_int8,eot_v0,deveshu_fp32"   # which models to load
    $env:PEHLEAAP_TURN_MODEL = "smart_turn_v3.2_int8"                         # which one decides first
    $env:PEHLEAAP_PHONE_LINE = "0"                                             # wideband audio instead

The caller's audio goes through a simulated 8 kHz G.711 phone line, then Silero VAD, then the
end-of-turn model that is switched on. By default that is Smart Turn v3.2 (the baseline) and eot_v0
(ours: weights/eot_v0/model.onnx from train_eot.py), starting with eot_v0; the /switch page flips
between them mid-call. At every pause the log shows the deciding model's verdict and what the
other would have said. There is no STT, LLM or TTS yet: the ding is the bot taking the turn, so
cut-ins (ding mid-sentence) and slow replies can be heard directly, and each turn's gap from end of
speech to ding is logged.
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

from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pipecat.runner.run import app  # noqa: E402

from pehleaap.baselines.turn_models import REGISTRY  # noqa: E402
from pehleaap.paths import WEIGHTS_DIR  # noqa: E402
from pehleaap.pipecat_ext.telephony_sim import PhoneLineProcessor  # noqa: E402
from pehleaap.pipecat_ext.turn_analyzer import ModelSwitch, SwitchTurnAnalyzer  # noqa: E402

SAMPLE_RATE = 16000
LABELS = {"smart_turn_v3.2_int8": "Smart Turn v3.2 (baseline)", "eot_v0": "eot_v0 (ours: phone-line fine-tune)"}


def available_models() -> list[str]:
    wanted = [m.strip() for m in os.environ.get("PEHLEAAP_TURN_MODELS", "smart_turn_v3.2_int8,eot_v0").split(",") if m.strip()]
    found = [m for m in wanted if m in REGISTRY or m.endswith(".onnx") or (WEIGHTS_DIR / m / "model.onnx").exists()]
    for m in sorted(set(wanted) - set(found)):
        logger.warning(f"[pehleaap] skipping turn model {m}: not in the registry and no {WEIGHTS_DIR / m / 'model.onnx'}")
    return found


_models = available_models()
SWITCH = ModelSwitch(_models, os.environ.get("PEHLEAAP_TURN_MODEL", "eot_v0" if "eot_v0" in _models else _models[0]))

SWITCH_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turn model switch</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 34rem; margin: 2rem auto; padding: 0 16px; background: #fff; color: #111; }
  button { display: block; width: 100%; font-size: 1.1rem; padding: .8rem 1rem; margin: .5rem 0; text-align: left;
           border: 2px solid #999; border-radius: 8px; background: #f4f4f4; color: #111; cursor: pointer; }
  button.on { border-color: #087f5b; background: #e6f6ef; font-weight: 600; }
  p { color: #444; line-height: 1.4; }
</style></head>
<body>
<h1>Who decides when your turn is over?</h1>
<p>Switch any time during the call: the next pause uses the new model. The terminal log shows what the other model would have said at every pause.</p>
<div id="models"></div>
<script>
async function show(state) {
  const box = document.getElementById("models");
  box.innerHTML = "";
  for (const m of state.models) {
    const b = document.createElement("button");
    b.textContent = (state.labels[m] || m) + (m === state.active ? "  - deciding now" : "");
    if (m === state.active) b.className = "on";
    b.onclick = async () => show(await (await fetch("/switch/" + encodeURIComponent(m), { method: "POST" })).json());
    box.appendChild(b);
  }
}
fetch("/switch/state").then(r => r.json()).then(show);
</script>
</body></html>
"""


def switch_state() -> dict:
    return {"active": SWITCH.active, "models": SWITCH.names, "labels": {m: LABELS.get(m, m) for m in SWITCH.names}}


@app.get("/switch", response_class=HTMLResponse)
async def switch_page():
    return SWITCH_PAGE


@app.get("/switch/state")
async def get_switch_state():
    return switch_state()


@app.post("/switch/{name}")
async def set_switch(name: str):
    try:
        SWITCH.set(name)
    except KeyError:
        return JSONResponse({"error": f"unknown turn model {name!r}", **switch_state()}, status_code=404)
    logger.info(f"[pehleaap] turn model switched to {name}")
    return switch_state()


def ding(sr: int = SAMPLE_RATE, hz: float = 880.0, secs: float = 0.15) -> bytes:
    t = np.arange(int(sr * secs)) / sr
    tone = 0.3 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
    return (tone * 32767).astype(np.int16).tobytes()


class DingOnTurnEnd(FrameProcessor):
    """Plays a ding when the user's turn ends and logs the gap from VAD end-of-speech to the decision."""

    def __init__(self, switch: ModelSwitch | None = None, **kwargs):
        super().__init__(**kwargs)
        self._sound = ding()
        self._speech_end = None
        self._switch = switch

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
            who = f" ({self._switch.active})" if self._switch else ""
            logger.info(f"[pehleaap] turn end{who} -> ding, gap from end of speech: {gap}")
            await self.push_frame(OutputAudioRawFrame(self._sound, SAMPLE_RATE, 1))


async def bot(runner_args: SmallWebRTCRunnerArguments):
    phone_line = os.environ.get("PEHLEAAP_PHONE_LINE", "1") != "0"
    logger.info(f"[pehleaap] turn models {SWITCH.names}, deciding: {SWITCH.active} (switch at /switch); "
                f"phone line: {'G.711 8 kHz' if phone_line else 'off'}")

    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            # No STT in the skeleton, so the turn model alone decides.
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=SwitchTurnAnalyzer(switch=SWITCH), wait_for_transcript=False)],
        )
    )
    pipeline = Pipeline(
        [
            transport.input(),
            PhoneLineProcessor(enabled=phone_line),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))),
            turns,
            DingOnTurnEnd(SWITCH),
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
