"""Put the caller on a simulated phone line before VAD and turn detection hear them."""

import numpy as np
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pehleaap.telephony.realtime import PhoneLine


class PhoneLineProcessor(FrameProcessor):
    """8 kHz G.711 mu-law channel on every mono InputAudioRawFrame (pipeline rate: a multiple of 8 kHz).

    Place it right after `transport.input()`. Adds ~6 ms of delay and keeps frame sizes unchanged.
    """

    def __init__(self, *, enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self._lines: dict[int, PhoneLine] = {}  # one stateful line per sample rate

    def degrade(self, frame: InputAudioRawFrame) -> None:
        if frame.num_channels != 1:
            raise ValueError("PhoneLineProcessor expects mono input audio")
        line = self._lines.get(frame.sample_rate)
        if line is None:
            line = self._lines[frame.sample_rate] = PhoneLine(frame.sample_rate)
        frame.audio = line.process(np.frombuffer(frame.audio, dtype=np.int16)).tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.enabled and isinstance(frame, InputAudioRawFrame):
            self.degrade(frame)
        await self.push_frame(frame, direction)
