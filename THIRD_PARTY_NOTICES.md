# Third-party notices

## Code

**Smart Turn** ([pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn)). The model class in
`pehleaap/models/smart_turn.py` is vendored from its `train.py` (commit 4786657), and the int8 recipe in
`pehleaap/models/quantize.py` follows its `quantize_onnx_model`. Our released models (eot_v0, eot_v1) are fine-tunes
of the published Smart Turn v3.2 weights.

```
BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Data used to train or test the released models (none of it is redistributed here)

| Data | Licence | Used for |
|---|---|---|
| [pipecat-ai/smart-turn-data-v3.2-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) | CC BY 4.0 | training (English + Hindi clips) |
| [pipecat-ai/smart-turn-data-v3.2-test](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test) | CC BY 4.0 | testing |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs) (Hindi): Conneau et al., 2022 | CC BY 4.0 | train split for eot_v1's training; test + dev splits for testing |
| [MUSAN](https://www.openslr.org/17/) noise: Snyder, Chen and Povey, 2015 | CC BY 4.0 | phone-line augmentation (80% of the files) and test conditions (the other 20%) |
