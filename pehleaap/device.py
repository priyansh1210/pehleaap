"""Which floating-point precision to run a model in, per machine."""

import torch


def precision_for(device: torch.device, setting: str = "auto") -> str:
    """bf16 on Ampere and newer (the GPU box's Ada), fp16 on older GPUs (Kaggle's T4), fp32 on the CPU.

    GTX 16xx cards (the Dev box's GTX 1650) get fp32: their fp16 convolutions return NaN for Smart
    Turn from the first layer on (checked 2026-10-24), a known quirk of those cards.
    """
    if setting != "auto":
        return setting
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.get_device_capability(device) >= (8, 0):
        return "bf16"
    return "fp32" if "GTX 16" in torch.cuda.get_device_name(device) else "fp16"


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
