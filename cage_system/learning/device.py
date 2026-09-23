import random

import numpy as np
import torch


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda without silently ignoring an explicit CUDA request."""

    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python process cannot access a CUDA device. "
            "Check the driver, /dev/nvidia* mapping, and the active camera environment."
        )
    return device


def seed_learning(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

