"""Device selection. Order: explicit arg > CC_DEVICE env > cuda > mps > cpu.

NOTE: this box is an Apple-Silicon Mac -- one integrated GPU via MPS, NOT a multi-GPU CUDA
host. So "parallel on GPU" means a single shared MPS device; multi-process CPU across the 14
cores is a legitimate (often faster) parallelization for the tiny full-batch MLPs here. We pick
the device by MEASUREMENT (see bench_device.py), not assumption.
"""
from __future__ import annotations

import os
import torch


def get_device(prefer=None):
    if prefer is not None:
        return torch.device(prefer)
    env = os.environ.get("CC_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
