"""
Reproducibility utilities — ensures identical results across runs.

In medical ML, reproducibility isn't optional. If a regulator asks
"can you reproduce these results?", the answer must be yes.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    This covers:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU and CUDA generators
    - CUDA deterministic algorithms (trades speed for reproducibility)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Force deterministic algorithms — slightly slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ deterministic mode
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            # Fall back gracefully
            pass


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority: CUDA GPU > Apple MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slower")
    return device
