"""Deterministic seeding utilities for reproducible experiments."""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
