"""
utils — shared helpers used across modules.

Planned modules:
    metrics        — Spike-train analysis: firing rate, synchrony, entropy, ISI stats
    seed           — Deterministic seeding of Python, NumPy, and PyTorch RNGs
    checkpointing  — Save / load experiment checkpoints
    logging_setup  — Configure Python logging with optional file handler
    data_io        — HDF5 / NumPy data loading helpers for MEA recordings
"""

from .metrics import compute_firing_rate, compute_synchrony
from .seed import set_global_seed

__all__ = ["compute_firing_rate", "compute_synchrony", "set_global_seed"]
