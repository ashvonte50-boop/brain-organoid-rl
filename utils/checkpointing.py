"""Checkpoint save / load helpers."""

from __future__ import annotations
from pathlib import Path
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    network: nn.Module,
    plasticity: nn.Module,
    trial: int,
    extra: dict | None = None,
) -> None:
    payload = {
        "trial": trial,
        "network_state": network.state_dict(),
        "plasticity_state": plasticity.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info("Checkpoint saved -> %s", path)


def load_checkpoint(
    path: str | Path,
    network: nn.Module,
    plasticity: nn.Module,
) -> int:
    """Load checkpoint in-place. Returns the saved trial number."""
    payload = torch.load(path, map_location="cpu")
    network.load_state_dict(payload["network_state"])
    plasticity.load_state_dict(payload["plasticity_state"])
    trial = payload.get("trial", 0)
    log.info("Checkpoint loaded <- %s (trial %d)", path, trial)
    return trial
