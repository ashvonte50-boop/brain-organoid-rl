"""
RL agent definitions for closed-loop cortical stimulation.

BaseAgent              — Abstract interface all agents must implement.
RandomStimulationAgent — Random baseline: selects channels and intensities at random.

Observation space: firing rate vector of the SNN readout population (n_readout,).
Action space:      Stimulation pattern vector (n_channels,) in [0, 1].
Reward:            Memory overlap score from the task environment.

Future work:
    - PPO agent with LSTM policy for temporal dependencies
    - Model-based planning using a learned SNN dynamics model
    - Multi-objective optimisation: memory performance vs. network metabolic cost
    - Safe RL constraints to avoid epileptiform activity patterns
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseAgent(abc.ABC):
    """Abstract base for all stimulation RL agents."""

    @abc.abstractmethod
    def select_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Choose a stimulation action given the current observation.

        Args:
            observation:  Network readout state (n_readout,).
            deterministic: Use greedy / mean action for evaluation.

        Returns:
            action: Stimulation pattern (n_channels,) in [0, 1].
        """

    @abc.abstractmethod
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update the agent's parameters from a batch of experience.

        Args:
            batch: Dict with keys 'obs', 'action', 'reward', 'next_obs', 'done'.

        Returns:
            Metrics dict (e.g., {'loss': 0.23, 'entropy': 1.1}).
        """

    def save(self, path: str) -> None:
        """Persist agent weights to disk."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent weights from disk."""
        raise NotImplementedError


class RandomStimulationAgent(BaseAgent):
    """Baseline agent that samples stimulation patterns uniformly at random.

    Useful for establishing a performance floor before training learned policies.

    Args:
        n_channels:   Number of stimulation electrodes.
        intensity_max: Maximum stimulation amplitude (normalised to [0, 1]).
        sparse:       If True, only activate `sparsity` fraction of channels.
        sparsity:     Fraction of channels active per timestep when sparse=True.
    """

    def __init__(
        self,
        n_channels: int,
        intensity_max: float = 1.0,
        sparse: bool = True,
        sparsity: float = 0.1,
    ) -> None:
        self.n_channels = n_channels
        self.intensity_max = intensity_max
        self.sparse = sparse
        self.sparsity = sparsity

    def select_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        action = torch.rand(self.n_channels) * self.intensity_max
        if self.sparse:
            mask = torch.zeros(self.n_channels)
            n_active = max(1, int(self.sparsity * self.n_channels))
            active_idx = torch.randperm(self.n_channels)[:n_active]
            mask[active_idx] = 1.0
            action = action * mask
        return action

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        # No learning in the random baseline
        return {"loss": 0.0}


class ActorCriticNetwork(nn.Module):
    """Shared feature extractor with separate actor and critic heads.

    Intended for use in PPO or A2C training. Accepts firing rate observations
    (continuous) and outputs a continuous stimulation action via a Gaussian policy.

    Args:
        obs_dim:    Dimensionality of the observation (readout population size).
        action_dim: Number of stimulation channels.
        hidden_dim: Width of shared MLP layers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_mean: (batch, action_dim) policy mean.
            log_std:     (action_dim,) log standard deviation.
            value:       (batch, 1) state value estimate.
        """
        features = self.shared(obs)
        return self.actor_mean(features), self.actor_log_std, self.critic(features)
