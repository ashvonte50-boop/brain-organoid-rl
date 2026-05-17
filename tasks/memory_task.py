"""
Working Memory Task environment.

Structure:
    1. Encoding phase  — a spike-coded stimulus pattern is injected for `encode_ms`.
    2. Delay phase     — no external input; network must sustain activity for `delay_ms`.
    3. Probe phase     — a partial or noisy cue is presented; network must reproduce
                         the original pattern within `probe_ms`.

The reward is the normalized overlap between the recalled pattern and the target,
making this directly suitable for reward-modulated STDP and RL agents.

Future work:
    - Multi-item working memory (capacity limits)
    - Distractor stimuli during delay to test interference
    - Variable delay durations sampled from an exponential distribution
    - Integration with real organoid MEA recording geometry
"""

from __future__ import annotations

import numpy as np
import torch


class WorkingMemoryTask:
    """Minimal working-memory environment for SNN experiments.

    Args:
        n_neurons:     Size of the input / output population.
        n_patterns:    Number of orthogonal patterns to store.
        encode_ms:     Duration of encoding phase.
        delay_ms:      Duration of silent delay.
        probe_ms:      Duration of probe / readout phase.
        noise_rate:    Poisson noise rate (spikes / ms / neuron) during delay.
        dt:            Timestep in ms.
        device:        Torch device for spike tensors.
    """

    def __init__(
        self,
        n_neurons: int = 256,
        n_patterns: int = 8,
        encode_ms: float = 100.0,
        delay_ms: float = 500.0,
        probe_ms: float = 100.0,
        noise_rate: float = 0.002,
        dt: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.encode_steps = int(encode_ms / dt)
        self.delay_steps = int(delay_ms / dt)
        self.probe_steps = int(probe_ms / dt)
        self.total_steps = self.encode_steps + self.delay_steps + self.probe_steps
        self.noise_rate = noise_rate
        self.dt = dt
        self.device = device or torch.device("cpu")

        self._patterns = self._generate_patterns()
        self._current_pattern_idx: int | None = None
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Pattern generation
    # ------------------------------------------------------------------

    def _generate_patterns(self) -> torch.Tensor:
        """Generate `n_patterns` sparse binary patterns with ~20% active neurons."""
        sparsity = 0.2
        patterns = torch.zeros(self.n_patterns, self.n_neurons, device=self.device)
        for i in range(self.n_patterns):
            active = torch.randperm(self.n_neurons)[: int(sparsity * self.n_neurons)]
            patterns[i, active] = 1.0
        return patterns

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------

    def reset(self, pattern_idx: int | None = None) -> dict:
        """Reset the task and select a target pattern.

        Args:
            pattern_idx: Index of pattern to memorise. Sampled randomly if None.

        Returns:
            obs: Dict with 'spike_input' (T, n_neurons) and 'phase' label.
        """
        if pattern_idx is None:
            pattern_idx = int(np.random.randint(self.n_patterns))
        self._current_pattern_idx = pattern_idx
        self._step_counter = 0

        encoding_sequence = self._build_encoding_sequence(pattern_idx)
        return {"spike_input": encoding_sequence, "target_pattern_idx": pattern_idx}

    def step(self, network_output: torch.Tensor) -> tuple[dict, float, bool, dict]:
        """Evaluate a single timestep of network output.

        During the probe phase the overlap between `network_output` and the
        target pattern is computed and returned as reward.

        Args:
            network_output: (n_neurons,) spike tensor from the network readout.

        Returns:
            obs:    Next observation dict.
            reward: Float in [−1, 1] (overlap above / below chance).
            done:   True when the trial is complete.
            info:   Diagnostic dictionary.
        """
        self._step_counter += 1
        done = self._step_counter >= self.total_steps

        reward = 0.0
        if self._in_probe_phase():
            target = self._patterns[self._current_pattern_idx]
            overlap = self._pattern_overlap(network_output, target)
            reward = float(overlap.detach())

        noise = self._generate_noise()
        obs = {"spike_input": noise, "phase": self._current_phase()}
        return obs, reward, done, {"step": self._step_counter}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_encoding_sequence(self, pattern_idx: int) -> torch.Tensor:
        """Poisson spike train with rate proportional to the pattern for `encode_steps`."""
        pattern = self._patterns[pattern_idx]
        prob = pattern * 0.8 + self.noise_rate
        T = self.encode_steps
        rand = torch.rand(T, self.n_neurons, device=self.device)
        return (rand < prob).float()

    def _generate_noise(self) -> torch.Tensor:
        """Background Poisson noise for delay and probe phases."""
        rand = torch.rand(self.n_neurons, device=self.device)
        return (rand < self.noise_rate).float()

    def _pattern_overlap(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Normalised overlap ∈ [−1, 1] (like a cosine but for binary patterns)."""
        norm = torch.clamp(output.norm() * target.norm(), min=1e-8)
        return torch.dot(output.flatten(), target.flatten()) / norm

    def _in_probe_phase(self) -> bool:
        return self._step_counter > (self.encode_steps + self.delay_steps)

    def _current_phase(self) -> str:
        s = self._step_counter
        if s <= self.encode_steps:
            return "encoding"
        if s <= self.encode_steps + self.delay_steps:
            return "delay"
        return "probe"

    @property
    def observation_space(self) -> dict:
        return {"shape": (self.n_neurons,), "dtype": "binary"}

    @property
    def action_space(self) -> dict:
        return {"shape": (self.n_neurons,), "dtype": "binary"}
