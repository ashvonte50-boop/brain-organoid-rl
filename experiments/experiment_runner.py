"""
ExperimentRunner — assembles all components and executes the training loop.

Responsibilities:
    - Instantiate the SNN, plasticity rules, task, and RL agent from config
    - Run the trial loop: encode → delay → probe → reward → weight update
    - Log metrics to TensorBoard and optionally WandB
    - Save checkpoints at configured intervals
    - Restore from checkpoint for experiment resumption

Future work:
    - Distributed training across GPUs (torch.distributed)
    - Early stopping based on validation memory overlap
    - Curriculum scheduling of task difficulty
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from neuron_models import LIFPopulation
from tasks import WorkingMemoryTask
from rl import RandomStimulationAgent
from plasticity import RewardModulatedSTDP
from utils.metrics import compute_firing_rate, compute_synchrony
from utils.checkpointing import save_checkpoint, load_checkpoint

log = logging.getLogger(__name__)


class ExperimentRunner:
    """Main training loop orchestrator.

    Args:
        config: Parsed YAML configuration dictionary.
        device: Torch compute device.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self._build_components()
        self._setup_logging()

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        nc = self.config.get("network", {})
        tc = self.config.get("task", {})
        rc = self.config.get("rl", {})
        pc = self.config.get("plasticity", {})

        layer_sizes = nc.get("layer_sizes", [128, 512, 256, 128])
        self.network = LIFPopulation(
            layer_sizes=layer_sizes,
            dt=nc.get("dt", 1.0),
        ).to(self.device)

        self.task = WorkingMemoryTask(
            n_neurons=layer_sizes[0],
            encode_ms=tc.get("encode_ms", 100.0),
            delay_ms=tc.get("delay_ms", 500.0),
            probe_ms=tc.get("probe_ms", 100.0),
            dt=nc.get("dt", 1.0),
            device=self.device,
        )

        self.agent = RandomStimulationAgent(
            n_channels=rc.get("n_channels", 32),
            sparsity=rc.get("sparsity", 0.1),
        )

        self.plasticity = RewardModulatedSTDP(
            n_pre=layer_sizes[-2],
            n_post=layer_sizes[-1],
            eta=pc.get("eta", 1e-3),
        ).to(self.device)

    def _setup_logging(self) -> None:
        lc = self.config.get("logging", {})
        run_name = lc.get("run_name", f"run_{int(time.time())}")
        log_dir = Path(lc.get("log_dir", "logs")) / run_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.checkpoint_dir = Path(self.config.get("training", {}).get("checkpoint_dir", "checkpoints")) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info("Logging to %s", log_dir)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        tc = self.config.get("training", {})
        n_trials = tc.get("n_trials", 1000)
        checkpoint_every = tc.get("checkpoint_every", 100)
        log_every = tc.get("log_every", 10)

        log.info("Starting training for %d trials.", n_trials)
        episode_rewards = []

        for trial in range(1, n_trials + 1):
            reward = self._run_trial()
            episode_rewards.append(reward)

            if trial % log_every == 0:
                mean_reward = sum(episode_rewards[-log_every:]) / log_every
                self.writer.add_scalar("reward/mean", mean_reward, trial)
                log.info("Trial %4d | mean reward: %.4f", trial, mean_reward)

            if trial % checkpoint_every == 0:
                save_checkpoint(
                    path=self.checkpoint_dir / f"ckpt_trial_{trial}.pt",
                    network=self.network,
                    plasticity=self.plasticity,
                    trial=trial,
                )

        self.writer.close()
        log.info("Training complete.")

    def _run_trial(self) -> float:
        """Execute a single encode → delay → probe trial.

        Returns:
            total_reward: Cumulative reward accumulated during the probe phase.
        """
        obs = self.task.reset()
        states = None
        total_reward = 0.0
        done = False

        trace_pre, trace_post, eligibility = self.plasticity.init_states(
            batch=1, device=self.device
        )

        while not done:
            spike_input = obs.get("spike_input")
            if isinstance(spike_input, torch.Tensor) and spike_input.dim() == 2:
                x = spike_input[0].unsqueeze(0)
            else:
                x = spike_input if spike_input is not None else torch.zeros(1, self.task.n_neurons, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            output, states = self.network(x.to(self.device), states)

            readout_rate = output.float().mean(dim=0)
            action = self.agent.select_action(readout_rate)

            obs, reward, done, info = self.task.step(output.squeeze(0))
            total_reward += reward

            # Apply reward-modulated weight update on the final layer.
            # states[i] is (LIFFeedForwardState, prev_spikes); index [1] gives spikes.
            pre_sp = states[-2][1] if states[-2] is not None else torch.zeros_like(output)
            post_sp = output
            _, trace_pre, trace_post, eligibility = self.plasticity(
                pre_spikes=pre_sp,
                post_spikes=post_sp,
                trace_pre=trace_pre,
                trace_post=trace_post,
                eligibility=eligibility,
                reward_signal=reward,
            )

        return total_reward
