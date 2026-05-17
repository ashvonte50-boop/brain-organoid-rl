"""
rl — reinforcement learning agents for closed-loop adaptive stimulation.

The RL agent observes multi-electrode array (MEA) signals from the SNN,
selects stimulation patterns, and receives reward based on memory performance.

Planned modules:
    agent           — Base agent interface and policy wrapper
    ppo_agent       — Proximal Policy Optimisation for continuous stimulation
    dqn_agent       — DQN for discrete stimulation channel selection
    reward_shaping  — Custom reward functions tied to network health metrics
    replay_buffer   — Experience replay for off-policy methods
"""

from .agent import BaseAgent, RandomStimulationAgent

__all__ = ["BaseAgent", "RandomStimulationAgent"]
