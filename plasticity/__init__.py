"""
plasticity — learning rules operating on spike timing and network activity.

Planned modules:
    stdp             — Spike-Timing-Dependent Plasticity (pair, triplet, voltage-based)
    homeostatic      — Synaptic scaling and intrinsic excitability regulation
    neuromodulatory  — Reward-modulated STDP (dopamine) for RL integration
    bcm              — Bienenstock-Cooper-Munro rate-based rule
"""

from .stdp import PairSTDP, RewardModulatedSTDP

__all__ = ["PairSTDP", "RewardModulatedSTDP"]
