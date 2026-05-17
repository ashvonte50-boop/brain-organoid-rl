"""
synapses — synapse models and structural connectivity.

Planned modules:
    synapse_models   — conductance-based, current-based, and short-term plasticity
    connectivity     — random, small-world, scale-free, and organoid-inspired wiring
    neuromodulation  — dopamine / acetylcholine gating of synaptic gain
"""

from .synapse_models import CurrentSynapse, ConductanceSynapse, STPSynapse

__all__ = ["CurrentSynapse", "ConductanceSynapse", "STPSynapse"]
