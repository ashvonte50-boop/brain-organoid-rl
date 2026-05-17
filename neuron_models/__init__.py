"""
neuron_models — biologically constrained single-neuron and population models.

Planned modules:
    lif_neuron     — Leaky Integrate-and-Fire (Norse LIFCell wrapper)
    adex_neuron    — Adaptive Exponential I&F for burst/adaptation dynamics
    izhikevich     — Izhikevich model for rich bifurcation phenomenology
    population     — Structured ensembles (excitatory / inhibitory columns)
"""

from .lif_neuron import LIFLayer, LIFPopulation

__all__ = ["LIFLayer", "LIFPopulation"]
