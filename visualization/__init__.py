"""
visualization — plotting utilities for spike trains, weights, and network topology.

Planned modules:
    spike_plots     — Raster plots, PSTH, inter-spike interval distributions
    weight_plots    — Weight matrix heatmaps and weight evolution over training
    network_graphs  — Connectivity graphs coloured by neuron type / layer
    activity_maps   — Spatial activity maps for organoid-geometry simulations
"""

from .spike_plots import raster_plot, psth_plot, firing_rate_heatmap

__all__ = ["raster_plot", "psth_plot", "firing_rate_heatmap"]
