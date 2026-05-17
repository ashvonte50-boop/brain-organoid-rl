"""
Spike train visualisation utilities.

raster_plot          — Classic dot-raster (neuron × time).
psth_plot            — Peri-stimulus time histogram.
firing_rate_heatmap  — 2-D heatmap of population firing rate over time.

All functions accept numpy arrays or torch tensors and return Matplotlib figures
so callers control whether to save, display, or log to TensorBoard.

Future work:
    - Animated rasters for real-time monitoring during training
    - Phase-plane portraits for single-neuron state trajectories
    - 3-D organoid geometry rendering with activity overlaid
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def raster_plot(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    neuron_subset: int | None = None,
    title: str = "Spike Raster",
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot a spike raster diagram.

    Args:
        spike_train:    Binary array of shape (T, N) — time × neurons.
        dt:             Timestep in ms (used to label x-axis in ms).
        neuron_subset:  Only plot the first `neuron_subset` neurons if provided.
        title:          Figure title.
        figsize:        Matplotlib figure size.

    Returns:
        fig: Matplotlib Figure ready for saving or TensorBoard logging.
    """
    spikes = _to_numpy(spike_train)
    T, N = spikes.shape

    if neuron_subset is not None:
        spikes = spikes[:, :neuron_subset]
        N = neuron_subset

    times, neurons = np.where(spikes > 0)
    time_ms = times * dt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(time_ms, neurons, s=1.0, c="black", alpha=0.6, rasterized=True)
    ax.set_xlim(0, T * dt)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def psth_plot(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    bin_ms: float = 10.0,
    title: str = "PSTH",
    figsize: tuple[float, float] = (12, 3),
) -> plt.Figure:
    """Peri-stimulus time histogram — population-averaged firing rate over time.

    Args:
        spike_train: Binary (T, N) array.
        dt:          Simulation timestep (ms).
        bin_ms:      Width of histogram bins in ms.
        title:       Figure title.
        figsize:     Matplotlib figure size.

    Returns:
        fig: Matplotlib Figure.
    """
    spikes = _to_numpy(spike_train)
    T, N = spikes.shape
    bin_steps = max(1, int(bin_ms / dt))

    n_bins = T // bin_steps
    rate = spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, N).sum(axis=(1, 2))
    rate = rate / (N * bin_steps * dt * 1e-3)  # convert to Hz

    time_axis = np.arange(n_bins) * bin_ms

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(time_axis, rate, width=bin_ms * 0.9, color="steelblue", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def firing_rate_heatmap(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    window_ms: float = 50.0,
    title: str = "Firing Rate Heatmap",
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """2-D heatmap of smoothed firing rates (neuron × time window).

    Args:
        spike_train: Binary (T, N) array.
        dt:          Simulation timestep (ms).
        window_ms:   Smoothing window width in ms.
        title:       Figure title.
        figsize:     Matplotlib figure size.

    Returns:
        fig: Matplotlib Figure.
    """
    spikes = _to_numpy(spike_train).T  # (N, T)
    N, T = spikes.shape
    win = max(1, int(window_ms / dt))

    kernel = np.ones(win) / win
    smoothed = np.array([np.convolve(spikes[n], kernel, mode="same") for n in range(N)])
    smoothed = smoothed / (dt * 1e-3)  # Hz

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        smoothed,
        aspect="auto",
        origin="lower",
        extent=[0, T * dt, 0, N],
        cmap="hot",
    )
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    fig.tight_layout()
    return fig
