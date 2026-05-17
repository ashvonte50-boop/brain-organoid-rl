"""
Spike-train analysis metrics.

All functions accept (T, N) binary spike tensors or numpy arrays.

compute_firing_rate   — Mean firing rate per neuron (Hz).
compute_synchrony     — Pairwise synchrony index (van Rossum / SPIKE distance).
compute_isi_stats     — Inter-spike interval mean and coefficient of variation.
compute_fano_factor   — Fano factor (variance / mean spike count) per neuron.
compute_population_coupling — Correlation of each neuron with population rate.

Future work:
    - Spectral analysis (LFP proxy from summed spikes)
    - Dimensionality reduction: PCA / UMAP on population vectors
    - Avalanche exponent fitting for criticality analysis
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


def compute_firing_rate(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """Mean firing rate per neuron in Hz.

    Args:
        spike_train: (T, N) binary array.
        dt:          Timestep in ms.

    Returns:
        rates: (N,) array of firing rates in Hz.
    """
    spikes = _to_numpy(spike_train)
    T = spikes.shape[0]
    return spikes.sum(axis=0) / (T * dt * 1e-3)


def compute_synchrony(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    window_ms: float = 5.0,
) -> float:
    """Population synchrony index via cross-correlation at zero lag.

    Computes the mean pairwise Pearson correlation of smoothed spike trains.
    Range [0, 1] where 1 = fully synchronised, 0 = independent.

    Args:
        spike_train: (T, N) binary array.
        dt:          Timestep in ms.
        window_ms:   Gaussian smoothing window (ms) before correlation.

    Returns:
        synchrony: Scalar synchrony index.
    """
    spikes = _to_numpy(spike_train).T  # (N, T)
    N, T = spikes.shape

    if N < 2:
        return 0.0

    win = max(1, int(window_ms / dt))
    kernel = np.ones(win) / win
    smoothed = np.array([np.convolve(spikes[n], kernel, mode="same") for n in range(N)])

    # Correlation matrix
    corr = np.corrcoef(smoothed)
    upper_tri = corr[np.triu_indices(N, k=1)]
    return float(np.nanmean(upper_tri))


def compute_isi_stats(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Inter-spike interval mean and coefficient of variation per neuron.

    Args:
        spike_train: (T, N) binary array.
        dt:          Timestep in ms.

    Returns:
        Dict with 'mean_isi' and 'cv_isi', each (N,). NaN for silent neurons.
    """
    spikes = _to_numpy(spike_train).T  # (N, T)
    N = spikes.shape[0]
    mean_isi = np.full(N, np.nan)
    cv_isi = np.full(N, np.nan)

    for n in range(N):
        spike_times = np.where(spikes[n] > 0)[0] * dt
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            mean_isi[n] = isis.mean()
            cv_isi[n] = isis.std() / mean_isi[n] if mean_isi[n] > 0 else np.nan

    return {"mean_isi": mean_isi, "cv_isi": cv_isi}


def compute_fano_factor(
    spike_train: torch.Tensor | np.ndarray,
    bin_ms: float = 50.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Fano factor (var / mean spike count) per neuron across time bins.

    Args:
        spike_train: (T, N) binary array.
        bin_ms:      Width of counting window in ms.
        dt:          Timestep in ms.

    Returns:
        fano: (N,) Fano factors. NaN for neurons with zero mean count.
    """
    spikes = _to_numpy(spike_train)
    T, N = spikes.shape
    bin_steps = max(1, int(bin_ms / dt))
    n_bins = T // bin_steps
    binned = spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, N).sum(axis=1)

    mean_count = binned.mean(axis=0)
    var_count = binned.var(axis=0)
    fano = np.where(mean_count > 0, var_count / mean_count, np.nan)
    return fano


def compute_population_coupling(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    window_ms: float = 10.0,
) -> np.ndarray:
    """Correlation of each neuron's spike train with the population mean rate.

    High coupling indicates that a neuron tracks the collective network state.

    Args:
        spike_train: (T, N) binary array.
        dt:          Timestep in ms.
        window_ms:   Smoothing window for rate estimation.

    Returns:
        coupling: (N,) Pearson correlation coefficients.
    """
    spikes = _to_numpy(spike_train)
    T, N = spikes.shape
    win = max(1, int(window_ms / dt))
    kernel = np.ones(win) / win

    smoothed = np.array(
        [np.convolve(spikes[:, n], kernel, mode="same") for n in range(N)]
    )  # (N, T)

    population_rate = smoothed.mean(axis=0)  # (T,)
    coupling = np.array(
        [np.corrcoef(smoothed[n], population_rate)[0, 1] for n in range(N)]
    )
    return coupling
