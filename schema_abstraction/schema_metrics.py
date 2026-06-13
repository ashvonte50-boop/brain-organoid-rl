import numpy as np
import torch

from compare_catastrophic_forgetting import DEVICE, CUE_STRENGTH, TEST_NOISE
import compare_catastrophic_forgetting as _ccf
from .schema_core import CENTROID_PROBE_STEPS, CENTROID_COSINE_EPS, CENTROID_INTERP_FACTOR


def probe_assembly_centroid(net, assemblies, n_steps=CENTROID_PROBE_STEPS):
    """Probe each assembly and return centroid vectors without altering network state.

    Uses I_syn[:_ccf.N_EXC] averaged over a fixed-length probe window.  Unlike
    probe_memory (which computes a differential score against background),
    this returns the raw per-neuron activation vector for pairwise distance
    and schema-convergence tracking.

    Network state (v, u, I_syn, traces, noise_std) is saved before probing
    and restored after, making this function a pure observer.
    """
    # ── Save network state ──
    _saved = dict(
        v=net.v.detach().clone(),
        u=net.u.detach().clone(),
        I_syn=net.I_syn.detach().clone(),
        spikes=net.spikes.detach().clone(),
        noise_std=net.noise_std,
    )
    if hasattr(net, "pre_trace") and net.pre_trace is not None:
        _saved["pre_trace"] = net.pre_trace.detach().clone()
        _saved["post_trace"] = net.post_trace.detach().clone()
    if hasattr(net, "stdp_enabled"):
        _saved["stdp_enabled"] = net.stdp_enabled

    # ── Probe ──
    centroids = []
    net.noise_std = TEST_NOISE
    net.reset_state()
    if net.stdp_enabled:
        net.pre_trace.zero_()
        net.post_trace.zero_()
    for asm in assemblies:
        centroid = np.zeros(_ccf.N_EXC, dtype=np.float32)
        cue = asm[:min(30, len(asm))]
        stim = torch.zeros(_ccf.N_NEURONS, device=DEVICE)
        stim[cue] = CUE_STRENGTH
        with torch.no_grad():
            for _ in range(n_steps):
                net.forward(stim)
                centroid += net.I_syn[:_ccf.N_EXC].cpu().numpy()
        centroid /= n_steps
        centroids.append(centroid)

    # ── Restore network state ──
    with torch.no_grad():
        net.v.copy_(_saved["v"])
        net.u.copy_(_saved["u"])
        net.I_syn.copy_(_saved["I_syn"])
        net.spikes.copy_(_saved["spikes"])
        net.noise_std = _saved["noise_std"]
        if "pre_trace" in _saved:
            net.pre_trace.copy_(_saved["pre_trace"])
            net.post_trace.copy_(_saved["post_trace"])
        if "stdp_enabled" in _saved:
            net.stdp_enabled = _saved["stdp_enabled"]

    return centroids


def pairwise_cosine_distances(centroids):
    n = len(centroids)
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ci = centroids[i].ravel()
            cj = centroids[j].ravel()
            d = 1.0 - float(np.dot(ci, cj) /
                (np.linalg.norm(ci) * np.linalg.norm(cj) + CENTROID_COSINE_EPS))
            dists[i, j] = dists[j, i] = d
    return dists


def update_pairwise_distances(snapshot, trajectory_store, assemblies=None):
    if snapshot is None:
        return
    centroids = snapshot["centroids"]
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            ci = centroids[i].ravel()
            cj = centroids[j].ravel()
            d = 1.0 - float(np.dot(ci, cj) /
                (np.linalg.norm(ci) * np.linalg.norm(cj) + CENTROID_COSINE_EPS))
            key = (i, j)
            if key not in trajectory_store:
                of = 0.0
                if assemblies is not None and i < len(assemblies) and j < len(assemblies):
                    from compare_catastrophic_forgetting import assembly_overlap_mask
                    shared, _, _ = assembly_overlap_mask(assemblies[i], assemblies[j])
                    of = len(shared) / max(len(assemblies[i]), len(assemblies[j]), 1)
                trajectory_store[key] = {"pair_dist": [], "overlap_frac": of}
            trajectory_store[key]["pair_dist"].append(d)


def update_schema_convergence(snapshot, overlap_pairs, convergence_store):
    if snapshot is None:
        return
    centroids = snapshot["centroids"]
    step = max((len(v) for v in convergence_store.values()), default=0)
    for i, j, shared in overlap_pairs:
        ci = centroids[i].ravel()
        cj = centroids[j].ravel()
        schema_centroid = CENTROID_INTERP_FACTOR * ci + (1.0 - CENTROID_INTERP_FACTOR) * cj
        d_i = 1.0 - float(np.dot(ci, schema_centroid) /
            (np.linalg.norm(ci) * np.linalg.norm(schema_centroid) + CENTROID_COSINE_EPS))
        d_j = 1.0 - float(np.dot(cj, schema_centroid) /
            (np.linalg.norm(cj) * np.linalg.norm(schema_centroid) + CENTROID_COSINE_EPS))
        key = (i, j)
        if key not in convergence_store:
            convergence_store[key] = []
        convergence_store[key].append({"dist_i_to_schema": d_i, "dist_j_to_schema": d_j, "step": step})
