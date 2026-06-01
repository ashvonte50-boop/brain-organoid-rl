"""Core data structures and hook wiring for schema-abstraction analysis.

The hooks attach schema data directly to the net object via ``net._hook_extra``.
This ensures data survives multiprocessing (the worker's net gets serialized,
the parent reads _hook_extra from the returned result).

All nine theoretical elements are registered here:
  1. Centroid tracking (centroid snapshots, pairwise distances, schema convergence)
  2. Synaptic downscaling (global weakening + replay protection)
  3. Generative cortical layer (VAE)
  4. Generalization probe (schema-consistent novel inputs)
  5. Anti-prediction test (high-fidelity vs natural generalisation)
  6. Metaplasticity (BCM sliding thresholds)
  7. Hidden-state tracking (consolidation status per memory)
  8. Structured forgetting analysis (basin geometry)
  9. Bayesian filtering framework for reverse replay
"""


# ── Configuration ───────────────────────────────────────────────────────
try:
    from compare_catastrophic_forgetting import DEV_MODE as _DEV
except ImportError:
    _DEV = False
CENTROID_PROBE_STEPS    = 10 if _DEV else 50
CENTROID_COSINE_EPS     = 1e-10
CENTROID_INTERP_FACTOR  = 0.30

# Feature flags — set False to disable specific elements
ENABLE_DOWNSCALING      = True
ENABLE_GENERATIVE_LAYER = True
ENABLE_PROBES           = True
ENABLE_METAPLASTICITY   = True
ENABLE_HIDDEN_STATES    = True

# Schema core mask (set by schema_experiments.py for hierarchical architecture)
_SCHEMA_CORE_MASK = None


# ── Assembly utilities ─────────────────────────────────────────────────

def _overlap_mask(asm_a, asm_b):
    import numpy as np
    a = np.asarray(asm_a, dtype=np.int64)
    b = np.asarray(asm_b, dtype=np.int64)
    shared = np.intersect1d(a, b)
    spec_a = np.setdiff1d(a, shared)
    spec_b = np.setdiff1d(b, shared)
    return shared, spec_a, spec_b


def _overlap_pairs(assemblies):
    pairs = []
    n = len(assemblies)
    for i in range(n):
        for j in range(i + 1, n):
            shared, _, _ = _overlap_mask(assemblies[i], assemblies[j])
            if len(shared) > 0:
                pairs.append((i, j, shared))
    return pairs


# ── Hook callbacks ─────────────────────────────────────────────────────

def _baseline_hook(net, assemblies, n_mem, j=-1, **_):
    # Pass schema core mask to net for probes to access
    if _SCHEMA_CORE_MASK is not None:
        net._schema_core_mask = _SCHEMA_CORE_MASK

    from .schema_metrics import probe_assembly_centroid, update_pairwise_distances, update_schema_convergence
    centroids = probe_assembly_centroid(net, assemblies)
    traj = {}
    conv = {}
    snap = {"centroids": centroids, "label": "baseline"}
    update_pairwise_distances(snap, traj, assemblies)
    update_schema_convergence(snap, _overlap_pairs(assemblies), conv)
    net._hook_extra = {
        "centroid_snapshots": [snap],
        "distance_trajectories": traj,
        "schema_convergence": conv,
    }
    if ENABLE_DOWNSCALING:
        from .schema_downscaling import _downscale_pre_replay_hook
        _downscale_pre_replay_hook(net, assemblies, n_mem, -1)
    if ENABLE_GENERATIVE_LAYER:
        from .schema_generative import _gen_baseline_hook
        _gen_baseline_hook(net, assemblies, n_mem, -1)
    if ENABLE_METAPLASTICITY or ENABLE_HIDDEN_STATES:
        from .schema_metaplasticity import _meta_baseline_hook
        _meta_baseline_hook(net, assemblies, n_mem, -1)


def _encode_hook(net, assemblies, n_mem, j, **_):
    extra = getattr(net, "_hook_extra", None)
    if extra is None:
        return
    from .schema_metrics import probe_assembly_centroid, update_pairwise_distances, update_schema_convergence
    centroids = probe_assembly_centroid(net, assemblies)
    snap = {"centroids": centroids, "label": f"post_encode_{j}"}
    extra["centroid_snapshots"].append(snap)
    update_pairwise_distances(snap, extra["distance_trajectories"], assemblies)
    update_schema_convergence(snap, _overlap_pairs(assemblies), extra["schema_convergence"])
    if ENABLE_METAPLASTICITY or ENABLE_HIDDEN_STATES:
        from .schema_metaplasticity import _meta_post_encode_hook
        _meta_post_encode_hook(net, assemblies, n_mem, j)


def _replay_hook(net, assemblies, n_mem, j, **_):
    extra = getattr(net, "_hook_extra", None)
    if extra is None:
        return
    from .schema_metrics import probe_assembly_centroid, update_pairwise_distances, update_schema_convergence
    centroids = probe_assembly_centroid(net, assemblies)
    snap = {"centroids": centroids, "label": f"post_replay_{j}"}
    extra["centroid_snapshots"].append(snap)
    update_pairwise_distances(snap, extra["distance_trajectories"], assemblies)
    update_schema_convergence(snap, _overlap_pairs(assemblies), extra["schema_convergence"])
    if ENABLE_DOWNSCALING:
        from .schema_downscaling import _downscale_post_replay_hook
        _downscale_post_replay_hook(net, assemblies, n_mem, j)
    if ENABLE_GENERATIVE_LAYER:
        from .schema_generative import _gen_post_replay_hook
        _gen_post_replay_hook(net, assemblies, n_mem, j)
    if ENABLE_METAPLASTICITY or ENABLE_HIDDEN_STATES:
        from .schema_metaplasticity import _meta_replay_hook
        _meta_replay_hook(net, assemblies, n_mem, j)


def _final_hook(net, assemblies, n_mem, **_):
    extra = getattr(net, "_hook_extra", None)
    if extra is None:
        return
    from .schema_metrics import probe_assembly_centroid, update_pairwise_distances, update_schema_convergence
    centroids = probe_assembly_centroid(net, assemblies)
    snap = {"centroids": centroids, "label": "final"}
    extra["centroid_snapshots"].append(snap)
    update_pairwise_distances(snap, extra["distance_trajectories"], assemblies)
    update_schema_convergence(snap, _overlap_pairs(assemblies), extra["schema_convergence"])
    if ENABLE_DOWNSCALING:
        from .schema_downscaling import _downscale_final_hook
        _downscale_final_hook(net, assemblies, n_mem)
    if ENABLE_GENERATIVE_LAYER:
        from .schema_generative import _gen_final_hook
        _gen_final_hook(net, assemblies, n_mem)
    if ENABLE_PROBES:
        from .schema_probes import _probes_final_hook
        _probes_final_hook(net, assemblies, n_mem)
    if ENABLE_METAPLASTICITY or ENABLE_HIDDEN_STATES:
        from .schema_metaplasticity import _meta_final_hook
        _meta_final_hook(net, assemblies, n_mem)


def register_schema_hooks():
    from compare_catastrophic_forgetting import register_hook
    register_hook("baseline", _baseline_hook)
    register_hook("post_encode", _encode_hook)
    register_hook("post_replay", _replay_hook)
    register_hook("final", _final_hook)
