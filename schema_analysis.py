"""Schema Analysis: centroid tracking, directional drift, and schema formation metrics.

Uses weight-vector snapshots from the experiment to measure how memories
move toward shared latent structure during consolidation.
"""
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import percentileofscore
import warnings


# ── Helpers ──────────────────────────────────────────────────────────

def _valid_snapshots(snapshots, n_mem=4):
    """Return list-of-lists of valid snapshot vectors."""
    out = []
    for i in range(n_mem):
        row = []
        for j in range(n_mem):
            v = None
            if i < len(snapshots) and j < len(snapshots[i]):
                v = snapshots[i][j]
            if v is not None and np.all(np.isfinite(v)):
                row.append(v.astype(np.float64))
            else:
                row.append(None)
        out.append(row)
    return out


def _pairwise_cosine(mem_i, mem_j):
    """Cosine distance between two memory vectors at same checkpoint.
    Returns nan if either is None.
    """
    if mem_i is None or mem_j is None:
        return np.nan
    try:
        return float(cosine_dist(mem_i, mem_j))
    except Exception:
        return np.nan


# ── M1 + M2: Memory & Schema Centroids ──────────────────────────────

def compute_centroids(snapshots, n_mem=4):
    """Compute memory centroids and schema centroid at each checkpoint.

    Returns
    -------
    centroids : dict (i, j) -> vector  (memory i at checkpoint j)
    schema_cents : dict j -> vector     (mean across memories at checkpoint j)
    distances : dict (i, j) -> float    (cosine distance from memory i to schema centroid j)
    """
    snap = _valid_snapshots(snapshots, n_mem)
    centroids = {}
    schema_cents = {}
    distances = {}

    for j in range(n_mem):
        vecs = []
        idxs = []
        for i in range(n_mem):
            v = snap[i][j]
            if v is not None:
                centroids[(i, j)] = v
                vecs.append(v)
                idxs.append(i)
        if len(vecs) >= 2:
            sc = np.mean(vecs, axis=0)
            schema_cents[j] = sc
            for i in idxs:
                v = snap[i][j]
                if v is not None:
                    distances[(i, j)] = _pairwise_cosine(v, sc)
        elif len(vecs) == 1:
            schema_cents[j] = vecs[0]
            distances[(idxs[0], j)] = 0.0
    return centroids, schema_cents, distances


# ── M3: Directional Drift ───────────────────────────────────────────

def compute_drift(distances, n_mem=4):
    """Compute drift toward schema centroid over time.
    Uses first and last available checkpoint for each memory.
    Negative = moving toward schema (convergence).
    """
    drift = {}
    for i in range(n_mem):
        first_d = last_d = None
        for j in range(n_mem):
            d = distances.get((i, j))
            if d is not None:
                if first_d is None:
                    first_d = d
                last_d = d
        if first_d is not None and last_d is not None:
            drift[i] = float(last_d - first_d)
    return drift


def compute_schema_convergence(distances, n_mem=4):
    """Schema convergence index: positive = memories converging toward centroid."""
    d_initial, d_final = [], []
    for i in range(n_mem):
        first = last = None
        for j in range(n_mem):
            d = distances.get((i, j))
            if d is not None:
                if first is None:
                    first = d
                last = d
        if first is not None and last is not None:
            d_initial.append(first)
            d_final.append(last)
    if not d_initial or not d_final:
        return 0.0
    return float(np.mean(d_initial) - np.mean(d_final))


# ── M4: Random vs Directional Drift (Monte Carlo) ───────────────────

def permutation_test(distances, n_mem=4, n_shuffles=1000, seed=42):
    """Test if drift is more directional than chance.

    Uses first and last available checkpoint for each memory.
    Only includes memories where first != last (at least 1 step between).

    Returns
    -------
    actual_drift : float    (mean drift across memories)
    p_value : float          (two-sided, 0=strongly directional, 1=random)
    shuffle_drifts : list    (all shuffle drift values, for histogram)
    """
    rng = np.random.RandomState(seed)

    # For each memory, find distances at first and last checkpoint
    mem_d0 = {}  # first distance
    mem_d1 = {}  # last distance
    for i in range(n_mem):
        first_j = last_j = None
        first_d = last_d = None
        for j in range(n_mem):
            d = distances.get((i, j))
            if d is not None:
                if first_j is None:
                    first_j = j
                    first_d = d
                last_j = j
                last_d = d
        if first_j is not None and last_j is not None and first_j < last_j:
            mem_d0[i] = first_d
            mem_d1[i] = last_d

    if not mem_d0:
        return 0.0, 1.0, []

    # Actual per-memory drifts
    mem_ids = sorted(mem_d0.keys())
    per_memory_drifts = np.array([mem_d1[i] - mem_d0[i] for i in mem_ids])
    actual_drift = float(np.mean(per_memory_drifts))

    # Sign-flip null: randomly flip the sign of each memory's drift.
    # Tests whether the observed directional consistency is beyond chance.
    abs_drifts = np.abs(per_memory_drifts)
    shuffle_drifts = []
    for _ in range(n_shuffles):
        signs = rng.choice([-1, 1], size=len(per_memory_drifts))
        shuffled = float(np.mean(abs_drifts * signs))
        shuffle_drifts.append(shuffled)

    # One-sided p-value: P(null <= actual) — negative drift = converging
    p_left = percentileofscore(shuffle_drifts, actual_drift, kind='mean') / 100.0
    p = 2.0 * min(p_left, 1.0 - p_left)  # two-sided
    return actual_drift, float(p), shuffle_drifts


# ── Schema Score ─────────────────────────────────────────────────────

def compute_schema_score(snapshots, n_mem=4):
    """Composite schema formation score ∈ [0,1].

    Handles triangular snapshot structure:
    memory i has valid vectors starting at checkpoint i (its creation time).
    init_i = first vector of memory i (at checkpoint i)
    final_i = last vector of memory i (at checkpoint n_mem-1)
    Score > 0 means memories are closer at end than at their initial states.
    """
    snap = _valid_snapshots(snapshots, n_mem)

    # Get each memory's first and last vector
    inits = {}
    finals = {}
    for i in range(n_mem):
        for j in range(n_mem):
            if snap[i][j] is not None:
                inits[i] = snap[i][j]
                break
        finals[i] = snap[i][n_mem - 1]

    if len(inits) < 2:
        return 0.0

    init_dists = []
    final_dists = []
    for i in range(n_mem):
        for k in range(i + 1, n_mem):
            if i in inits and k in inits:
                d_init = _pairwise_cosine(inits[i], inits[k])
                if not np.isnan(d_init):
                    init_dists.append(d_init)
            if i in finals and k in finals:
                d_final = _pairwise_cosine(finals[i], finals[k])
                if not np.isnan(d_final):
                    final_dists.append(d_final)

    if not init_dists or not final_dists:
        return 0.0
    mean_init = float(np.mean(init_dists))
    mean_final = float(np.mean(final_dists))
    if mean_init <= 0:
        return 0.0
    score = 1.0 - mean_final / mean_init
    return float(np.clip(score, 0.0, 1.0))


# ── Distortion Index (centroid-based) ───────────────────────────

def compute_distortion_index(result, mode='natural', centroid_log=None):
    """Quantify replay distortion from centroid movement.

    Uses _CENTROID_LOG if available (list of before/after centroids).
    Measures how much each replay event changes the centroid.

    Returns ∈ [0, ∞) where 0 = no distortion, higher = more movement.
    """
    if centroid_log is not None and len(centroid_log) > 0:
        deltas = []
        for e in centroid_log:
            cb = e.get('centroid_before', {})
            ca = e.get('centroid_after', {})
            mem_idx = e.get('memory_idx', -1)
            if mem_idx >= 0 and mem_idx in cb and mem_idx in ca:
                delta = np.linalg.norm(np.array(ca[mem_idx]) - np.array(cb[mem_idx]))
                deltas.append(delta)
        if deltas:
            return float(np.mean(deltas))
    return 0.0


# ── M7: Cross-memory drift correlation ──────────────────────────────

def compute_cross_memory_drift(snapshots, n_mem=4):
    """Mean pairwise cosine distance between memories at each checkpoint."""
    snap = _valid_snapshots(snapshots, n_mem)
    trajectories = {}
    for j in range(n_mem):
        dists = []
        for i in range(n_mem):
            for k in range(i + 1, n_mem):
                d = _pairwise_cosine(snap[i][j], snap[k][j])
                if not np.isnan(d):
                    dists.append(float(d))
        trajectories[j] = float(np.mean(dists)) if dists else np.nan
    return trajectories


# ── Main entry point ─────────────────────────────────────────────────

def compute_all(result, n_mem=4, centroid_log=None):
    """Run all schema analyses on an experiment result dict.

    Parameters
    ----------
    result : dict
        Must contain 'snapshots' (list-of-lists of weight vectors).
    n_mem : int
        Number of memories (default 4).
    centroid_log : list, optional
        List of before/after centroid dicts from _CENTROID_LOG.

    Returns
    -------
    metrics : dict with all computed values.
    """
    snapshots = result.get('snapshots', [])
    if not snapshots or len(snapshots) < n_mem:
        return {"error": "snapshots missing or incomplete"}

    centroids, schema_cents, distances = compute_centroids(snapshots, n_mem)

    # Base metrics
    drift = compute_drift(distances, n_mem)
    convergence = compute_schema_convergence(distances, n_mem)
    actual_drift, p_val, shuffle_drifts = permutation_test(distances, n_mem)
    schema_score = compute_schema_score(snapshots, n_mem)
    distortion = compute_distortion_index(result, centroid_log=centroid_log)
    cross = compute_cross_memory_drift(snapshots, n_mem)

    # Per-memory drift
    mem_drift = [drift.get(i, np.nan) for i in range(n_mem)]

    # Distance trajectories (for plotting)
    dist_traj = {i: [distances.get((i, j), np.nan) for j in range(n_mem)]
                 for i in range(n_mem)}

    # Cross-memory trajectory (mean pairwise distance per checkpoint)
    cross_traj = [cross.get(j, np.nan) for j in range(n_mem)]

    return {
        "schema_score":         schema_score,
        "convergence":          float(convergence),
        "drift_mean":           float(np.nanmean(mem_drift)),
        "drift_per_memory":     mem_drift,
        "permutation_p":        float(p_val),
        "permutation_actual":   float(actual_drift),
        "permutation_shuffles": shuffle_drifts,
        "distortion_index":     float(distortion),
        "distance_trajectories": dist_traj,
        "cross_memory_trajectory": cross_traj,
        "n_mem":                n_mem,
    }
