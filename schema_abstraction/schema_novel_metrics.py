"""Novel "Groundbreaking" Metrics for Schema Abstraction.

Three metrics that make this paper unique:

1. **Schema Crystallization Index (SCI)**: After each sleep cycle, mean
   pairwise correlation between the unique parts of all memories.  If
   unique parts remain decorrelated (SCI ≈ 0), the network is memorizing.
   If they become negatively correlated (SCI < -0.1), the network is using
   the schema to differentiate memories — a sign of intelligent compression.

2. **Catastrophic Forgetting Resistance (CFR)**: After learning Memory D,
   re-test Memory A.  (Retention_A_after_D / Retention_A_immediately_after_learning).
   In standard networks this is ~0.0.  Slow+Replay should have >0.5.

3. **Representational Drift Velocity**: Track centroid of each memory in
   PCA space across sleep cycles.  Drift velocity should decrease over
   sleep cycles (memories settle into attractors), and the angle between
   drift vectors of overlapping memories should decrease (they converge
   toward a shared schema basin).
"""

import numpy as np
from scipy.stats import pearsonr as _pearsonr

from compare_catastrophic_forgetting import _safe_mean
import compare_catastrophic_forgetting as _ccf


# ═════════════════════════════════════════════════════════════════════════
# 1.  SCHEMA CRYSTALLIZATION INDEX (SCI)
# ═════════════════════════════════════════════════════════════════════════

def compute_schema_crystallization_index(centroid_snapshots, assemblies,
                                          core_mask=None):
    """Compute the Schema Crystallization Index from centroid trajectories.

    SCI = mean pairwise Pearson r between unique parts of all memories.

    High positive SCI: unique parts are correlated (network treats them
    as overlapping).  SCI ≈ 0: unique parts are decorrelated (healthy
    memory separation).  SCI < -0.1: unique parts are anti-correlated
    (network differentiates through the schema — intelligent compression).

    Args:
        centroid_snapshots: list of {"centroids": [...], "label": str} dicts.
        assemblies: list of assembly index arrays.
        core_mask: array of schema-core indices (or None).

    Returns:
        dict with "SCI" trajectory list, "final_SCI", "interpretation".
    """
    if not centroid_snapshots or len(assemblies) < 2:
        return {"SCI": [], "final_SCI": 0.0, "interpretation": "insufficient_data"}

    n_mem = len(assemblies)
    sci_over_time = []

    for snap in centroid_snapshots:
        centroids = snap.get("centroids", [])
        if len(centroids) < 2:
            continue

        # Extract unique parts of each memory
        unique_parts = []
        for aidx in range(min(n_mem, len(centroids))):
            asm = assemblies[aidx]
            asm_exc = asm[asm < _ccf.N_EXC]
            cent = centroids[aidx].ravel() if hasattr(centroids[aidx], 'ravel') else np.asarray(centroids[aidx]).ravel()

            if core_mask is not None:
                core_exc = core_mask[core_mask < _ccf.N_EXC]
                unique_mask = np.setdiff1d(asm_exc, core_exc)
            else:
                unique_mask = asm_exc

            # The centroid is a vector over _ccf.N_EXC; extract the unique positions
            unique_part = cent[unique_mask] if len(unique_mask) <= len(cent) else cent[:len(unique_mask)]
            if len(unique_part) > 0:
                unique_parts.append(unique_part)

        if len(unique_parts) < 2:
            continue

        # Pairwise correlations
        corrs = []
        for i in range(len(unique_parts)):
            for j in range(i + 1, len(unique_parts)):
                a, b = unique_parts[i], unique_parts[j]
                min_len = min(len(a), len(b))
                if min_len < 2:
                    continue
                try:
                    r, _ = _pearsonr(a[:min_len], b[:min_len])
                    corrs.append(r)
                except Exception:
                    continue

        if corrs:
            sci_over_time.append(_safe_mean(corrs))

    if not sci_over_time:
        return {"SCI": [], "final_SCI": 0.0, "interpretation": "insufficient_data"}

    final_sci = sci_over_time[-1]
    if final_sci < -0.1:
        interpretation = "differentiating (healthy schema compression)"
    elif final_sci < 0.1:
        interpretation = "decorrelated (healthy memory separation)"
    elif final_sci < 0.3:
        interpretation = "slightly correlated (mild schema blending)"
    else:
        interpretation = "strongly correlated (possible catastrophic overlap)"

    return {
        "SCI_trajectory": sci_over_time,
        "final_SCI": final_sci,
        "interpretation": interpretation,
        "n_timepoints": len(sci_over_time),
    }


# ═════════════════════════════════════════════════════════════════════════
# 2.  CATASTROPHIC FORGETTING RESISTANCE (CFR)
# ═════════════════════════════════════════════════════════════════════════

def compute_cfr(final_scores, baseline_scores):
    """Compute Catastrophic Forgetting Resistance.

    CFR = retention_A_after_D / retention_A_immediately_after_learning.

    Args:
        final_scores: array of final retention scores [A, B, C, D].
        baseline_scores: array of baseline retention scores [A, B, C, D].

    Returns:
        dict with per-memory CFR and overall CFR.
    """
    if len(final_scores) < 1 or len(baseline_scores) < 1:
        return {"overall_CFR": 0.0, "per_memory_CFR": {}}

    per_memory = {}
    for i in range(min(len(final_scores), len(baseline_scores))):
        bl = baseline_scores[i] if np.isfinite(baseline_scores[i]) else 0.0
        fn = final_scores[i] if np.isfinite(final_scores[i]) else 0.0
        if bl > 1e-10:
            cfr = fn / bl
        else:
            cfr = 0.0
        per_memory[i] = cfr

    # Overall CFR = mean of per-memory values
    cfrs = [v for v in per_memory.values()]
    overall = _safe_mean(cfrs)

    # Memory A CFR is the most important (first-learned, most vulnerable)
    cfr_a = per_memory.get(0, 0.0)

    return {
        "overall_CFR": overall,
        "memory_A_CFR": cfr_a,
        "per_memory_CFR": per_memory,
    }


# ═════════════════════════════════════════════════════════════════════════
# 3.  REPRESENTATIONAL DRIFT VELOCITY
# ═════════════════════════════════════════════════════════════════════════

def compute_drift_velocity(centroid_snapshots, assemblies):
    """Compute representational drift velocity across sleep cycles.

    Tracks the centroid of each memory across snapshots and computes:
      - Drift velocity magnitude per memory per time step.
      - Mean drift velocity over the experiment.
      - Convergence angle: angle between drift vectors of overlapping
        memories should decrease (they converge toward shared schema).

    Args:
        centroid_snapshots: list of {"centroids": [...], "label": str} dicts.
        assemblies: list of assembly index arrays.

    Returns:
        dict with drift trajectories, velocities, convergence angles.
    """
    if not centroid_snapshots or len(centroid_snapshots) < 3:
        return {"drift_velocity": 0.0, "convergence_angle": 0.0,
                "velocity_trajectory": [], "angle_trajectory": []}

    n_mem = min(len(assemblies), len(centroid_snapshots[0].get("centroids", [])))

    # Extract centroid positions over time
    positions = []
    for snap in centroid_snapshots:
        cents = snap.get("centroids", [])
        if len(cents) >= n_mem:
            pos = np.array([c.ravel() for c in cents[:n_mem]])
            positions.append(pos)

    if len(positions) < 3:
        return {"drift_velocity": 0.0, "convergence_angle": 0.0,
                "velocity_trajectory": [], "angle_trajectory": []}

    positions = np.array(positions)  # (n_time, n_mem, n_features)

    # Velocity per time step
    velocities = np.diff(positions, axis=0)  # (n_time-1, n_mem, n_features)
    speed_per_step = np.linalg.norm(velocities, axis=2)  # (n_time-1, n_mem)

    # Mean drift velocity over all memories and time
    mean_velocity = float(np.mean(speed_per_step))

    # Convergence angle: mean angle between drift vectors of overlapping pairs
    angles = []
    n_pairs = 0
    for i in range(n_mem):
        for j in range(i + 1, n_mem):
            vec_i = velocities[:, i, :]  # (n_time-1, n_features)
            vec_j = velocities[:, j, :]
            cos_sim = np.sum(vec_i * vec_j, axis=1) / (
                np.linalg.norm(vec_i, axis=1) * np.linalg.norm(vec_j, axis=1) + 1e-10)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = np.arccos(cos_sim) * 180.0 / np.pi  # degrees
            angles.append(angle)
            n_pairs += 1

    if angles:
        all_angles = np.concatenate(angles)
        mean_angle = float(np.mean(all_angles))
        # Angle trajectory: decreasing = convergence
        angle_traj = [float(np.mean(a)) for a in zip(*angles)] if len(angles) > 1 else []
    else:
        mean_angle = 90.0
        angle_traj = []

    return {
        "drift_velocity": mean_velocity,
        "convergence_angle": mean_angle,
        "velocity_trajectory": [float(np.mean(v)) for v in speed_per_step],
        "angle_trajectory": angle_traj,
        "per_memory_velocity": [float(np.mean(speed_per_step[:, i])) for i in range(n_mem)],
    }


# ═════════════════════════════════════════════════════════════════════════
# 4.  MASTER RUNNER
# ═════════════════════════════════════════════════════════════════════════

def compute_all_novel_metrics(trial, assemblies, core_mask=None):
    """Compute all novel metrics for a single trial.

    Args:
        trial: trial result dict (with centroid_snapshots, baseline_scores,
               final_scores, etc.)
        assemblies: list of assembly index arrays.
        core_mask: array of schema-core indices (or None).

    Returns:
        dict with SCI, CFR, drift_velocity keys.
    """
    centroids = trial.get("centroid_snapshots", [])
    baseline = trial.get("baseline_scores", None)
    final = trial.get("final_scores", None)

    sci = compute_schema_crystallization_index(centroids, assemblies, core_mask)
    cfr = compute_cfr(final, baseline) if final is not None and baseline is not None else {"overall_CFR": 0.0}
    drift = compute_drift_velocity(centroids, assemblies)

    return {
        "schema_crystallization_index": sci,
        "catastrophic_forgetting_resistance": cfr,
        "drift_velocity": drift,
    }
