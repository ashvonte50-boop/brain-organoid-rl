#!/usr/bin/env python3
"""
PHASE 1: Full Metric Audit
Prints every formula, input, and intermediate value for one seed.
"""
import pickle
import numpy as np

def audit_metric(filepath):
    print(f"\n{'='*70}")
    print(f"AUDITING: {filepath}")
    print(f"{'='*70}")
    
    data = pickle.load(open(filepath, 'rb'))
    
    traj = data.get('trajectory', [])
    replay_events = data.get('replay_events', [])
    
    print(f"\n--- TRAJECTORY STRUCTURE ---")
    print(f"Total stages: {len(traj)}")
    for i, stage in enumerate(traj[:3]):
        print(f"  Stage {i}: {stage.get('stage_name', 'UNKNOWN')} "
              f"centroids_keys={list(stage.get('centroids', {}).keys())}")
    
    print(f"\n--- SCHEMASCORE AUDIT ---")
    print(f"Formula: SchemaScore = mean(core_activation) / (mean(unique_activation) + eps)")
    
    final_stage = traj[-1] if traj else None
    if final_stage:
        cents = final_stage.get('centroids', {})
        print(f"  Final centroids available: {list(cents.keys())}")
        for mem_idx, centroid in cents.items():
            if centroid is not None:
                c = np.array(centroid)
                print(f"  Memory {mem_idx}: centroid shape={c.shape} "
                      f"mean={c.mean():.6f} std={c.std():.6f}")
            else:
                print(f"  Memory {mem_idx}: centroid is NONE")
    
    print(f"\n--- SCI AUDIT ---")
    print(f"Formula: SCI = (core_core_mean - unique_mean) / (core_core_mean + unique_mean + eps)")
    
    # Try to compute SCI manually from weights
    W = None
    if final_stage and 'network_state' in final_stage:
        W = final_stage['network_state'].get('weights')
    if W is None and 'final_scores' in data:
        print(f"  No weights in trajectory, using centroid-based approximation")
        cents = final_stage.get('centroids', {})
        core_idx = data.get('core_idx', [])
        assemblies = data.get('assemblies', [])
        print(f"  Core indices: {len(core_idx)} core neurons")
        print(f"  Number of assemblies: {len(assemblies)}")
        
        if cents and len(core_idx) > 0:
            core_np = np.array(core_idx)
            for mem_idx, cent in cents.items():
                if cent is not None:
                    c = np.array(cent)
                    # Split centroid into core and unique parts
                    core_part = c[:len(core_np)] if len(c) >= len(core_np) else c
                    unique_part = c[len(core_np):] if len(c) > len(core_np) else np.array([0.0])
                    print(f"  Memory {mem_idx}: core_part_mean={core_part.mean():.6f}, unique_part_mean={unique_part.mean() if len(unique_part) > 0 else 0:.6f}")
            print(f"  NOTE: Centroids are mean row vectors of assembly weight submatrices")
            print(f"  The SCI formula requires full weight matrix access")
    
    print(f"\n--- DISTORTION INDEX AUDIT ---")
    print(f"Formula: DistIdx = std(drift_vectors) / (mean(drift_magnitude) + eps)")
    
    if replay_events:
        drifts = []
        for ev in replay_events[:5]:
            c_before = ev.get('centroid_before')
            c_after = ev.get('centroid_after')
            if c_before is not None and c_after is not None:
                # Compute drift for each memory
                for mem in c_before:
                    if mem in c_after and c_before[mem] is not None and c_after[mem] is not None:
                        cb = np.array(c_before[mem])
                        ca = np.array(c_after[mem])
                        drift = ca - cb
                        drifts.append(drift)
                        print(f"  Event {ev.get('replay_id')}: Mem{mem} drift_norm={np.linalg.norm(drift):.6f}")
        
        if drifts:
            drift_array = np.array(drifts)
            print(f"  Total drift vectors: {len(drifts)}")
            mags = np.linalg.norm(drift_array, axis=1)
            print(f"  Mean drift magnitude: {mags.mean():.6f}")
            print(f"  Std of drift directions: {drift_array.std(axis=0).mean():.6f}")
            if mags.mean() > 0:
                dist_idx = drift_array.std(axis=0).mean() / (mags.mean() + 1e-9)
                print(f"  COMPUTED DistIdx: {dist_idx:.6f}")
            else:
                print(f"  COMPUTED DistIdx: 0 (zero drift magnitude)")
    else:
        print(f"  NO REPLAY EVENTS")
    
    print(f"\n--- CONVERGENCE AUDIT ---")
    print(f"Formula: Conv = mean(pairwise_distance_change) / initial_mean_distance")
    
    if len(traj) >= 2:
        first = traj[0].get('centroids', {})
        last = traj[-1].get('centroids', {})
        
        if first and last:
            dists_before = []
            dists_after = []
            for i in range(4):
                for j in range(i+1, 4):
                    ci = first.get(i) if i in first else None
                    cj = first.get(j) if j in first else None
                    if ci is not None and cj is not None:
                        d = np.linalg.norm(np.array(ci) - np.array(cj))
                        dists_before.append(d)
                    ci = last.get(i) if i in last else None
                    cj = last.get(j) if j in last else None
                    if ci is not None and cj is not None:
                        d = np.linalg.norm(np.array(ci) - np.array(cj))
                        dists_after.append(d)
            
            if dists_before and dists_after:
                print(f"  Initial pairwise distances: {np.round(dists_before, 4)}")
                print(f"  Final pairwise distances: {np.round(dists_after, 4)}")
                changes = [(a - b) / (b + 1e-9) for a, b in zip(dists_after, dists_before)]
                print(f"  Relative changes: {np.round(changes, 4)}")
                conv = np.mean(changes)
                print(f"  COMPUTED Conv: {conv:.6f}")

if __name__ == '__main__':
    for mode in ['no_replay', 'natural', 'hyper']:
        filepath = f'trajectory_{mode}_seed42.pkl'
        try:
            audit_metric(filepath)
        except Exception as e:
            print(f"\n[ERROR] {filepath}: {e}")
            import traceback
            traceback.print_exc()
