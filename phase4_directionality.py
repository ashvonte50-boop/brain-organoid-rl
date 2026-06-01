#!/usr/bin/env python3
"""
PHASE 4: Directionality Analysis (v3 — replay-event-level)
The most important analysis in the paper.
Uses replay event log to isolate PURE replay effect on directionality.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def replay_event_directionality(filepath):
    print(f"\n{'='*70}")
    print(f"REPLAY-EVENT DIRECTIONALITY: {filepath}")
    print(f"{'='*70}")
    
    data = pickle.load(open(filepath, 'rb'))
    traj = data['trajectory']
    events = data.get('replay_events', [])
    
    if not events:
        print("NO REPLAY EVENTS (no_replay mode)")
        return None
    
    # Get final centroids for LOO schema
    final_cents = {}
    for t in traj:
        if t['stage_name'] == 'final':
            for mem, cent in t.get('centroids', {}).items():
                if cent is not None:
                    final_cents[mem] = np.array(cent)
    
    all_mems = sorted(final_cents.keys())
    print(f"Final centroids for {len(all_mems)} memories: {all_mems}")
    
    # For each event, determine if centroid movement is toward LOO schema
    event_results = []
    
    for ev in events:
        cb = ev.get('centroid_before')
        ca = ev.get('centroid_after')
        replay_id = ev.get('replay_id', -1)
        
        if cb is None or ca is None:
            continue
        
        for mem in cb:
            if mem not in ca:
                continue
            c_before = np.array(cb[mem])
            c_after = np.array(ca[mem])
            if c_before is None or c_after is None:
                continue
            
            # LOO schema centroid (all OTHER memories at final)
            other = [m for m in all_mems if m != mem]
            if len(other) < 1:
                continue
            schema = np.mean([final_cents[m] for m in other], axis=0)
            
            # Direction from before-centroid to schema
            to_schema = schema - c_before
            
            # Actual drift during this event
            drift = c_after - c_before
            
            # Cosine similarity
            n_to = np.linalg.norm(to_schema)
            n_drift = np.linalg.norm(drift)
            if n_to < 1e-12 or n_drift < 1e-12:
                continue
            
            cos_theta = np.dot(to_schema, drift) / (n_to * n_drift)
            
            event_results.append({
                'replay_id': replay_id,
                'memory': mem,
                'cos_theta': cos_theta,
                'drift_norm': n_drift,
                'to_schema_norm': n_to,
                'toward_schema': cos_theta > 0,
            })
    
    if not event_results:
        print("No valid event results")
        return None
    
    cos_vals = np.array([r['cos_theta'] for r in event_results])
    toward = np.array([r['toward_schema'] for r in event_results])
    drift_norms = np.array([r['drift_norm'] for r in event_results])
    
    print(f"Total events analyzed: {len(event_results)}")
    print(f"Mean cos(theta): {cos_vals.mean():.6f}")
    print(f"Median cos(theta): {np.median(cos_vals):.6f}")
    print(f"Std cos(theta): {cos_vals.std():.6f}")
    print(f"Fraction toward schema: {toward.mean():.2%}")
    print(f"Mean drift norm: {drift_norms.mean():.6f}")
    
    # Permutation test: null = random signs (mean zero directionality)
    rng = np.random.RandomState(42)
    n_perm = 2000
    perm_means = []
    observed_mean = cos_vals.mean()
    for _ in range(n_perm):
        # Randomly flip signs: null hypothesis is zero directional bias
        signs = rng.choice([-1, 1], size=len(cos_vals))
        shuffled = cos_vals * signs
        perm_means.append(shuffled.mean())
    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= observed_mean)  # one-sided: is observed > null?
    z_score = (observed_mean - perm_means.mean()) / (perm_means.std() + 1e-9)
    
    print(f"Permutation test (shuffle event labels):")
    print(f"  Null mean: {perm_means.mean():.6f} +/- {perm_means.std():.6f}")
    print(f"  Observed: {cos_vals.mean():.6f}")
    print(f"  z-score: {z_score:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Also test: per-memory directionality
    print(f"\n--- Per-Memory Directionality ---")
    for mem in sorted(set(r['memory'] for r in event_results)):
        mem_cos = [r['cos_theta'] for r in event_results if r['memory'] == mem]
        mem_toward = [r['toward_schema'] for r in event_results if r['memory'] == mem]
        if mem_cos:
            print(f"  Memory {mem}: mean cos={np.mean(mem_cos):.6f}  toward={np.mean(mem_toward):.2%}  n={len(mem_cos)}")
    
    return {
        'cos_vals': cos_vals,
        'toward': toward,
        'drift_norms': drift_norms,
        'p_value': p_value,
        'z_score': z_score,
        'mean_cos': float(cos_vals.mean()),
    }

def core_vs_unique_directionality(filepath):
    """Split centroid into core and unique parts and test directionality separately."""
    print(f"\n--- CORE vs UNIQUE DIRECTIONALITY ---")
    
    data = pickle.load(open(filepath, 'rb'))
    traj = data['trajectory']
    events = data.get('replay_events', [])
    core_idx = data.get('core_idx', [])
    assemblies = data.get('assemblies', [])
    
    if not events:
        print("No events")
        return
    
    # Get assembly sizes and core size
    core_size = len(core_idx) if core_idx else 20
    asm_size = len(assemblies[0]) if assemblies else 40
    
    # Final centroids for LOO schema
    final_cents = {}
    for t in traj:
        if t['stage_name'] == 'final':
            for mem, cent in t.get('centroids', {}).items():
                if cent is not None:
                    final_cents[mem] = np.array(cent)
    
    all_mems = sorted(final_cents.keys())
    
    # Split directionality by core vs unique components
    core_cos_list = []
    unique_cos_list = []
    
    for ev in events:
        cb = ev.get('centroid_before')
        ca = ev.get('centroid_after')
        if cb is None or ca is None:
            continue
        
        for mem in cb:
            if mem not in ca:
                continue
            c_before = np.array(cb[mem])
            c_after = np.array(ca[mem])
            if c_before is None or c_after is None:
                continue
            
            other = [m for m in all_mems if m != mem]
            if len(other) < 1:
                continue
            schema = np.mean([final_cents[m] for m in other], axis=0)
            
            # Core part (first core_size elements)
            c_before_core = c_before[:core_size]
            c_after_core = c_after[:core_size]
            schema_core = schema[:core_size]
            
            # Unique part (remaining elements)
            c_before_uniq = c_before[core_size:]
            c_after_uniq = c_after[core_size:]
            schema_uniq = schema[core_size:]
            
            # Core directionality
            to_s_core = schema_core - c_before_core
            drift_core = c_after_core - c_before_core
            n_tc = np.linalg.norm(to_s_core)
            n_dc = np.linalg.norm(drift_core)
            if n_tc > 1e-12 and n_dc > 1e-12:
                core_cos = np.dot(to_s_core, drift_core) / (n_tc * n_dc)
                core_cos_list.append(core_cos)
            
            # Unique directionality
            to_s_uniq = schema_uniq - c_before_uniq
            drift_uniq = c_after_uniq - c_before_uniq
            n_tu = np.linalg.norm(to_s_uniq)
            n_du = np.linalg.norm(drift_uniq)
            if n_tu > 1e-12 and n_du > 1e-12:
                unique_cos = np.dot(to_s_uniq, drift_uniq) / (n_tu * n_du)
                unique_cos_list.append(unique_cos)
    
    if core_cos_list and unique_cos_list:
        cc = np.array(core_cos_list)
        uc = np.array(unique_cos_list)
        # Paired t-test: core vs unique within the same events
        paired_diff = cc - uc
        from scipy import stats
        t_stat, p_paired = stats.ttest_1samp(paired_diff, 0)
        print(f"  Core component cos(theta): mean={cc.mean():.6f}  toward={np.mean(cc>0):.2%}  n={len(cc)}")
        print(f"  Unique component cos(theta): mean={uc.mean():.6f}  toward={np.mean(uc>0):.2%}  n={len(uc)}")
        print(f"  Core > Unique delta: {cc.mean() - uc.mean():.6f}")
        print(f"  Paired t-test (core vs unique): t={t_stat:.3f}  p={p_paired:.6f}")
    elif core_cos_list:
        cc = np.array(core_cos_list)
        print(f"  Core component cos(theta): mean={cc.mean():.6f}  toward={np.mean(cc>0):.2%}  n={len(cc)}")
        print(f"  Unique component: NO DATA")
    else:
        print(f"  No directionality data")

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    all_means = {}
    all_p = {}
    
    for mode in ['no_replay', 'natural', 'hyper']:
        filepath = f'trajectory_{mode}_seed42.pkl'
        try:
            result = replay_event_directionality(filepath)
            core_vs_unique_directionality(filepath)
            
            if result:
                all_means[mode] = result['mean_cos']
                all_p[mode] = result['p_value']
                
                # Plot distribution
                cos = result['cos_vals']
                ax1.hist(cos, bins=20, alpha=0.5, label=f"{mode} (mean={result['mean_cos']:.3f})")
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('cos(theta) per replay event')
    ax1.set_ylabel('Count')
    ax1.set_title('Replay Event Directionality Distribution')
    ax1.legend()
    
    # Summary bar plot
    if all_means:
        modes = list(all_means.keys())
        means = [all_means[m] for m in modes]
        colors = ['#e74c3c', '#2ecc71', '#f39c12']
        ax2.bar(modes, means, color=colors, alpha=0.8)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        for i, (m, p) in enumerate(zip(modes, [all_p.get(m, 1) for m in modes])):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax2.text(i, means[i], f'{means[i]:.4f}\n{p=:.4f}\n{sig}', ha='center', fontsize=9)
        ax2.set_ylabel('Mean cos(theta) per replay event')
        ax2.set_title('Replay Directionality: Mean Across Events')
    
    plt.tight_layout()
    plt.savefig('fig_directionality.png', dpi=300)
    plt.close()
    print("\nSaved: fig_directionality.png")
