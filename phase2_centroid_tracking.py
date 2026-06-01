#!/usr/bin/env python3
"""
PHASE 2: Verify Centroid Tracking
Produces trajectory plots and numerical answers.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_trajectory(filepath):
    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS: {filepath}")
    print(f"{'='*70}")
    
    data = pickle.load(open(filepath, 'rb'))
    traj = data['trajectory']
    replay_events = data.get('replay_events', [])
    
    stages = {}
    for t in traj:
        name = t['stage_name']
        if name not in stages:
            stages[name] = []
        stages[name].append(t)
    
    print(f"\n--- STAGE COUNTS ---")
    for name, entries in stages.items():
        print(f"  {name:20s}: {len(entries)} entries")
    
    print(f"\n--- CENTROID MOVEMENT ---")
    initial = traj[0]['centroids'] if traj else {}
    final = traj[-1]['centroids'] if traj else {}
    
    for mem_idx in sorted(initial.keys()):
        c0 = initial.get(mem_idx)
        c1 = final.get(mem_idx)
        if c0 is not None and c1 is not None:
            movement = np.linalg.norm(np.array(c1) - np.array(c0))
            print(f"  Memory {mem_idx}: initial->final movement = {movement:.6f}")
    
    print(f"\n--- REPLAY EVENT ANALYSIS ---")
    print(f"Total replay events: {len(replay_events)}")
    
    if replay_events:
        movements = []
        for ev in replay_events:
            cb = ev.get('centroid_before')
            ca = ev.get('centroid_after')
            if cb is not None and ca is not None:
                for mem in cb:
                    if mem in ca and cb[mem] is not None and ca[mem] is not None:
                        d = np.linalg.norm(np.array(ca[mem]) - np.array(cb[mem]))
                        movements.append(d)
        
        if movements:
            print(f"  Mean movement per event: {np.mean(movements):.6f}")
            print(f"  Max movement: {np.max(movements):.6f}")
            print(f"  Min movement: {np.min(movements):.6f}")
            print(f"  Events with movement > 0.01: {sum(1 for m in movements if m > 0.01)}")
    
    return data

def plot_trajectories(all_data):
    """Plot centroid trajectories in 2D PCA space."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (mode, data) in enumerate(all_data.items()):
        traj = data['trajectory']
        
        all_cents = []
        labels = []
        for t in traj:
            for mem, cent in t.get('centroids', {}).items():
                if cent is not None:
                    all_cents.append(np.array(cent))
                    labels.append(f"{t['stage_name']}_M{mem}")
        
        if len(all_cents) < 2:
            axes[idx].text(0.5, 0.5, 'Insufficient data', ha='center')
            continue
        
        X = np.array(all_cents)
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        
        if X_centered.shape[1] > 1:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            X_2d = X_centered @ Vt[:2].T
        else:
            X_2d = X_centered[:, :2]
        
        colors = ['red', 'blue', 'green', 'purple']
        for mem_idx in range(4):
            mask = [i for i, l in enumerate(labels) if f'M{mem_idx}' in l]
            if mask:
                axes[idx].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                                c=colors[mem_idx], label=f'Memory {mem_idx}', alpha=0.6, s=20)
                axes[idx].plot(X_2d[mask, 0], X_2d[mask, 1], 
                             c=colors[mem_idx], alpha=0.3, linewidth=0.5)
        
        axes[idx].set_title(f'{mode}\n(n={len(all_cents)} centroids)')
        axes[idx].legend(fontsize=8)
        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig('fig_centroid_trajectories.png', dpi=300)
    plt.close()
    print("\nSaved: fig_centroid_trajectories.png")

if __name__ == '__main__':
    all_data = {}
    for mode in ['no_replay', 'natural', 'hyper']:
        filepath = f'trajectory_{mode}_seed42.pkl'
        try:
            all_data[mode] = analyze_trajectory(filepath)
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_data:
        plot_trajectories(all_data)
