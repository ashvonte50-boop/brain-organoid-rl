#!/usr/bin/env python3
"""
PHASE 3: Representational Similarity Analysis
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_similarity_matrix(centroids_dict):
    """Compute cosine similarity between all memory centroids."""
    mems = sorted([k for k in centroids_dict.keys() if centroids_dict[k] is not None])
    n = len(mems)
    sim = np.eye(n)
    
    for i in range(n):
        for j in range(i+1, n):
            ci = np.array(centroids_dict[mems[i]])
            cj = np.array(centroids_dict[mems[j]])
            if ci is not None and cj is not None:
                cos_sim = np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-9)
                sim[i, j] = cos_sim
                sim[j, i] = cos_sim
    
    return sim, mems

def analyze_representational_drift(filepath):
    print(f"\n{'='*70}")
    print(f"REPRESENTATIONAL DRIFT: {filepath}")
    print(f"{'='*70}")
    
    data = pickle.load(open(filepath, 'rb'))
    traj = data['trajectory']
    
    initial = None
    final = None
    for t in traj:
        if t['stage_name'] == 'initial':
            initial = t['centroids']
        elif t['stage_name'] == 'final':
            final = t['centroids']
    
    if initial is None or final is None:
        print("Missing initial or final stage")
        return None, None
    
    sim_initial, mems = compute_similarity_matrix(initial)
    sim_final, _ = compute_similarity_matrix(final)
    
    print(f"\n--- INITIAL SIMILARITY ---")
    print(np.round(sim_initial, 3))
    
    print(f"\n--- FINAL SIMILARITY ---")
    print(np.round(sim_final, 3))
    
    print(f"\n--- SIMILARITY CHANGE ---")
    delta = sim_final - sim_initial
    print(np.round(delta, 3))
    off_diag = delta[np.triu_indices_from(delta, k=1)]
    print(f"Mean off-diagonal change: {np.mean(off_diag):.6f}")
    
    print(f"\n--- MEMORY STABILITY ---")
    for i, mem in enumerate(mems):
        c_init = np.array(initial.get(mem))
        c_final = np.array(final.get(mem))
        if c_init is not None and c_final is not None:
            self_sim = np.dot(c_init, c_final) / (np.linalg.norm(c_init) * np.linalg.norm(c_final) + 1e-9)
            print(f"  Memory {mem} self-similarity: {self_sim:.6f}")
    
    return sim_initial, sim_final

if __name__ == '__main__':
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for idx, mode in enumerate(['no_replay', 'natural', 'hyper']):
        filepath = f'trajectory_{mode}_seed42.pkl'
        try:
            sim_i, sim_f = analyze_representational_drift(filepath)
            
            if sim_i is not None:
                im0 = axes[idx, 0].imshow(sim_i, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[idx, 0].set_title(f'{mode}: Initial')
                plt.colorbar(im0, ax=axes[idx, 0])
                
                im1 = axes[idx, 1].imshow(sim_f, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[idx, 1].set_title(f'{mode}: Final')
                plt.colorbar(im1, ax=axes[idx, 1])
                
                im2 = axes[idx, 2].imshow(sim_f - sim_i, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                axes[idx, 2].set_title(f'{mode}: Change')
                plt.colorbar(im2, ax=axes[idx, 2])
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    plt.tight_layout()
    plt.savefig('fig_similarity_heatmaps.png', dpi=300)
    plt.close()
    print("\nSaved: fig_similarity_heatmaps.png")
