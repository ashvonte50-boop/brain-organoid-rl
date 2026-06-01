"""Reconstruct distortion_data.pkl from existing trajectory_*.pkl files.

Computes all metrics (retention, DAI, schema score, distortion, REAL_SCHEMA,
functional schema proxy) from the centroid + score data saved per seed.

Usage:
    python rebuild_dataset.py
"""
import os, sys, glob, pickle
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp

sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import schema_analysis as sa
from _distortion_paper import (
    compute_directional_alignment,
    _compute_rs_from_centroids,
)
from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE

OUT_PATH = r'C:\Users\Admin\brain-organoid-rl\figures\schema\distortion_data.pkl'
MODES    = ['no_replay', 'natural', 'hyper']


def _load_traj(mode, seed):
    p = f'trajectory_{mode}_seed{seed}.pkl'
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def _compute_real_schema_from_trajectory(traj, core_size=SCHEMA_CORE_SIZE):
    """REAL_SCHEMA from final centroid in trajectory."""
    stages = traj.get('trajectory', [])
    # Use the last 'centroid_after' from replay events, or last trajectory stage
    re = traj.get('replay_events', [])
    if re:
        # Get latest centroid for each memory
        latest = {}
        for e in re:
            ca = e.get('centroid_after', {})
            for k, v in ca.items():
                latest[int(k)] = np.array(v)
        if latest:
            return float(_compute_rs_from_centroids(latest, core_size=core_size))
    # Fallback: last trajectory stage
    for stage in reversed(stages):
        cents = stage.get('centroids', {})
        valid = {k: np.array(v) for k, v in cents.items()
                 if v is not None and np.all(np.isfinite(v))}
        if valid and any(np.array(v).shape[0] > core_size for v in valid.values()):
            return float(_compute_rs_from_centroids(valid, core_size=core_size))
    return 0.0


def _compute_schema_from_trajectory(traj, n_mem=4, core_size=SCHEMA_CORE_SIZE):
    """Reconstruct schema_analysis.compute_all result from trajectory stages."""
    stages = traj.get('trajectory', [])
    if not stages:
        return {}

    # Build snapshots matrix [n_mem][n_stages] as weight vectors
    # Each stage has 'centroids': {mem_idx: array}
    n_stages = len(stages)
    snapshots = []
    for i in range(n_mem):
        row = []
        for j, stage in enumerate(stages):
            cents = stage.get('centroids', {})
            v = cents.get(i)
            if v is None:
                v = cents.get(str(i))
            if v is not None and np.all(np.isfinite(np.array(v))):
                row.append(np.array(v, dtype=np.float64))
            else:
                row.append(None)
        snapshots.append(row)

    try:
        # Build distances manually from centroids
        from schema_analysis import compute_centroids, compute_drift, compute_schema_convergence
        from schema_analysis import permutation_test, compute_schema_score, compute_cross_memory_drift
        centroids, schema_cents, distances = compute_centroids(snapshots, n_mem)
        drift = compute_drift(distances, n_mem)
        convergence = compute_schema_convergence(distances, n_mem)
        actual_drift, p_val, shuffle_drifts = permutation_test(distances, n_mem)
        schema_score = compute_schema_score(snapshots, n_mem)
        cross = compute_cross_memory_drift(snapshots, n_mem)

        re = traj.get('replay_events', [])
        clog = []
        for e in re:
            if 'centroid_before' not in e:
                continue
            cb = {int(k): np.array(v) for k, v in e['centroid_before'].items()}
            ca = {int(k): np.array(v) for k, v in e['centroid_after'].items()}
            clog.append({'replay_id': e.get('replay_id', 0),
                         'memory_idx': int(e.get('memory_idx', -1)),
                         'centroid_before': cb, 'centroid_after': ca})
        from schema_analysis import compute_distortion_index
        distortion = compute_distortion_index(None, centroid_log=clog)

        mem_drift = [drift.get(i, np.nan) for i in range(n_mem)]
        dist_traj = {i: [distances.get((i, j), np.nan) for j in range(n_mem)] for i in range(n_mem)}
        cross_traj = [cross.get(j, np.nan) for j in range(n_mem)]
        return {
            'schema_score': schema_score,
            'convergence': float(convergence),
            'drift_mean': float(np.nanmean(mem_drift)),
            'drift_per_memory': mem_drift,
            'permutation_p': float(p_val),
            'permutation_actual': float(actual_drift),
            'permutation_shuffles': shuffle_drifts,
            'distortion_index': float(distortion),
            'distance_trajectories': dist_traj,
            'cross_memory_trajectory': cross_traj,
            'n_mem': n_mem,
        }
    except Exception as e:
        print(f'  schema compute error: {e}')
        return {'error': str(e)}


def _build_centroid_log(traj, core_size=SCHEMA_CORE_SIZE):
    re = traj.get('replay_events', [])
    clog = []
    for e in re:
        if 'centroid_before' not in e:
            continue
        cb = {int(k): v for k, v in e['centroid_before'].items()}
        ca = {int(k): v for k, v in e['centroid_after'].items()}
        clog.append({'replay_id': e.get('replay_id', 0),
                     'memory_idx': int(e.get('memory_idx', -1)),
                     'centroid_before': cb, 'centroid_after': ca})
    return clog


def _print_sep():
    print('-' * 50, flush=True)


def main():
    # Find available seeds
    all_seeds = set()
    for f in glob.glob('trajectory_natural_seed*.pkl'):
        try:
            seed = int(f.split('seed')[1].replace('.pkl', ''))
            all_seeds.add(seed)
        except:
            pass
    seeds = sorted(all_seeds)
    print(f'Found {len(seeds)} seeds: {seeds}', flush=True)
    if not seeds:
        print('ERROR: no trajectory_*.pkl files found', flush=True)
        return

    all_data = {m: {
        'results': [], 'finals': [], 'baselines': [],
        'schema': [], 'func_schemas': [], 'real_schemas': [],
        'directional_alignment': [], 'forward': [],
    } for m in MODES}

    for seed in seeds:
        print(f'\nSeed {seed}:', flush=True)
        for mode in MODES:
            traj = _load_traj(mode, seed)
            if traj is None:
                print(f'  {mode}: missing', flush=True)
                continue

            fs = np.array(traj['final_scores'])
            bs = np.array(traj['baseline_scores'])
            fs = np.nan_to_num(fs, nan=0.0)
            bs = np.nan_to_num(bs, nan=0.0)

            schema_m = _compute_schema_from_trajectory(traj)
            clog     = _build_centroid_log(traj)
            dall     = compute_directional_alignment(clog, n_mem=4, core_size=SCHEMA_CORE_SIZE)
            rs       = _compute_real_schema_from_trajectory(traj)

            all_data[mode]['finals'].append(fs.tolist())
            all_data[mode]['baselines'].append(bs.tolist())
            all_data[mode]['schema'].append(schema_m)
            all_data[mode]['directional_alignment'].append(dall)
            all_data[mode]['real_schemas'].append(rs)
            # Functional schema not available from trajectory pkl — set to NaN
            all_data[mode]['func_schemas'].append(np.nan)

            ss  = schema_m.get('schema_score', 'N/A')
            di  = schema_m.get('distortion_index', 'N/A')
            cnv = schema_m.get('convergence', 'N/A')
            pv  = schema_m.get('permutation_p', 'N/A')
            print(f'  {mode:12s} A={fs[0]:.4f}  RS={rs:.4f}  SS={ss:.4f}  '
                  f'DAI_core={dall["mean_core"]:+.4f}  p_core={dall["p_core"]:.4e}  '
                  f'n_ev={dall["n_events"]}', flush=True)

    # Aggregate
    print(f'\n{"="*60}', flush=True)
    print('AGGREGATED RESULTS', flush=True)
    print('='*60, flush=True)
    agg = {}
    for mode in MODES:
        d = all_data[mode]
        finals_list  = [f for f in d['finals']   if f]
        base_list    = [b for b in d['baselines'] if b]
        n = len(finals_list)
        if n == 0:
            print(f'\n  {mode.upper()}  (no data)', flush=True)
            continue

        fs_arr = np.array(finals_list)
        bs_arr = np.array(base_list)
        mf = fs_arr.mean(0)
        sf = fs_arr.std(0) / np.sqrt(n)
        mb = bs_arr.mean(0) if len(bs_arr) > 0 else np.zeros(4)

        ss_vals  = [sm.get('schema_score', np.nan) for sm in d['schema']]
        di_vals  = [sm.get('distortion_index', np.nan) for sm in d['schema']]
        cnv_vals = [sm.get('convergence', np.nan) for sm in d['schema']]
        pv_vals  = [sm.get('permutation_p', np.nan) for sm in d['schema']]
        rs_vals  = d['real_schemas']
        fs_func  = [x for x in d['func_schemas'] if not np.isnan(x)]

        dall_list = d['directional_alignment']
        dc_vals   = [x.get('mean_core', np.nan)   for x in dall_list]
        du_vals   = [x.get('mean_unique', np.nan) for x in dall_list]
        pc_vals   = [x.get('p_core', 1.0)         for x in dall_list]
        pu_vals   = [x.get('p_unique', 1.0)        for x in dall_list]
        nev_vals  = [x.get('n_events', 0)          for x in dall_list]
        rs_d_vals = [x.get('mean_rs_delta', np.nan) for x in dall_list]

        agg[mode] = {
            'n': n,
            'retention_mean':    mf.tolist(),
            'retention_sem':     sf.tolist(),
            'baseline_mean':     mb.tolist(),
            'schema_score_mean': float(np.nanmean(ss_vals)),
            'schema_score_sem':  float(np.nanstd(ss_vals) / np.sqrt(n)),
            'convergence_mean':  float(np.nanmean(cnv_vals)),
            'distortion_mean':   float(np.nanmean(di_vals)),
            'p_drift_mean':      float(np.nanmean(pv_vals)),
            'real_schema_mean':  float(np.nanmean(rs_vals)),
            'real_schema_sem':   float(np.nanstd(rs_vals) / np.sqrt(n)) if n > 1 else 0.0,
            'func_schema_mean':  float(np.nanmean(fs_func)) if fs_func else np.nan,
            'func_schema_sem':   0.0,
            'dai_core_mean':     float(np.nanmean(dc_vals)),
            'dai_core_sem':      float(np.nanstd(dc_vals) / np.sqrt(n)) if n > 1 else 0.0,
            'dai_unique_mean':   float(np.nanmean(du_vals)),
            'dai_unique_sem':    float(np.nanstd(du_vals) / np.sqrt(n)) if n > 1 else 0.0,
            'p_core_mean':       float(np.nanmean(pc_vals)),
            'p_unique_mean':     float(np.nanmean(pu_vals)),
            'n_events_mean':     float(np.nanmean(nev_vals)),
            'rs_delta_mean':     float(np.nanmean(rs_d_vals)),
        }

        print(f'\n  {mode.upper()}  (n={n})', flush=True)
        for i, name in enumerate(['A', 'B', 'C', 'D']):
            print(f'    {name}: {mb[i]:.4f} -> {mf[i]:.4f} +/-{sf[i]:.4f}', flush=True)
        print(f'    SchemaScore:  {agg[mode]["schema_score_mean"]:.4f} +/-{agg[mode]["schema_score_sem"]:.4f}', flush=True)
        print(f'    REAL_SCHEMA:  {agg[mode]["real_schema_mean"]:.4f} +/-{agg[mode]["real_schema_sem"]:.4f}', flush=True)
        print(f'    Convergence:  {agg[mode]["convergence_mean"]:.4f}', flush=True)
        print(f'    Distortion:   {agg[mode]["distortion_mean"]:.4f}', flush=True)
        print(f'    p_drift:      {agg[mode]["p_drift_mean"]:.4f}', flush=True)
        print(f'    DAI_core:     {agg[mode]["dai_core_mean"]:+.4f} +/-{agg[mode]["dai_core_sem"]:.4f}  p={agg[mode]["p_core_mean"]:.4e}', flush=True)
        print(f'    DAI_unique:   {agg[mode]["dai_unique_mean"]:+.4f} +/-{agg[mode]["dai_unique_sem"]:.4f}  p={agg[mode]["p_unique_mean"]:.4e}', flush=True)
        print(f'    n_events:     {agg[mode]["n_events_mean"]:.0f}', flush=True)

    # Hypothesis tests
    print(f'\n{"="*60}', flush=True)
    print('HYPOTHESIS TESTS', flush=True)
    print('='*60, flush=True)

    for label, key in [
        ('DAI_core',   'mean_core'),
        ('DAI_unique', 'mean_unique'),
        ('REAL_SCHEMA', None),
        ('Retention_A', None),
    ]:
        for cond_a, cond_b in [('natural', 'hyper'), ('natural', 'no_replay'), ('hyper', 'no_replay')]:
            if cond_a not in agg or cond_b not in agg:
                continue
            if label == 'REAL_SCHEMA':
                va = np.array(all_data[cond_a]['real_schemas'])
                vb = np.array(all_data[cond_b]['real_schemas'])
            elif label == 'Retention_A':
                va = np.array([f[0] for f in all_data[cond_a]['finals'] if f])
                vb = np.array([f[0] for f in all_data[cond_b]['finals'] if f])
            else:
                va = np.array([x.get(key, np.nan) for x in all_data[cond_a]['directional_alignment']])
                vb = np.array([x.get(key, np.nan) for x in all_data[cond_b]['directional_alignment']])
                if cond_b == 'no_replay':
                    vb = np.zeros(max(len(va), 1))

            va = va[np.isfinite(va)]; vb = vb[np.isfinite(vb)]
            if len(va) < 2 and len(vb) < 2:
                continue
            if len(vb) < 2:
                vb = np.zeros(len(va))
            if len(va) < 2:
                va = np.zeros(len(vb))
            t, p = ttest_ind(va, vb)
            stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
            print(f'  {label:14s} {cond_a:10s} vs {cond_b:10s}: t={t:+.3f}  p={p:.4f}  {stars}', flush=True)

    # Save
    save = {}
    for mode in MODES:
        d = all_data[mode]
        save[mode] = {
            'finals':               d['finals'],
            'baselines':            d['baselines'],
            'schema':               d['schema'],
            'directional_alignment':d['directional_alignment'],
            'real_schemas':         d['real_schemas'],
            'func_schemas':         d['func_schemas'],
            'forward':              d['forward'],
            'metrics':              [],
            'agg':                  agg.get(mode, {}),
        }
    save['config'] = {'n_seeds': len(seeds), 'seeds': seeds,
                      'core': SCHEMA_CORE_SIZE, 'source': 'trajectory_pkls'}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(save, f)
    print(f'\nSaved -> {OUT_PATH}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
