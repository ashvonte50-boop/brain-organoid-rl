"""
PHASE 2 — Distortion Decomposition
=====================================
Decompose centroid movement per replay event into:

  Conservative Distortion  = component parallel to schema direction
                             (toward the schema attractor)
  Dissipative Distortion   = component orthogonal to schema direction
                             (does not contribute to abstraction)

  Total Distortion         = ||delta||
  Efficiency               = Conservative / (Conservative + Dissipative)

Apply to all 5 seeds of real experiment data (trajectory pkls).
Compare NoReplay / Natural / Hyper.
"""
import sys, os, glob, pickle, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import ttest_ind, sem as scipy_sem

from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE, UNIQUE_SIZE

CORE_SIZE = SCHEMA_CORE_SIZE   # 20
DIM       = SCHEMA_CORE_SIZE + UNIQUE_SIZE   # 40
OUT       = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT, exist_ok=True)


# ── Decomposition ─────────────────────────────────────────────────────────────

def decompose_event(delta, toward_schema):
    """
    Decompose delta into:
      conservative = projection onto toward_schema (parallel component)
      dissipative  = residual (orthogonal component)

    Returns (conservative_mag, dissipative_mag, total_mag, efficiency)
    """
    ts_norm = np.linalg.norm(toward_schema)
    if ts_norm < 1e-12:
        total = float(np.linalg.norm(delta))
        return 0.0, total, total, 0.0

    unit_schema = toward_schema / ts_norm
    conservative_vec = np.dot(delta, unit_schema) * unit_schema
    dissipative_vec  = delta - conservative_vec

    conservative = float(np.linalg.norm(conservative_vec))
    dissipative  = float(np.linalg.norm(dissipative_vec))
    total        = float(np.linalg.norm(delta))
    efficiency   = conservative / (conservative + dissipative + 1e-12)

    return conservative, dissipative, total, efficiency


def process_trajectory(traj, core_size=CORE_SIZE):
    """
    Process a saved trajectory pkl and return per-event decomposition.
    """
    re = traj.get('replay_events', [])
    if not re:
        return None   # NoReplay

    # Build schema_attractor from latest centroids (matches compute_directional_alignment)
    latest = {}
    for e in re:
        for k, v in e.get('centroid_after', {}).items():
            latest[int(k)] = np.array(v)

    if not latest:
        return None

    schema_attractor = np.mean(list(latest.values()), axis=0)

    events = []
    for e in re:
        cb = e.get('centroid_before', {})
        ca = e.get('centroid_after', {})
        mem_idx = int(e.get('memory_idx', -1))
        if mem_idx < 0 or mem_idx not in cb or mem_idx not in ca:
            continue

        before = np.array(cb[mem_idx])
        after  = np.array(ca[mem_idx])
        if before.shape[0] <= core_size:
            continue

        delta         = after - before
        toward_schema = schema_attractor - before

        # Core component
        dc = delta[:core_size]
        tc = toward_schema[:core_size]
        con_c, dis_c, tot_c, eff_c = decompose_event(dc, tc)

        # Unique component
        du = delta[core_size:]
        tu = toward_schema[core_size:]
        con_u, dis_u, tot_u, eff_u = decompose_event(du, tu)

        # Full vector
        con_f, dis_f, tot_f, eff_f = decompose_event(delta, toward_schema)

        events.append({
            'conservative_core':  con_c,
            'dissipative_core':   dis_c,
            'total_core':         tot_c,
            'efficiency_core':    eff_c,
            'conservative_full':  con_f,
            'dissipative_full':   dis_f,
            'total_full':         tot_f,
            'efficiency_full':    eff_f,
        })

    return events if events else None


def aggregate_events(events_list):
    """Aggregate per-event metrics across seeds."""
    if not events_list or all(e is None for e in events_list):
        return None

    valid = [e for e in events_list if e is not None]
    all_events = [ev for seed_events in valid for ev in seed_events]

    keys = list(all_events[0].keys())
    agg = {}
    for k in keys:
        vals = [ev[k] for ev in all_events]
        agg[k] = {'mean': float(np.mean(vals)), 'sem': float(scipy_sem(vals)),
                  'all': vals}

    # Per-seed means (for between-condition tests)
    seed_means = {}
    for k in keys:
        seed_means[k] = [float(np.mean([ev[k] for ev in seed_ev]))
                         for seed_ev in valid]
    agg['seed_means'] = seed_means

    return agg


# ── Main ─────────────────────────────────────────────────────────────────────

def run_phase2():
    print('='*65, flush=True)
    print('PHASE 2: DISTORTION DECOMPOSITION', flush=True)
    print('='*65, flush=True)

    modes = ['no_replay', 'natural', 'hyper']
    seeds = [42, 1042, 2042, 3042, 4042]

    mode_agg = {}
    for mode in modes:
        events_per_seed = []
        for seed in seeds:
            path = f'trajectory_{mode}_seed{seed}.pkl'
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                traj = pickle.load(f)
            events = process_trajectory(traj)
            events_per_seed.append(events)
        mode_agg[mode] = aggregate_events(events_per_seed)

    # Print summary
    print(f'\n{"Mode":12s}  {"Conservative":>14}  {"Dissipative":>12}  {"Efficiency":>12}', flush=True)
    print('-'*55, flush=True)

    for mode in modes:
        agg = mode_agg[mode]
        if agg is None:
            print(f'  {mode:12s}  (no replay events)', flush=True)
            continue
        con = agg['conservative_core']
        dis = agg['dissipative_core']
        eff = agg['efficiency_core']
        print(f'  {mode:12s}  '
              f'{con["mean"]:.4f}+/-{con["sem"]:.4f}  '
              f'{dis["mean"]:.4f}+/-{dis["sem"]:.4f}  '
              f'{eff["mean"]:.4f}+/-{eff["sem"]:.4f}', flush=True)

    # Statistical tests
    print(f'\nHypothesis tests (between-seed means, core component):', flush=True)
    for metric in ('conservative_core', 'dissipative_core', 'efficiency_core',
                   'total_core'):
        nat_sm = mode_agg.get('natural', {})
        hyp_sm = mode_agg.get('hyper', {})
        if nat_sm and hyp_sm:
            nat_v = np.array(nat_sm.get('seed_means', {}).get(metric, []))
            hyp_v = np.array(hyp_sm.get('seed_means', {}).get(metric, []))
            if len(nat_v) >= 2 and len(hyp_v) >= 2:
                t, p = ttest_ind(nat_v, hyp_v)
                stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
                print(f'  {metric:25s} Natural vs Hyper: t={t:+.3f}  p={p:.4f}  {stars}',
                      flush=True)

    # Key finding: does Natural show higher efficiency than Hyper?
    nat_eff = mode_agg.get('natural', {})
    hyp_eff = mode_agg.get('hyper', {})
    if nat_eff and hyp_eff:
        nv = nat_eff.get('seed_means', {}).get('efficiency_core', [])
        hv = hyp_eff.get('seed_means', {}).get('efficiency_core', [])
        if len(nv) >= 2 and len(hv) >= 2:
            t, p = ttest_ind(nv, hv)
            print(f'\nKEY: Efficiency_core Natural={np.mean(nv):.4f} vs Hyper={np.mean(hv):.4f}'
                  f'  t={t:.2f}  p={p:.4f}  '
                  f'{"Natural more efficient (PASS)" if np.mean(nv) > np.mean(hv) else "Hyper more efficient"}',
                  flush=True)

    # Save
    save = {}
    for mode in modes:
        agg = mode_agg[mode]
        if agg is None:
            save[mode] = None
        else:
            save[mode] = {k: {'mean': v['mean'], 'sem': v['sem']}
                          if isinstance(v, dict) and 'mean' in v else v
                          for k, v in agg.items()
                          if k != 'seed_means'}
            save[mode]['seed_means'] = {
                k: [float(x) for x in v]
                for k, v in agg.get('seed_means', {}).items()
            }

    with open(os.path.join(OUT, 'phase2_decomposition_raw.json'), 'w') as f:
        json.dump(save, f)
    print(f'\nSaved -> {OUT}/phase2_decomposition_raw.json', flush=True)

    return mode_agg


if __name__ == '__main__':
    run_phase2()
