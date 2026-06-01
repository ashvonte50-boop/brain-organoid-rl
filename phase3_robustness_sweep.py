"""
PHASE 3 — Robustness Parameter Sweep
=======================================
Analytical sensitivity analysis: given existing centroid logs,
simulate how DAI and distortion would change if we varied replay parameters.

Two sweeps:
  1. Core Boost Factor (b): [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
     Controls how much core-to-core weights are amplified per replay
     In centroid space: scales the 'toward_schema' step size

  2. Noise Level (sigma): [0.0, 0.002, 0.004, 0.008, 0.016, 0.032]
     Controls random weight perturbation after replay (hyper-specific)
     In centroid space: adds isotropic noise to delta

For each parameter value:
  - Re-compute modified centroid logs from existing trajectory pkl data
  - Measure DAI_core, Distortion, Efficiency
  - Compare Natural vs Hyper ordering

Also sweeps replay frequency (fraction of events kept).

Note: retention cannot be swept analytically — reported at fixed value.
"""
import sys, os, pickle, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
from scipy.stats import sem as scipy_sem

from _distortion_paper import compute_directional_alignment
from phase2_distortion_decomposition import process_trajectory, aggregate_events
from schema_abstraction.schema_experiments import SCHEMA_CORE_SIZE, UNIQUE_SIZE

CORE_SIZE = SCHEMA_CORE_SIZE
DIM       = SCHEMA_CORE_SIZE + UNIQUE_SIZE
OUT       = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
SEEDS     = [42, 1042, 2042, 3042, 4042]
MODES     = ['natural', 'hyper']
os.makedirs(OUT, exist_ok=True)


# ── Centroid log modifier ─────────────────────────────────────────────────────

def load_centroid_log(mode, seed):
    path = f'trajectory_{mode}_seed{seed}.pkl'
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        traj = pickle.load(f)
    re = traj.get('replay_events', [])
    return re


def modify_log(log, boost_scale=1.0, noise_sigma=0.0, keep_fraction=1.0, seed=0):
    """
    Modify an existing centroid log to simulate parameter changes.

    boost_scale:     scales the core-component of each delta
                     (simulates changing core boost factor)
    noise_sigma:     adds isotropic Gaussian noise to delta
                     (simulates post-replay weight noise in hyper)
    keep_fraction:   randomly drops events (simulates replay frequency)
    """
    if not log:
        return []

    rng = np.random.RandomState(seed)
    new_log = []
    centroids = {}  # track current centroid state

    # Initialise from first event's centroid_before
    for e in log:
        cb = e.get('centroid_before', {})
        for k, v in cb.items():
            ik = int(k)
            if ik not in centroids:
                centroids[ik] = np.array(v)

    for ev_idx, e in enumerate(log):
        # Keep-fraction: randomly skip events
        if rng.uniform() > keep_fraction:
            continue

        cb_orig = {int(k): np.array(v) for k, v in e.get('centroid_before', {}).items()}
        ca_orig = {int(k): np.array(v) for k, v in e.get('centroid_after',  {}).items()}
        mem_idx = int(e.get('memory_idx', -1))

        if mem_idx < 0 or mem_idx not in cb_orig or mem_idx not in ca_orig:
            new_log.append(e)
            continue

        before = centroids.get(mem_idx, cb_orig[mem_idx]).copy()
        orig_delta = ca_orig[mem_idx] - cb_orig[mem_idx]

        # Apply boost to core component
        delta = orig_delta.copy()
        if boost_scale != 1.0 and len(delta) >= CORE_SIZE:
            delta[:CORE_SIZE] *= boost_scale

        # Apply noise
        if noise_sigma > 0:
            delta += rng.randn(len(delta)) * noise_sigma

        after = before + delta

        # Build cb BEFORE updating state, ca AFTER
        new_cb = {k: v.tolist() for k, v in centroids.items()}
        centroids[mem_idx] = after.copy()
        new_ca = {k: v.tolist() for k, v in centroids.items()}

        new_log.append({
            'replay_id':       e.get('replay_id', ev_idx),
            'memory_idx':      mem_idx,
            'centroid_before': new_cb,
            'centroid_after':  new_ca,
        })

    return new_log


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(log):
    """Compute DAI_core and distortion from a centroid log."""
    if not log:
        return {'dai_core': 0.0, 'distortion': 0.0, 'efficiency': 0.0}

    dai_out = compute_directional_alignment(log, n_mem=4, core_size=CORE_SIZE)

    # Distortion: mean centroid movement magnitude
    deltas = []
    for e in log:
        cb = e.get('centroid_before', {})
        ca = e.get('centroid_after', {})
        mem_idx = int(e.get('memory_idx', -1))
        if mem_idx >= 0 and mem_idx in cb and mem_idx in ca:
            d = np.linalg.norm(np.array(ca[mem_idx]) - np.array(cb[mem_idx]))
            deltas.append(d)

    distortion = float(np.mean(deltas)) if deltas else 0.0

    # Efficiency from decomposition
    schema_attractor = None
    latest = {}
    for e in log:
        for k, v in e.get('centroid_after', {}).items():
            latest[int(k)] = np.array(v)
    if latest:
        schema_attractor = np.mean(list(latest.values()), axis=0)

    efficiencies = []
    if schema_attractor is not None:
        for e in log:
            cb = e.get('centroid_before', {})
            ca = e.get('centroid_after',  {})
            mem_idx = int(e.get('memory_idx', -1))
            if mem_idx < 0 or mem_idx not in cb or mem_idx not in ca:
                continue
            before = np.array(cb[mem_idx])
            after  = np.array(ca[mem_idx])
            if len(before) <= CORE_SIZE:
                continue
            dc = after[:CORE_SIZE] - before[:CORE_SIZE]
            tc = schema_attractor[:CORE_SIZE] - before[:CORE_SIZE]
            tn = np.linalg.norm(tc)
            if tn < 1e-12:
                continue
            unit_t = tc / tn
            con = abs(float(np.dot(dc, unit_t)))
            dis = float(np.linalg.norm(dc - np.dot(dc, unit_t) * unit_t))
            efficiencies.append(con / (con + dis + 1e-12))

    efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0

    return {'dai_core': float(dai_out['mean_core']),
            'distortion': distortion,
            'efficiency': efficiency}


# ── Sweeps ────────────────────────────────────────────────────────────────────

def sweep_parameter(param_name, param_values, modifier_fn):
    """
    For each parameter value, compute metrics across all seeds and modes.
    modifier_fn(log, param_value, seed) -> modified log
    """
    results = {mode: {pv: [] for pv in param_values} for mode in MODES}

    for mode in MODES:
        for seed in SEEDS:
            log = load_centroid_log(mode, seed)
            if log is None:
                continue
            for pv in param_values:
                mod_log = modifier_fn(log, pv, seed)
                m = compute_metrics(mod_log)
                results[mode][pv].append(m)

    # Aggregate
    agg = {}
    for mode in MODES:
        agg[mode] = {}
        for pv in param_values:
            metrics_list = results[mode][pv]
            if not metrics_list:
                continue
            agg[mode][pv] = {
                metric: {
                    'mean': float(np.mean([m[metric] for m in metrics_list])),
                    'sem':  float(scipy_sem([m[metric] for m in metrics_list]))
                }
                for metric in metrics_list[0].keys()
            }
    return agg


def run_phase3():
    print('='*65, flush=True)
    print('PHASE 3: ROBUSTNESS PARAMETER SWEEP', flush=True)
    print('='*65, flush=True)
    print('Analytical sensitivity analysis from existing trajectory data', flush=True)

    all_results = {}

    # ── Sweep 1: Core Boost Factor ─────────────────────────────────────────
    boost_values = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
    print(f'\nSweep 1: Core Boost Factor {boost_values}', flush=True)

    def boost_modifier(log, boost, seed):
        return modify_log(log, boost_scale=boost, seed=seed)

    boost_agg = sweep_parameter('boost', boost_values, boost_modifier)
    all_results['boost'] = {'values': boost_values, 'agg': boost_agg}

    print(f'  {"Boost":>6}  {"Nat_DAI":>8}  {"Hyp_DAI":>8}  '
          f'{"Nat_Eff":>8}  {"Hyp_Eff":>8}', flush=True)
    for bv in boost_values:
        nd = boost_agg.get('natural', {}).get(bv, {}).get('dai_core', {}).get('mean', float('nan'))
        hd = boost_agg.get('hyper',   {}).get(bv, {}).get('dai_core', {}).get('mean', float('nan'))
        ne = boost_agg.get('natural', {}).get(bv, {}).get('efficiency', {}).get('mean', float('nan'))
        he = boost_agg.get('hyper',   {}).get(bv, {}).get('efficiency', {}).get('mean', float('nan'))
        flag = 'N>H' if nd > hd else 'H>N'
        print(f'  {bv:6.1f}  {nd:+8.4f}  {hd:+8.4f}  {ne:8.4f}  {he:8.4f}  {flag}',
              flush=True)

    nat_wins_boost = sum(1 for bv in boost_values
                         if boost_agg.get('natural',{}).get(bv,{}).get('dai_core',{}).get('mean',0) >
                            boost_agg.get('hyper',{}).get(bv,{}).get('dai_core',{}).get('mean',0))
    print(f'  Natural > Hyper DAI_core in {nat_wins_boost}/{len(boost_values)} boost levels', flush=True)

    # ── Sweep 2: Noise Level ───────────────────────────────────────────────
    noise_values = [0.0, 0.002, 0.004, 0.008, 0.016, 0.032]
    print(f'\nSweep 2: Replay Noise Sigma {noise_values}', flush=True)

    def noise_modifier(log, sigma, seed):
        return modify_log(log, noise_sigma=sigma, seed=seed)

    noise_agg = sweep_parameter('noise', noise_values, noise_modifier)
    all_results['noise'] = {'values': noise_values, 'agg': noise_agg}

    print(f'  {"Noise":>6}  {"Nat_DAI":>8}  {"Hyp_DAI":>8}  '
          f'{"Nat_Dis":>8}  {"Hyp_Dis":>8}', flush=True)
    for nv in noise_values:
        nd = noise_agg.get('natural', {}).get(nv, {}).get('dai_core',   {}).get('mean', float('nan'))
        hd = noise_agg.get('hyper',   {}).get(nv, {}).get('dai_core',   {}).get('mean', float('nan'))
        ndi= noise_agg.get('natural', {}).get(nv, {}).get('distortion', {}).get('mean', float('nan'))
        hdi= noise_agg.get('hyper',   {}).get(nv, {}).get('distortion', {}).get('mean', float('nan'))
        flag = 'N>H' if nd > hd else 'H>N'
        print(f'  {nv:6.3f}  {nd:+8.4f}  {hd:+8.4f}  {ndi:8.4f}  {hdi:8.4f}  {flag}',
              flush=True)

    nat_wins_noise = sum(1 for nv in noise_values
                         if noise_agg.get('natural',{}).get(nv,{}).get('dai_core',{}).get('mean',0) >
                            noise_agg.get('hyper',{}).get(nv,{}).get('dai_core',{}).get('mean',0))
    print(f'  Natural > Hyper DAI_core in {nat_wins_noise}/{len(noise_values)} noise levels', flush=True)

    # ── Sweep 3: Replay Frequency ──────────────────────────────────────────
    freq_values = [0.25, 0.5, 0.75, 1.0]
    print(f'\nSweep 3: Replay Frequency (fraction kept) {freq_values}', flush=True)

    def freq_modifier(log, frac, seed):
        return modify_log(log, keep_fraction=frac, seed=seed)

    freq_agg = sweep_parameter('frequency', freq_values, freq_modifier)
    all_results['frequency'] = {'values': freq_values, 'agg': freq_agg}

    print(f'  {"Freq":>6}  {"Nat_DAI":>8}  {"Hyp_DAI":>8}', flush=True)
    for fv in freq_values:
        nd = freq_agg.get('natural', {}).get(fv, {}).get('dai_core', {}).get('mean', float('nan'))
        hd = freq_agg.get('hyper',   {}).get(fv, {}).get('dai_core', {}).get('mean', float('nan'))
        flag = 'N>H' if nd > hd else 'H>N'
        print(f'  {fv:6.2f}  {nd:+8.4f}  {hd:+8.4f}  {flag}', flush=True)

    # Summary robustness
    total_N_wins = nat_wins_boost + nat_wins_noise
    total_points = len(boost_values) + len(noise_values)
    print(f'\nROBUSTNESS SUMMARY:', flush=True)
    print(f'  Natural > Hyper DAI_core in {total_N_wins}/{total_points} parameter conditions', flush=True)
    passed = total_N_wins >= total_points * 0.8
    print(f'  Phase 3: {"PASS" if passed else "PARTIAL — check specific conditions"}', flush=True)

    # Save
    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {str(k): _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(x) for x in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(os.path.join(OUT, 'phase3_robustness_raw.json'), 'w') as f:
        json.dump(_to_serialisable(all_results), f)
    print(f'\nSaved -> {OUT}/phase3_robustness_raw.json', flush=True)

    return all_results, passed


if __name__ == '__main__':
    all_results, passed = run_phase3()
