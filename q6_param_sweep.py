"""
Q6: Parameter Sensitivity Sweep
================================
Vary 4 key parameters around baseline, measure retention under FULL replay.
Single seed (42), DEV_MODE. Compare each to the known baseline (seed 42 FULL = 0.3064).

Parameters and sweep values:
  gamma:        [0.3, 0.5, 0.65(baseline), 0.8, 0.95]
  tau_slow:     [500, 2000, 4000(baseline), 8000, 16000]
  core_size:    [5, 10, 20(baseline), 30, 40]  — but 40 saturates, so [5, 10, 20, 30]
  w_max:        [0.5, 1.0, 1.5(baseline), 2.0, 3.0]

For each, run FULL condition only and record retention.
"""
import os, sys, pickle, time, json
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net

SEED = 42
N_MEM = 4
OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\q6_sweep'
os.makedirs(OUT_DIR, exist_ok=True)

# Baseline values
BASELINE = {'gamma': 0.65, 'tau_slow': 4000, 'core_size': 20, 'w_max': 1.5}
BASELINE_RET = 0.3064  # seed 42 FULL from Task 2

# Parameter sweeps — only non-baseline values (baseline already known)
SWEEPS = {
    'gamma':     [0.3, 0.5, 0.8, 0.95],
    'tau_slow':  [500, 2000, 8000, 16000],
    'core_size': [5, 10, 30],
    'w_max':     [0.5, 1.0, 2.0, 3.0],
}

t_total_start = time.time()

# Resume from saved results if they exist
PKL_PATH = os.path.join(OUT_DIR, 'Q6_sweep_results.pkl')
if os.path.exists(PKL_PATH):
    with open(PKL_PATH, 'rb') as f:
        results = pickle.load(f)
    print(f'[Q6] Resuming with {len(results)} completed conditions: {list(results.keys())}', flush=True)
else:
    results = {}

for param_name, values in SWEEPS.items():
    print(f'\n{"="*60}', flush=True)
    print(f'[Q6] Sweeping {param_name}: {values}', flush=True)
    print(f'{"="*60}', flush=True)

    for val in values:
        cond_name = f'{param_name}_{val}'
        if cond_name in results:
            print(f'[Q6] {cond_name} already completed (ret={results[cond_name]["retention"]:.4f}), skipping.', flush=True)
            continue
        t_start = time.time()
        print(f'\n[Q6] {cond_name} ...', flush=True)

        # Reset seeds
        ccf.torch.manual_seed(SEED)
        ccf.np.random.seed(SEED)

        # Determine core_size for this run
        cs = int(val) if param_name == 'core_size' else SCHEMA_CORE_SIZE

        # Make assemblies with possibly different core_size
        assemblies, core_mask = make_schema_assemblies(N_MEM, cs, UNIQUE_SIZE)
        core = np.asarray(core_mask, dtype=np.int64)

        # Apply parameter override
        # gamma: controls W_eff = (1-gamma)*W + gamma*W_slow
        # tau_slow: slow weight time constant
        # w_max: maximum weight
        # These need to be set before building the network

        # Save originals
        orig_vals = {}

        if param_name == 'gamma':
            # gamma is used in the network's forward pass
            orig_vals['GAMMA'] = getattr(ccf, 'GAMMA', 0.65)
            ccf.GAMMA = val
            # Also patch in schema_core if it exists
            if hasattr(sc, 'GAMMA'):
                orig_vals['sc_GAMMA'] = sc.GAMMA
                sc.GAMMA = val

        elif param_name == 'tau_slow':
            orig_vals['TAU_SLOW'] = getattr(ccf, 'TAU_SLOW', 4000)
            ccf.TAU_SLOW = val
            if hasattr(sc, 'TAU_SLOW'):
                orig_vals['sc_TAU_SLOW'] = sc.TAU_SLOW
                sc.TAU_SLOW = val

        elif param_name == 'w_max':
            orig_vals['W_MAX'] = getattr(ccf, 'W_MAX', 1.5)
            ccf.W_MAX = val

        # Run experiment
        _CENTROID_LOG.clear(); _last_net[0] = None

        try:
            r = ccf.run_sequential_experiment(True, True, assemblies, SEED, ablation={})
        except Exception as e:
            print(f'[Q6] {cond_name} FAILED: {e}', flush=True)
            results[cond_name] = {'param': param_name, 'value': val, 'retention': np.nan, 'error': str(e)}
            continue
        finally:
            # Restore originals
            for k, v in orig_vals.items():
                if k.startswith('sc_'):
                    setattr(sc, k[3:], v)
                else:
                    setattr(ccf, k, v)

        # Extract retention from the result
        net = _last_net[0]
        if net is not None:
            ne = net.n_exc
            ret_scores = []
            for asm in assemblies:
                try:
                    ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
                except Exception:
                    ret_scores.append(0.0)
            ret_scores = np.nan_to_num(ret_scores, nan=0.0)
            mean_ret = float(np.mean(ret_scores))

            # W_slow stats
            with torch.no_grad():
                WS = net.W_slow.cpu().numpy()
            core_l = core.tolist()
            WScc = float(WS[np.ix_(core_l, core_l)].mean()) if len(core_l) > 0 else 0.0
        else:
            # Try getting retention from r dict
            if isinstance(r, dict) and 'final_scores' in r:
                mean_ret = float(np.mean(r['final_scores']))
                ret_scores = r['final_scores']
            else:
                mean_ret = np.nan
                ret_scores = [np.nan]*4
            WScc = np.nan

        elapsed = time.time() - t_start
        print(f'[Q6] {cond_name}: retention={mean_ret:.4f} WScc={WScc:.4f} ({elapsed:.1f}s)', flush=True)

        results[cond_name] = {
            'param': param_name,
            'value': val,
            'retention': mean_ret,
            'per_mem_ret': ret_scores if isinstance(ret_scores, list) else ret_scores.tolist(),
            'WScc': WScc,
            'elapsed': elapsed,
        }

        # Save intermediate
        with open(os.path.join(OUT_DIR, 'Q6_sweep_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

total_elapsed = time.time() - t_total_start
print(f'\n[Q6] ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)', flush=True)

# Final save
with open(os.path.join(OUT_DIR, 'Q6_sweep_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Summary table
print('\n[Q6] PARAMETER SENSITIVITY SUMMARY:')
print(f'{"Parameter":<12} {"Value":<10} {"Retention":<12} {"WScc":<10} {"vs Baseline":<12}')
print('-' * 60)
for param_name, values in SWEEPS.items():
    # Print baseline
    print(f'{param_name:<12} {BASELINE[param_name]:<10} {BASELINE_RET:<12.4f} {"—":<10} {"baseline":<12}')
    for val in values:
        cond = f'{param_name}_{val}'
        if cond in results:
            d = results[cond]
            delta = d['retention'] - BASELINE_RET if not np.isnan(d['retention']) else np.nan
            pct = delta / BASELINE_RET * 100 if not np.isnan(delta) else np.nan
            print(f'{"":12} {val:<10} {d["retention"]:<12.4f} {d.get("WScc", np.nan):<10.4f} {f"{pct:+.1f}%":<12}')
    print()

print('[Q6] Results saved to', os.path.join(OUT_DIR, 'Q6_sweep_results.pkl'))
