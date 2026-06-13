#!/usr/bin/env python3
"""
MOD-5: Biological-Range Parameter Sweep.

For each of 6 free parameters, vary it across its biological range,
running 3 seeds * 2 conditions (FULL, NO_REPLAY).

CONSERVATIVE COUNT (memory-aware):
  - 5 params x 4 values x 3 seeds x 2 cond = 120 runs at ~500s each = ~16.7 h
  - Will append to CSV so crashes/restarts don't lose progress.
"""
import os, sys, time, gc, traceback, psutil
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import pandas as pd
import torch
import compare_catastrophic_forgetting as ccf
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

ccf.DEV_MODE = True
ccf.N_WORKERS = 1

OUT = r'C:\Users\Admin\brain-organoid-rl\mod_results'
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, 'mod5_bio_param_sweep.csv')

# Biological-range sweep. Defaults marked with *.
BIO_SWEEP = {
    'GAMMA':           [0.40, 0.55, 0.65, 0.80],         # *0.65
    'TAU_SLOW':        [1000.0, 2000.0, 3000.0, 6000.0], # *3000
    'W_MAX':           [1.0, 1.25, 1.5, 1.75],           # *1.5
}
# Note: REPLAY_COHERENCE_THR, STDP_GATE_BIAS, MB_BOOST require deeper
# integration changes; skipped for this conservative sweep. See
# mod5_param_grounding.md for biological justification.

SEEDS = [42, 1042, 2042]

def is_done(param, value, seed, cond):
    if not os.path.exists(RESULTS): return False
    df = pd.read_csv(RESULTS)
    return ((df.param == param) & (df.value == value) &
            (df.seed == seed) & (df.condition == cond)).any()

def save_row(row):
    pd.DataFrame([row]).to_csv(RESULTS, mode='a', index=False,
                                header=not os.path.exists(RESULTS))

print("=" * 60, flush=True)
print("MOD-5: BIO-RANGE PARAMETER SWEEP", flush=True)
print("=" * 60, flush=True)

total_runs = sum(len(v) for v in BIO_SWEEP.values()) * len(SEEDS) * 2
print(f"Total runs: {total_runs} ({len(BIO_SWEEP)} params)", flush=True)
print(f"Available RAM: {psutil.virtual_memory().available/1e9:.2f} GB", flush=True)

orig = {
    'GAMMA': ccf.GAMMA,
    'TAU_SLOW': ccf.TAU_SLOW,
    'W_MAX': ccf.W_MAX,
}

run_n = 0
for param, values in BIO_SWEEP.items():
    for value in values:
        for seed in SEEDS:
            for cond_name, use_replay in [('FULL', True), ('NO_REPLAY', False)]:
                run_n += 1
                if is_done(param, value, seed, cond_name):
                    print(f"  [{run_n}/{total_runs}] Skip {param}={value} seed={seed} {cond_name}",
                          flush=True)
                    continue

                # Restore defaults, then perturb the target param
                for k, v in orig.items():
                    setattr(ccf, k, v)
                setattr(ccf, param, value)

                t0 = time.time()
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    assemblies, core_mask = make_schema_assemblies(
                        4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
                    results = ccf.run_sequential_experiment(
                        True, use_replay, assemblies, seed, ablation={})

                    # Extract from retention_matrix
                    R = results.get('retention_matrix')
                    ret_list = []
                    if R is not None:
                        for mi in range(4):
                            val = float(R[mi, -1])
                            if not np.isnan(val):
                                ret_list.append(val)
                    mean_ret = float(np.mean(ret_list)) if ret_list else 0.0

                    save_row({
                        'param': param, 'value': value,
                        'seed': seed, 'condition': cond_name,
                        'time_sec': time.time() - t0,
                        'mean_retention': mean_ret,
                        'M0_retention': ret_list[0] if len(ret_list) > 0 else None,
                        'M1_retention': ret_list[1] if len(ret_list) > 1 else None,
                        'M2_retention': ret_list[2] if len(ret_list) > 2 else None,
                        'M3_retention': ret_list[3] if len(ret_list) > 3 else None,
                        'ok': True,
                    })
                    print(f"  [{run_n}/{total_runs}] {param}={value} seed={seed} "
                          f"{cond_name}: ret={mean_ret:.3f}",
                          flush=True)
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:150]}"
                    save_row({
                        'param': param, 'value': value,
                        'seed': seed, 'condition': cond_name,
                        'time_sec': time.time() - t0,
                        'mean_retention': None,
                        'M0_retention': None, 'M1_retention': None,
                        'M2_retention': None, 'M3_retention': None,
                        'ok': False, 'error': err,
                    })
                    print(f"  [{run_n}/{total_runs}] FAIL: {err}", flush=True)
                gc.collect()

# Restore originals
for k, v in orig.items():
    setattr(ccf, k, v)

print("[MOD-5] SWEEP DONE", flush=True)
