#!/usr/bin/env python3
"""
MOD-4: Inhibitory plasticity robustness (Vogels-Sprekeler iSTDP).

Toggles ccf.INHIBITORY_STDP between False (control) and True (iSTDP active).
For each condition: 10 seeds x (FULL, NO_REPLAY) = 20 runs.
Total: 40 runs.
"""
import os, sys, time, gc
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
RESULTS = os.path.join(OUT, 'mod4_istdp_results.csv')

SEEDS = [42 + 1000 * i for i in range(10)]
ISTDP_CONDS = [('no_istdp', False), ('with_istdp', True)]


def is_done(cond, seed, use_replay):
    if not os.path.exists(RESULTS):
        return False
    df = pd.read_csv(RESULTS)
    return ((df.cond == cond) & (df.seed == seed)
            & (df.use_replay == use_replay)).any()


def save_row(row):
    pd.DataFrame([row]).to_csv(RESULTS, mode='a', index=False,
                                header=not os.path.exists(RESULTS))


print("=" * 60, flush=True)
print("MOD-4: INHIBITORY PLASTICITY ROBUSTNESS", flush=True)
print("=" * 60, flush=True)
total = len(ISTDP_CONDS) * len(SEEDS) * 2
print(f"Total runs: {total}", flush=True)

run_n = 0
for cond_name, istdp_flag in ISTDP_CONDS:
    for seed in SEEDS:
        for use_replay in (True, False):
            run_n += 1
            if is_done(cond_name, seed, use_replay):
                print(f"  [{run_n}/{total}] skip {cond_name} seed={seed} replay={use_replay}", flush=True)
                continue
            t0 = time.time()
            try:
                ccf.INHIBITORY_STDP = istdp_flag
                torch.manual_seed(seed)
                np.random.seed(seed)
                assemblies, _ = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
                results = ccf.run_sequential_experiment(
                    True, use_replay, assemblies, seed, ablation={})
                R = results.get('retention_matrix')
                rets = []
                if R is not None:
                    for mi in range(4):
                        v = float(R[mi, -1])
                        if not np.isnan(v):
                            rets.append(v)
                mean_ret = float(np.mean(rets)) if rets else 0.0
                save_row({
                    'cond': cond_name, 'seed': seed, 'use_replay': use_replay,
                    'time_sec': time.time() - t0,
                    'mean_retention': mean_ret,
                    'M0_retention': rets[0] if len(rets) > 0 else None,
                    'M1_retention': rets[1] if len(rets) > 1 else None,
                    'M2_retention': rets[2] if len(rets) > 2 else None,
                    'M3_retention': rets[3] if len(rets) > 3 else None,
                    'ok': True, 'error': '',
                })
                print(f"  [{run_n}/{total}] {cond_name} seed={seed} replay={use_replay}: "
                      f"ret={mean_ret:.4f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                save_row({
                    'cond': cond_name, 'seed': seed, 'use_replay': use_replay,
                    'time_sec': time.time() - t0,
                    'mean_retention': None,
                    'M0_retention': None, 'M1_retention': None,
                    'M2_retention': None, 'M3_retention': None,
                    'ok': False, 'error': f"{type(e).__name__}: {str(e)[:200]}",
                })
                print(f"  [{run_n}/{total}] FAIL: {e}", flush=True)
            gc.collect()

ccf.INHIBITORY_STDP = False
print("[MOD-4] DONE", flush=True)
