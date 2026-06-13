#!/usr/bin/env python3
"""
MOD-0: Diagnostic profile.

Single baseline run at N=1000 (current default). Measures:
  - wall time (sec)
  - peak RSS (MB)
  - retention output
"""
import os, sys, time, gc, psutil
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import torch
import numpy as np
import compare_catastrophic_forgetting as ccf
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

ccf.DEV_MODE = True
ccf.N_WORKERS = 1

proc = psutil.Process(os.getpid())
gc.collect()

print("=" * 60)
print("MOD-0: DIAGNOSTIC PROFILE")
print("=" * 60)
print(f"N_NEURONS (current): {ccf.N_NEURONS}")
print(f"N_EXC: {ccf.N_EXC}, N_INH: {ccf.N_INH}")
print(f"Total RAM: {psutil.virtual_memory().total/1e9:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available/1e9:.2f} GB")
print(f"CUDA: {torch.cuda.is_available()}")
print()

mem_before = proc.memory_info().rss / 1e6
print(f"Memory before run: {mem_before:.1f} MB")
t0 = time.time()

torch.manual_seed(42)
np.random.seed(42)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
results = ccf.run_sequential_experiment(
    True, True, assemblies, 42, ablation={})

elapsed = time.time() - t0
mem_after = proc.memory_info().rss / 1e6

print(f"\nRun time: {elapsed:.1f} sec")
print(f"Memory after run: {mem_after:.1f} MB")
print(f"Memory delta: {mem_after - mem_before:.1f} MB")
print(f"Peak (estimated): {mem_after:.1f} MB")

# Project to larger N (assuming O(N^2) dense weight matrices)
print("\n=== PROJECTIONS (assuming dense O(N^2) weight storage) ===")
N0 = ccf.N_NEURONS
W_MB_at_N = lambda N: N * N * 4 / 1e6  # float32, single matrix
total_W_at_N = lambda N: W_MB_at_N(N) * 8  # W, W_init, W_slow, traces (rough)

for N in [1000, 2000, 5000, 10000]:
    factor = (N / N0) ** 2
    est_mem = mem_after + (total_W_at_N(N) - total_W_at_N(N0))
    est_time = elapsed * factor
    flag = "[INFEASIBLE]" if est_mem > 0.8 * psutil.virtual_memory().total/1e6 else "[ok]"
    print(f"  N={N:>5}: est mem {est_mem:>7.0f} MB, est time {est_time:>7.0f} sec {flag}")

print("\n[MOD-0] DONE")
