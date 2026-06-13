#!/usr/bin/env python3
"""
MOD-1 worker: runs ONE (N, seed, condition) in a fresh process.
Called by mod1_scaling.py via subprocess. Prints JSON result to stdout.
"""
import os, sys, json, traceback, time
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import torch
import psutil

N        = int(sys.argv[1])
seed     = int(sys.argv[2])
use_rep  = sys.argv[3].lower() == 'true'

try:
    import compare_catastrophic_forgetting as ccf
    import schema_abstraction.schema_core as sc
    sc.register_schema_hooks()
    from schema_abstraction.schema_experiments import (
        make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    ccf.DEV_MODE   = True
    ccf.N_WORKERS  = 1

    # Patch N — fresh process so no stale singletons
    ccf.N_NEURONS = N
    ccf.N_INH     = N // 4
    ccf.N_EXC     = N - ccf.N_INH
    ccf.N_HC      = int(ccf.N_EXC * ccf.HC_RATIO)
    ccf.N_CTX     = ccf.N_EXC - ccf.N_HC

    # ── FIX 1: re-derive module-level constants frozen at import time ──
    # _MODULE_SIZE, BG_START, BG_END, ASSEMBLY_SIZE, CUE_SIZE, PARTIAL_CUE_SIZE
    # are computed at IMPORT TIME using the original N_EXC=750. They land
    # inside the assembly's module at larger N, contaminating isyn_score.
    REF_MODULE_SIZE = 750 // ccf.N_MODULES  # =93 at the reference N=1000
    ccf._MODULE_SIZE = ccf.N_EXC // ccf.N_MODULES
    ccf.ASSEMBLY_SIZE = max(20, ccf._MODULE_SIZE // 4)
    ccf.CUE_SIZE = max(5, ccf.ASSEMBLY_SIZE // 4)
    ccf.PARTIAL_CUE_SIZE = max(8, ccf.ASSEMBLY_SIZE // 2)
    ccf._BG_MODULE = (ccf.MEMORY_MODULE + 1) % ccf.N_MODULES
    ccf.BG_START = ccf._BG_MODULE * ccf._MODULE_SIZE
    ccf.BG_SIZE = min(30, ccf._MODULE_SIZE)
    ccf.BG_END = ccf.BG_START + ccf.BG_SIZE

    ccf._STRUCTURAL_MASK_EE = None
    ccf.CACHE_MASKS_TO_DISK = False
    ccf.clear_probe_cache()

    print(f"[PATCH] N_EXC={ccf.N_EXC} module_size={ccf._MODULE_SIZE} "
          f"BG=[{ccf.BG_START},{ccf.BG_END}) conn_prob={ccf.INTRA_MODULE_CONN_PROB:.4f}",
          flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Assembly pool scales proportionally; core/unique sizes stay fixed (20/20)
    scaled_pool = max(100, int(ccf.N_EXC * 400 / 750))
    assert scaled_pool <= ccf.N_EXC, f"pool {scaled_pool} > N_EXC {ccf.N_EXC}"
    assemblies, core_mask = make_schema_assemblies(
        4, SCHEMA_CORE_SIZE, UNIQUE_SIZE, total_neurons=scaled_pool)

    # Assertions
    net_check = ccf.build_network(use_slow=True)
    assert net_check.W.shape      == (ccf.N_NEURONS, ccf.N_NEURONS), \
        f"W {net_check.W.shape}"
    assert net_check.W_slow.shape == (ccf.N_EXC, ccf.N_EXC), \
        f"W_slow {net_check.W_slow.shape}"
    assert net_check.v.shape[0]    == ccf.N_NEURONS, f"v {net_check.v.shape}"
    assert net_check.I_syn.shape[0] == ccf.N_NEURONS, f"I_syn {net_check.I_syn.shape}"
    for ai, asm in enumerate(assemblies):
        assert max(asm) < ccf.N_EXC, \
            f"asm[{ai}] max index {max(asm)} >= N_EXC {ccf.N_EXC}"
    print(f"[ASSERT OK] W:{net_check.W.shape} W_slow:{net_check.W_slow.shape} "
          f"max_asm:{max(max(a) for a in assemblies)} N_EXC:{ccf.N_EXC}", flush=True)
    del net_check

    t0 = time.time()
    results = ccf.run_sequential_experiment(
        True, use_rep, assemblies, seed, ablation={})
    elapsed = time.time() - t0

    R = results.get('retention_matrix')
    rets = []
    if R is not None:
        for mi in range(4):
            v = float(R[mi, -1])
            if not np.isnan(v):
                rets.append(v)
    mean_ret = float(np.mean(rets)) if rets else 0.0

    out = {
        'ok': True, 'mean_retention': mean_ret,
        'M0': rets[0] if len(rets)>0 else None,
        'M1': rets[1] if len(rets)>1 else None,
        'M2': rets[2] if len(rets)>2 else None,
        'M3': rets[3] if len(rets)>3 else None,
        'time_sec': elapsed,
        'mem_peak_MB': psutil.Process().memory_info().rss / 1e6,
        'available_RAM_GB': psutil.virtual_memory().available / 1e9,
        'error': '',
    }
except Exception:
    tb = traceback.format_exc()
    print(f"[WORKER FAIL]\n{tb}", flush=True)
    out = {
        'ok': False, 'mean_retention': None,
        'M0': None, 'M1': None, 'M2': None, 'M3': None,
        'time_sec': 0, 'mem_peak_MB': 0, 'available_RAM_GB': 0,
        'error': tb[-400:],
    }

print('RESULT_JSON:' + json.dumps(out), flush=True)
