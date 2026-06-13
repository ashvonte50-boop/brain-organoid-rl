#!/usr/bin/env python3
"""
MOD-3: Learned-schema variant at 15 seeds with fine correlation sweep.

Extends E3 from 3 seeds/4 strengths to 15 seeds/10 strengths.
- 15 hand-assigned baseline runs (FULL + NO_REPLAY each, 2*15 = 30 runs)
- 10 correlation levels x 15 seeds, FULL + NO_REPLAY (300 runs)
- Total: 330 runs (append-safe).

Each row records: emergent core size, RGCC signatures, retention diff.
"""
import os, sys, time, gc, traceback, threading
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import numpy as np
import pandas as pd
import torch
import compare_catastrophic_forgetting as ccf
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net
from schema_abstraction.schema_experiments import (
    make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

ccf.DEV_MODE = True
ccf.N_WORKERS = 1

OUT = r'C:\Users\Admin\brain-organoid-rl\mod_results'
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, 'mod3_learned_schema_15seeds.csv')

N_MEM = 4
NE = 750
POOL = 400
ASSEMBLY_SIZE_E3 = 40
CORE_MIN_MEMBERSHIP = 3

# Reduced sweep: 5 seeds x 6 levels = 70 total runs (~22h)
CORR_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42 + 1000 * i for i in range(5)]


def generate_correlated_memories(shared_feature_strength=0.5, seed=42):
    rng = np.random.default_rng(seed)
    shared_drive = rng.random(POOL)
    asms = []
    for _ in range(N_MEM):
        unique = rng.random(POOL)
        comb = shared_feature_strength * shared_drive + (1 - shared_feature_strength) * unique
        idx = np.argsort(comb)[-ASSEMBLY_SIZE_E3:]
        asms.append(np.sort(idx).astype(int))
    return asms


def emergent_core(asms):
    from collections import Counter
    c = Counter()
    for a in asms:
        for n in a.tolist():
            c[n] += 1
    return np.array(sorted([n for n, k in c.items() if k >= CORE_MIN_MEMBERSHIP]), int), c


def is_done(seed, condition, use_replay):
    if not os.path.exists(RESULTS):
        return False
    df = pd.read_csv(RESULTS)
    return ((df.seed == seed) & (df.condition == condition)
            & (df.use_replay == use_replay)).any()


def save_row(row):
    pd.DataFrame([row]).to_csv(RESULTS, mode='a', index=False,
                                header=not os.path.exists(RESULTS))


def run_one(assemblies, seed, use_replay):
    _net_ref = [None]
    _orig_build = ccf.build_network
    _hb_time = [time.time()]

    def _track(use_slow=True):
        n = _orig_build(use_slow=use_slow)
        _net_ref[0] = n
        return n

    ccf.build_network = _track
    _CENTROID_LOG.clear()
    _last_net[0] = None
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)
    try:
        results = ccf.run_sequential_experiment(
            True, use_replay, assemblies, seed, ablation={})
    finally:
        ccf.build_network = _orig_build
    R = results.get('retention_matrix')
    rets = []
    if R is not None:
        for mi in range(N_MEM):
            v = float(R[mi, -1])
            if not np.isnan(v):
                rets.append(v)
    net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
    return float(np.mean(rets)) if rets else 0.0, net


def block_mean(W, rows, cols):
    rows = [r for r in rows if r < NE]
    cols = [c for c in cols if c < NE]
    if len(rows) < 1 or len(cols) < 1:
        return float('nan')
    return float(W[np.ix_(rows, cols)].mean())


print("=" * 60, flush=True)
print("MOD-3: LEARNED SCHEMA, 15 seeds, fine sweep", flush=True)
print("=" * 60, flush=True)
print(f"Seeds: {len(SEEDS)} | Corr levels: {CORR_SWEEP}", flush=True)
total_runs = len(SEEDS) * 2 + len(CORR_SWEEP) * len(SEEDS) * 2
print(f"Total runs: {total_runs}", flush=True)

run_n = 0
_hb_label = ['starting']
_hb_stop = threading.Event()

def _hb_thread():
    while not _hb_stop.wait(120):
        print(f"  [heartbeat] {_hb_label[0]} | {time.strftime('%H:%M:%S')}", flush=True)

threading.Thread(target=_hb_thread, daemon=True).start()

# ── Hand-assigned baseline ────────────────────────────────────────────────
for seed in SEEDS:
    for use_replay in (True, False):
        run_n += 1
        cond = 'hand_assigned'
        if is_done(seed, cond, use_replay):
            print(f"  [{run_n}/{total_runs}] skip {cond} seed={seed} replay={use_replay}", flush=True)
            continue
        t0 = time.time()
        _hb_label[0] = f'hand_assigned seed={seed} replay={use_replay} [{run_n}/{total_runs}]'
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            asms_h, _ = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
            mean_ret, net = run_one(asms_h, seed, use_replay)
            wcc = wuu = wuc = float('nan')
            core_size = SCHEMA_CORE_SIZE
            try:
                if net is not None:
                    with torch.no_grad():
                        WS = net.W_slow.cpu().numpy()
                    core_list = list(range(SCHEMA_CORE_SIZE))
                    uniq_all = sorted(set(int(n) for a in asms_h for n in a.tolist()) - set(core_list))
                    wcc = block_mean(WS, core_list, core_list)
                    wuu = block_mean(WS, uniq_all, uniq_all)
                    wuc = block_mean(WS, uniq_all, core_list)
            except Exception:
                pass
            save_row({
                'condition': cond, 'corr_strength': None,
                'seed': seed, 'use_replay': use_replay,
                'time_sec': time.time() - t0,
                'mean_retention': mean_ret,
                'emergent_core_size': core_size,
                'wslow_cc': wcc, 'wslow_uu': wuu, 'wslow_uc': wuc,
                'ok': True, 'error': '',
            })
            print(f"  [{run_n}/{total_runs}] {cond} seed={seed} replay={use_replay}: "
                  f"ret={mean_ret:.4f} wcc={wcc:.3f}", flush=True)
        except Exception as e:
            save_row({
                'condition': cond, 'corr_strength': None,
                'seed': seed, 'use_replay': use_replay,
                'time_sec': time.time() - t0,
                'mean_retention': None, 'emergent_core_size': None,
                'wslow_cc': None, 'wslow_uu': None, 'wslow_uc': None,
                'ok': False, 'error': f"{type(e).__name__}: {str(e)[:200]}",
            })
            print(f"  [{run_n}/{total_runs}] FAIL: {e}", flush=True)
        gc.collect()

# ── Learned-schema sweep ──────────────────────────────────────────────────
for corr in CORR_SWEEP:
    for seed in SEEDS:
        for use_replay in (True, False):
            run_n += 1
            cond = f'learned_{corr:.2f}'
            if is_done(seed, cond, use_replay):
                print(f"  [{run_n}/{total_runs}] skip {cond} seed={seed} replay={use_replay}", flush=True)
                continue
            t0 = time.time()
            _hb_label[0] = f'learned_{corr:.2f} seed={seed} replay={use_replay} [{run_n}/{total_runs}]'
            try:
                asms = [np.array(a) for a in generate_correlated_memories(corr, seed)]
                core, _ = emergent_core(asms)
                mean_ret, net = run_one(asms, seed, use_replay)
                wcc = wuu = wuc = float('nan')
                try:
                    if net is not None and len(core) >= 1:
                        with torch.no_grad():
                            WS = net.W_slow.cpu().numpy()
                        core_list = core.tolist()
                        uniq_all = sorted(set(int(n) for a in asms for n in a.tolist()) - set(core_list))
                        if len(core) >= 2:
                            wcc = block_mean(WS, core_list, core_list)
                        if len(uniq_all) >= 2:
                            wuu = block_mean(WS, uniq_all, uniq_all)
                        if len(uniq_all) >= 1 and len(core) >= 1:
                            wuc = block_mean(WS, uniq_all, core_list)
                except Exception:
                    pass
                save_row({
                    'condition': cond, 'corr_strength': corr,
                    'seed': seed, 'use_replay': use_replay,
                    'time_sec': time.time() - t0,
                    'mean_retention': mean_ret,
                    'emergent_core_size': int(len(core)),
                    'wslow_cc': wcc, 'wslow_uu': wuu, 'wslow_uc': wuc,
                    'ok': True, 'error': '',
                })
                print(f"  [{run_n}/{total_runs}] {cond} seed={seed} replay={use_replay}: "
                      f"ret={mean_ret:.4f} core={len(core)} wcc={wcc:.3f}", flush=True)
            except Exception as e:
                save_row({
                    'condition': cond, 'corr_strength': corr,
                    'seed': seed, 'use_replay': use_replay,
                    'time_sec': time.time() - t0,
                    'mean_retention': None, 'emergent_core_size': None,
                    'wslow_cc': None, 'wslow_uu': None, 'wslow_uc': None,
                    'ok': False, 'error': f"{type(e).__name__}: {str(e)[:200]}",
                })
                print(f"  [{run_n}/{total_runs}] FAIL: {e}", flush=True)
            gc.collect()

_hb_stop.set()
print("[MOD-3] DONE", flush=True)
