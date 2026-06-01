#!/usr/bin/env python3
"""PHASE 5: Ablation - Schema Boost ON vs OFF."""
import sys, os
os.environ['DEV_MODE'] = 'True'  # MUST be set before ccf import
sys.path.insert(0, '.')
import time, pickle, numpy as np, torch
import compare_catastrophic_forgetting as ccf
from _distortion_paper import (
    N_EXC, N_SEEDS, BASE_SEED, SCHEMA_CORE_SIZE, UNIQUE_SIZE,
    make_schema_assemblies, compute_real_schema_index,
    _core_boost, _HYPER_NOISE_STD,
)
from schema_abstraction.schema_core import register_schema_hooks
register_schema_hooks()

_ORIG_REPLAY_HOLDER = None
_CENTROID_LOG = []

def _extract_cents(net, assemblies):
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne].cpu().numpy()
        cents = {}
        for i, asm in enumerate(assemblies):
            valid = [int(x) for x in asm if 0 <= int(x) < ne]
            if len(valid) > 0:
                cents[i] = w[np.ix_(valid, valid)].mean(axis=1)
        return cents

def _save_traj(r, net, mode, seed, assemblies, core_mask, log):
    stages = []
    snapshots = r.get('snapshots', [])
    asm_size = len(assemblies[0])
    for j in range(len(snapshots)):
        cents = {}
        for i in range(len(snapshots)):
            sv = snapshots[i][j]
            if sv is not None:
                try:
                    cents[i] = sv.reshape(asm_size, asm_size).mean(axis=1)
                except Exception:
                    cents[i] = None
            else:
                cents[i] = None
        stages.append({'stage_name': ['initial','post_B','post_C','post_D'][j], 'centroids': cents})
    if net is not None and hasattr(net, 'W'):
        W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
        fc = {}
        for i, asm in enumerate(assemblies):
            valid = [int(x) for x in asm if 0 <= int(x) < W.shape[0]]
            if len(valid) > 0:
                try:
                    fc[i] = W[np.ix_(valid, valid)].mean(axis=1)
                except Exception:
                    fc[i] = None
            else:
                fc[i] = None
        stages.append({'stage_name': 'final', 'centroids': fc})
    core_idx = np.where(np.array(core_mask))[0].tolist() if hasattr(core_mask,'__len__') else []
    asm_list = [a.tolist() if hasattr(a,'tolist') else list(a) for a in assemblies]
    data = {
        'mode': mode, 'seed': seed, 'assemblies': asm_list,
        'core_idx': core_idx,
        'trajectory': stages, 'replay_events': list(log),
        'baseline_scores': r.get('baseline_scores',[]).tolist(),
        'final_scores': r.get('final_scores',[]).tolist(),
    }
    fname = f"trajectory_{mode}_seed{seed}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(f"  [SAVED] {fname}", flush=True)

def run_ablation(boost_on=True, n_seeds=1):
    label = 'natural_boost_on' if boost_on else 'natural_boost_off'
    print(f"\n--- Natural Replay: {'BOOST ON' if boost_on else 'BOOST OFF'} ---")
    results = []
    for si in range(n_seeds):
        seed = 42 + si * 1000
        ccf.torch.manual_seed(seed); ccf.np.random.seed(seed)
        assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

        global _ORIG_REPLAY_HOLDER, _CENTROID_LOG
        _CENTROID_LOG = []
        p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

        def _wrapper(net, assembly, tags=None, **kw):
            global _ORIG_REPLAY_HOLDER
            cb = _extract_cents(net, assemblies)
            result = _ORIG_REPLAY_HOLDER(net, assembly, tags=tags, **p, **kw)
            with torch.no_grad():
                ne = net.n_exc
                w = net.W.data[:ne, :ne]
                if boost_on:
                    _core_boost(net)
                ca = _extract_cents(net, assemblies)
                _CENTROID_LOG.append({
                    'replay_id': kw.get('burst_id',0)*1000 + kw.get('event_id',0),
                    'memory_idx': kw.get('assembly_idx', -1),
                    'centroid_before': {k: v.tolist() for k,v in cb.items()},
                    'centroid_after': {k: v.tolist() for k,v in ca.items()},
                })
            return result

        # Set _CORE_INDICES for _core_boost
        core_indices_arr = np.where(np.array(core_mask))[0]
        import _distortion_paper as _dp
        _dp._CORE_INDICES = core_indices_arr

        old = ccf._replay_one_event
        _ORIG_REPLAY_HOLDER = ccf._replay_one_event
        ccf._replay_one_event = _wrapper

        try:
            r = ccf.run_sequential_experiment(True, True, assemblies, seed)
        except Exception as e:
            print(f"  CRASH: {e}", flush=True)
            ccf._replay_one_event = old
            import traceback; traceback.print_exc()
            continue
        ccf._replay_one_event = old

        fs = r['final_scores']
        bs = r['baseline_scores']
        if np.any(np.isnan(fs)) or np.any(np.isinf(fs)):
            fs = np.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)
            r['final_scores'] = fs
        net = r.get('net', None)
        if net is None:
            try: net = _dp._last_net
            except: net = None
        real_schema = compute_real_schema_index(net, assemblies, core_mask) if net else 0
        print(f"  A={fs[0]:.4f}  REAL_SCHEMA={real_schema:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        _save_traj(r, net, label, seed, assemblies, core_mask, _CENTROID_LOG)
        results.append({'seed': seed, 'retention': fs.tolist(), 'real_schema': real_schema})
    return results

if __name__ == '__main__':
    t0 = time.time()
    r_on = run_ablation(boost_on=True, n_seeds=1)
    r_off = run_ablation(boost_on=False, n_seeds=1)

    print(f"\n{'='*50}")
    print("ABLATION RESULTS")
    print(f"{'='*50}")
    print(f"{'Condition':<22s} {'Ret A':>8s} {'REAL_SCHEMA':>12s}")
    print('-' * 44)
    if r_on:  print(f"{'Natural Boost ON':<22s} {r_on[0]['retention'][0]:>8.4f} {r_on[0]['real_schema']:>12.4f}")
    if r_off: print(f"{'Natural Boost OFF':<22s} {r_off[0]['retention'][0]:>8.4f} {r_off[0]['real_schema']:>12.4f}")
    if r_on and r_off:
        print(f"\nBoost effect: Retention A={r_on[0]['retention'][0]-r_off[0]['retention'][0]:+.4f}")
        print(f"Boost effect: REAL_SCHEMA={r_on[0]['real_schema']-r_off[0]['real_schema']:+.4f}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
