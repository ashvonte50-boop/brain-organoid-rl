"""
Q5: Dose-Response Suppression Curve
====================================
Vary M0 suppression bias from 1.0 (control) to 0.0 (total suppression)
in 5 levels. Single seed (42), DEV_MODE.

Levels: bias = [1.0, 0.5, 0.2, 0.1, 0.0]
For each level, bias_vec = [bias, 1, 1, 1]
"""
import os, sys, pickle, time
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
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

SEED = 42
N_MEM = 4
OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\q5_dose'
os.makedirs(OUT_DIR, exist_ok=True)

# 5 suppression levels for M0
BIAS_LEVELS = [1.0, 0.5, 0.2, 0.1, 0.0]

t_total_start = time.time()

# Build assemblies once
ccf.torch.manual_seed(SEED)
ccf.np.random.seed(SEED)
assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
assemblies_np = [np.array(a) for a in assemblies]
core = np.asarray(core_mask, dtype=np.int64)

ne_est = 750
core_set = set(core.tolist())
unique_all = np.array(sorted(set(
    int(i) for asm in assemblies for i in asm
    if int(i) not in core_set and int(i) < ne_est
)), dtype=np.int64)

per_mem_uniq = {}
for i, asm in enumerate(assemblies):
    per_mem_uniq[i] = np.array([x for x in asm if int(x) not in core_set and int(x) < ne_est],
                                dtype=np.int64)

print(f'[Q5] seed={SEED} core={len(core)} unique={len(unique_all)} n_mem={N_MEM}', flush=True)
print(f'[Q5] Suppression levels: {BIAS_LEVELS}', flush=True)

results = {}

for bias_m0 in BIAS_LEVELS:
    cond_name = f'bias_{bias_m0:.1f}'
    bias_raw = np.array([bias_m0, 1.0, 1.0, 1.0])

    # Handle bias_m0 = 0: set tiny epsilon to avoid div-by-zero
    if bias_raw.sum() == 0:
        bias_probs = np.array([0.0, 1/3, 1/3, 1/3])
    else:
        bias_probs = bias_raw / bias_raw.sum()

    t_cond_start = time.time()
    print(f'\n[Q5] === {cond_name} === bias={bias_raw.tolist()} probs={bias_probs.tolist()}', flush=True)

    _net_ref = [None]
    _replay_log = []

    _orig_build = ccf.build_network
    def _track_build(use_slow=False):
        n = _orig_build(use_slow=use_slow)
        _net_ref[0] = n
        return n
    ccf.build_network = _track_build

    _orig_replay = ccf._replay_one_event

    def make_biased_replay(bp, asmb_list, core_arr, replay_log, net_ref):
        def _biased_replay(net, assembly, tags=None, **kw):
            net_ref[0] = net
            _last_net[0] = net
            p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
            chosen_mem = int(np.random.choice(N_MEM, p=bp))
            actual_asm = asmb_list[chosen_mem]
            result = _orig_replay(net, actual_asm, tags=tags, **p, **kw)
            # MB boost on core
            ne = net.n_exc
            if len(core_arr) > 0:
                with torch.no_grad():
                    w = net.W.data[:ne, :ne]
                    ci_t = torch.as_tensor(core_arr, device=w.device, dtype=torch.long)
                    w[ci_t[:, None], ci_t[None, :]] *= 1.3
                    w.clamp_(0.0, net.w_max)
            replay_log.append(chosen_mem)
            return result
        return _biased_replay

    ccf._replay_one_event = make_biased_replay(bias_probs, assemblies_np, core, _replay_log, _net_ref)
    _CENTROID_LOG.clear(); _last_net[0] = None; _net_ref[0] = None
    _replay_log.clear()

    ccf.torch.manual_seed(SEED)
    ccf.np.random.seed(SEED)

    try:
        r = ccf.run_sequential_experiment(True, True, assemblies, SEED, ablation={})
    finally:
        ccf._replay_one_event = _orig_replay
        ccf.build_network = _orig_build

    net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
    assert net is not None, f'{cond_name}: network not captured'
    ne = net.n_exc

    def bmean(M, r, c):
        r = np.asarray(r); c = np.asarray(c)
        if len(r) == 0 or len(c) == 0: return 0.0
        return float(M[np.ix_(r, c)].mean())

    with torch.no_grad():
        WS = net.W_slow.cpu().numpy()

    core_l = core.tolist()
    uniq_l = unique_all.tolist()
    WScc = bmean(WS, core_l, core_l)
    WSuc = bmean(WS, uniq_l, core_l)

    per_mem_ws = {}
    for mi, ui in per_mem_uniq.items():
        if len(ui) > 0:
            per_mem_ws[mi] = float(WS[np.ix_(ui.tolist(), ui.tolist())].mean())

    ret_scores = []
    for asm in assemblies:
        try:
            ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret_scores.append(0.0)
    ret_scores = np.nan_to_num(ret_scores, nan=0.0)

    from collections import Counter
    mc = Counter(_replay_log)
    replay_counts = [mc.get(i, 0) for i in range(N_MEM)]

    elapsed = time.time() - t_cond_start
    print(f'[Q5] {cond_name} done in {elapsed:.1f}s', flush=True)
    print(f'[Q5]   replay_counts={replay_counts}', flush=True)
    print(f'[Q5]   retention={[f"{r:.4f}" for r in ret_scores]}  mean={np.mean(ret_scores):.4f}', flush=True)
    print(f'[Q5]   M0_retention={ret_scores[0]:.4f}  M0_replays={replay_counts[0]}', flush=True)
    print(f'[Q5]   WScc={WScc:.4f}  WSuc={WSuc:.4f}', flush=True)

    results[cond_name] = {
        'bias_m0': bias_m0,
        'bias_raw': bias_raw.tolist(),
        'replay_counts': replay_counts,
        'retention': ret_scores.tolist(),
        'WScc': WScc, 'WSuc': WSuc,
        'per_mem_ws': per_mem_ws,
        'elapsed': elapsed,
    }

total_elapsed = time.time() - t_total_start
print(f'\n[Q5] ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)', flush=True)

with open(os.path.join(OUT_DIR, 'Q5_dose_response.pkl'), 'wb') as f:
    pickle.dump(results, f)
print('[Q5] Results saved.', flush=True)

# Quick summary table
print('\n[Q5] DOSE-RESPONSE SUMMARY:')
print(f'{"Bias M0":<10} {"M0 replays":<12} {"M0 ret":<10} {"Mean ret":<10} {"WScc":<10}')
for bias_m0 in BIAS_LEVELS:
    cond = f'bias_{bias_m0:.1f}'
    d = results[cond]
    print(f'{bias_m0:<10.1f} {d["replay_counts"][0]:<12} {d["retention"][0]:<10.4f} {np.mean(d["retention"]):<10.4f} {d["WScc"]:<10.4f}')
