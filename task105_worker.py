"""
TASK 10.5 WORKER — Replay Allocation Intervention (Causal Test)
===============================================================
3 conditions, seed=42 only, DEV_MODE.

Conditions:
  CONTROL       bias=[1,1,1,1]
  BOOST_MEM3    bias=[1,1,1,3]
  SUPPRESS_MEM0 bias=[0.2,1,1,1]

Bias is applied by intercepting _replay_one_event and re-routing
to a biased-sampled assembly instead of the scheduled one.
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
OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task105'
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = {
    'CONTROL':       np.array([1.0, 1.0, 1.0, 1.0]),
    'BOOST_MEM3':    np.array([1.0, 1.0, 1.0, 3.0]),
    'SUPPRESS_MEM0': np.array([0.2, 1.0, 1.0, 1.0]),
}

t_total_start = time.time()

# ── Build assemblies once ────────────────────────────────────────────────────
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

print(f'[T105] seed={SEED} core={len(core)} unique={len(unique_all)} n_mem={N_MEM}', flush=True)

results = {}

for cond_name, bias_raw in CONDITIONS.items():
    t_cond_start = time.time()
    print(f'\n[T105] === {cond_name} === bias={bias_raw.tolist()}', flush=True)

    bias_probs = bias_raw / bias_raw.sum()

    # ── State for this condition ─────────────────────────────────────────────
    _net_ref   = [None]
    _replay_log = []  # which memory actually replayed

    _orig_build = ccf.build_network
    def _track_build(use_slow=False):
        n = _orig_build(use_slow=use_slow)
        _net_ref[0] = n
        return n
    ccf.build_network = _track_build

    _orig_replay = ccf._replay_one_event

    def make_biased_replay(bp, asmb_list, asmb_np, core_arr, replay_log, net_ref):
        def _biased_replay(net, assembly, tags=None, **kw):
            net_ref[0] = net
            _last_net[0] = net
            p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

            # Choose memory according to bias
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
            n_ev = len(replay_log)
            if n_ev % 10 == 0:
                print(f'[T105] {cond_name} event={n_ev} chose_mem={chosen_mem}', flush=True)
            return result
        return _biased_replay

    ccf._replay_one_event = make_biased_replay(
        bias_probs, assemblies_np, assemblies_np, core, _replay_log, _net_ref
    )
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

    # ── Measure weight blocks ────────────────────────────────────────────────
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
    WSuu = bmean(WS, uniq_l, uniq_l)

    per_mem_ws = {}
    for mi, ui in per_mem_uniq.items():
        if len(ui) > 0:
            per_mem_ws[mi] = float(WS[np.ix_(ui.tolist(), ui.tolist())].mean())

    # ── Probe retention ──────────────────────────────────────────────────────
    ret_scores = []
    for asm in assemblies:
        try:
            ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret_scores.append(0.0)
    ret_scores = np.nan_to_num(ret_scores, nan=0.0)

    # ── Replay counts ────────────────────────────────────────────────────────
    from collections import Counter
    mc = Counter(_replay_log)
    replay_counts = [mc.get(i, 0) for i in range(N_MEM)]
    total_events  = len(_replay_log)
    replay_fracs  = [c / max(total_events, 1) for c in replay_counts]

    elapsed = time.time() - t_cond_start
    print(f'[T105] {cond_name} done in {elapsed:.1f}s', flush=True)
    print(f'[T105]   replay_counts={replay_counts}  total={total_events}', flush=True)
    print(f'[T105]   retention={[f"{r:.4f}" for r in ret_scores]}  mean={ret_scores.mean():.4f}', flush=True)
    print(f'[T105]   WScc={WScc:.4f}  WSuc={WSuc:.4f}  WSuu={WSuu:.4f}', flush=True)
    for mi in range(N_MEM):
        print(f'[T105]   mem{mi}: ws={per_mem_ws.get(mi,0):.4f}  '
              f'replay={replay_counts[mi]}  ret={ret_scores[mi]:.4f}', flush=True)

    results[cond_name] = {
        'bias':           bias_raw.tolist(),
        'bias_probs':     bias_probs.tolist(),
        'replay_counts':  replay_counts,
        'replay_fracs':   replay_fracs,
        'total_events':   total_events,
        'retention':      ret_scores.tolist(),
        'WScc': WScc, 'WSuc': WSuc, 'WSuu': WSuu,
        'per_mem_ws':     per_mem_ws,
        'elapsed':        elapsed,
    }

    fname = f'T105_{cond_name}_seed{SEED}.pkl'
    with open(os.path.join(OUT_DIR, fname), 'wb') as f:
        pickle.dump(results[cond_name], f)
    print(f'[T105]   saved {fname}', flush=True)

total_elapsed = time.time() - t_total_start
print(f'\n[T105] ALL CONDITIONS DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)', flush=True)

# Save combined results
with open(os.path.join(OUT_DIR, 'T105_all_seed42.pkl'), 'wb') as f:
    pickle.dump(results, f)
print('[T105] Combined results saved.', flush=True)
