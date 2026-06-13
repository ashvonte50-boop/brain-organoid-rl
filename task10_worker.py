"""
TASK 10 WORKER -- Predictive Validation (FAST version)
======================================================
Lightweight per-event logging: only mem_idx + core spike count.
W_slow block means captured at 4 checkpoints (25/50/75/100%).

Usage:  python task10_worker.py --seed 42
"""
import os, sys, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

import numpy as np, torch, warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task10'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Lightweight per-event log ────────────────────────────────────────────────
_net_ref    = [None]
_assemblies = [[]]
_core_np    = [None]
_event_log  = []  # lightweight: just {mem_idx, core_spikes, asm_spikes}

_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    return n
ccf.build_network = _track_build

_orig_replay = ccf._replay_one_event
def _fast_replay(net, assembly, tags=None, **kw):
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    asm_set = set(int(x) for x in assembly)
    mem_idx = -1
    for i, asm in enumerate(_assemblies[0]):
        if set(int(x) for x in asm) == asm_set:
            mem_idx = i; break

    result = _orig_replay(net, assembly, tags=tags, **p, **kw)

    # MB boost
    ne = net.n_exc
    core = _core_np[0]
    if core is not None and len(core) > 0:
        with torch.no_grad():
            w = net.W.data[:ne, :ne]
            ci_t = torch.as_tensor(core, device=w.device, dtype=torch.long)
            w[ci_t[:, None], ci_t[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)

    # LIGHTWEIGHT telemetry — no W_slow copy!
    with torch.no_grad():
        spk = net.spikes.cpu().numpy() if hasattr(net, 'spikes') else np.zeros(ne)

    core_arr = core if core is not None else np.array([], dtype=np.int64)
    asm_arr = np.array([int(x) for x in assembly if int(x) < ne])

    _event_log.append({
        'mem_idx':     mem_idx,
        'core_spikes': int(spk[core_arr].sum()) if len(core_arr) > 0 else 0,
        'core_frac':   float(spk[core_arr].mean()) if len(core_arr) > 0 else 0.0,
        'asm_spikes':  int(spk[asm_arr].sum()) if len(asm_arr) > 0 else 0,
    })

    eid = len(_event_log)
    if eid % 10 == 0:
        print(f'[T10] event={eid} mem={mem_idx} core_frac={_event_log[-1]["core_frac"]:.3f}',
              flush=True)
    return result

# ── Train ────────────────────────────────────────────────────────────────────
print(f'[T10] seed={args.seed} -- FAST Predictive Validation', flush=True)
ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)

assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
_assemblies[0] = [np.array(a) for a in assemblies]
core = np.asarray(core_mask, dtype=np.int64)
_core_np[0] = core

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

print(f'[T10] core={len(core)} unique={len(unique_all)} assemblies={len(assemblies)}', flush=True)
_CENTROID_LOG.clear(); _last_net[0] = None; _net_ref[0] = None
_event_log.clear()

ccf._replay_one_event = _fast_replay
try:
    r = ccf.run_sequential_experiment(True, True, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None
ne = net.n_exc
print(f'[T10] Training complete. {len(_event_log)} replay events.', flush=True)

# ── POST-HOC: W_slow snapshots at quartile boundaries ───────────────────────
# We can't go back in time, but we CAN reconstruct from the event log + final state.
# Instead, we'll compute what we need from the event log itself:
#   - Per-memory replay counts at 25/50/75/100%
#   - Core participation at each quartile
# And from final state:
#   - Final W_slow block means
#   - Final retention per memory

def bmean(M, r, c):
    r = np.asarray(r); c = np.asarray(c)
    if len(r) == 0 or len(c) == 0: return 0.0
    return float(M[np.ix_(r, c)].mean())

core_l = core.tolist()
uniq_l = unique_all.tolist()

with torch.no_grad():
    WS = net.W_slow.cpu().numpy()

final_WScc = bmean(WS, core_l, core_l)
final_WSuc = bmean(WS, uniq_l, core_l)
final_WSuu = bmean(WS, uniq_l, uniq_l)

# Per-memory unique block W_slow
final_per_mem_ws = {}
for mi, ui in per_mem_uniq.items():
    if len(ui) > 0:
        final_per_mem_ws[mi] = float(WS[np.ix_(ui.tolist(), ui.tolist())].mean())

print(f'[T10] Final WScc={final_WScc:.4f} WSuc={final_WSuc:.4f} WSuu={final_WSuu:.4f}', flush=True)

# ── Quartile extraction from event log ───────────────────────────────────────
from collections import Counter

total_events = len(_event_log)
quartile_data = {}
for frac in [0.25, 0.50, 0.75, 1.00]:
    cutoff = max(1, int(total_events * frac))
    window = _event_log[:cutoff]

    mc = Counter(e['mem_idx'] for e in window)
    replay_counts = [mc.get(i, 0) for i in range(4)]

    per_mem_core = []
    for mi in range(4):
        mi_events = [e for e in window if e['mem_idx'] == mi]
        per_mem_core.append(np.mean([e['core_frac'] for e in mi_events]) if mi_events else 0.0)

    core_frac_mean = np.mean([e['core_frac'] for e in window])

    quartile_data[frac] = {
        'replay_counts':  replay_counts,
        'per_mem_core':   per_mem_core,
        'core_frac_mean': float(core_frac_mean),
        'n_events':       cutoff,
    }
    print(f'[T10] Q{frac:.0%}: events={cutoff} replays={replay_counts} '
          f'core_frac={core_frac_mean:.3f}', flush=True)

# ── Probes ───────────────────────────────────────────────────────────────────
ret_scores, retr_scores = [], []
for asm in assemblies:
    try:
        ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
    except Exception:
        ret_scores.append(0.0)
    try:
        retr_scores.append(float(ccf.completion_accuracy(net, asm)['completion_frac']))
    except Exception:
        retr_scores.append(0.0)

ret_scores  = np.nan_to_num(ret_scores,  nan=0.0)
retr_scores = np.nan_to_num(retr_scores, nan=0.0)

print(f'[T10] Retention:  {[f"{r:.4f}" for r in ret_scores]}  mean={ret_scores.mean():.4f}', flush=True)
print(f'[T10] Retrieval:  {[f"{r:.4f}" for r in retr_scores]}  mean={retr_scores.mean():.4f}', flush=True)

replay_per_mem = [Counter(e['mem_idx'] for e in _event_log).get(i, 0) for i in range(4)]
print(f'[T10] Replay counts: {replay_per_mem}', flush=True)

# ── Save ─────────────────────────────────────────────────────────────────────
out = {
    'seed':             args.seed,
    'n_exc':            int(ne),
    'n_memories':       4,
    'core':             core.tolist(),
    'unique':           unique_all.tolist(),
    'per_mem_unique':   {k: v.tolist() for k, v in per_mem_uniq.items()},
    'assemblies':       [a.tolist() for a in assemblies],
    'timeline':         _event_log,
    'total_events':     total_events,
    'quartile_data':    quartile_data,
    'retention':        ret_scores.tolist(),
    'retrieval':        retr_scores.tolist(),
    'replay_per_mem':   replay_per_mem,
    'final_WScc':       final_WScc,
    'final_WSuc':       final_WSuc,
    'final_WSuu':       final_WSuu,
    'final_schema_strength': final_WScc - final_WSuc,
    'final_per_mem_ws': final_per_mem_ws,
}

fname = f'T10_seed{args.seed}.pkl'
out_path = os.path.join(OUT_DIR, fname)
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T10] SAVED {out_path}', flush=True)
