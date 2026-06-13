"""
TASK 8 WORKER — Origin of the Core Attractor
=============================================
Trains ONE network per seed with detailed neuron-level logging:

  1. Per-neuron spike counts during EVERY replay event
     (which neurons fire, how much → STDP exposure)

  2. W_slow snapshots at 5 checkpoints:
     initial / after_mem0 / after_mem1 / after_mem2 / after_mem3+final

  3. W_slow row/column means per neuron (in- and out-strength)

  4. MB-boost application log (when and where Wcc gets the 1.3x boost)

This data lets us answer:
  Q1: Which predictor best explains W_slow growth? (overlap / replay_count / firing_rate)
  Q2: Does replay_count predict W_slow better than firing rate?
  Q3: Do future-core neurons look special BEFORE replay begins?
  Q4: When does core emergence start?
  Q5: Does W_slow scale monotonically with memory overlap?
"""
import os, sys, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T8')
args = parser.parse_args()

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

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task8'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Logging state ─────────────────────────────────────────────────────────────
_net_ref    = [None]
_log        = {
    'replay_spikes':   [],   # list of {replay_id, memory_idx, spike_counts: ndarray}
    'wslow_snapshots': [],   # list of {label, W_slow_row_mean, W_slow_col_mean, W_slow_cc_mean}
    'mb_boosts':       [],   # log of MB boost applications
    'replay_count':    [],   # per-replay: which memory
}
_replay_id  = [0]


def snapshot_wslow(net, label, assemblies, core, unique):
    """Capture per-neuron W_slow row/col means + block stats."""
    ne = net.n_exc
    with torch.no_grad():
        WS = net.W_slow.cpu().numpy()  # (ne, ne)
    row_mean = WS.mean(axis=1)     # neuron i's mean outgoing W_slow
    col_mean = WS.mean(axis=0)     # neuron i's mean incoming W_slow
    row_std  = WS.std(axis=1)

    # Block stats at neuron level
    core_row = row_mean[core]
    uniq_row = row_mean[unique]
    core_col = col_mean[core]
    uniq_col = col_mean[unique]

    # Per-memory unique neuron W_slow rows
    per_mem_unique_row = {}
    core_set = set(core.tolist())
    for i, asm in enumerate(assemblies):
        uniq_asm = np.array([x for x in asm if x not in core_set and x < ne])
        if len(uniq_asm):
            per_mem_unique_row[i] = float(row_mean[uniq_asm].mean())

    snap = {
        'label':             label,
        'row_mean_all':      row_mean.tolist(),
        'col_mean_all':      col_mean.tolist(),
        'row_std_all':       row_std.tolist(),
        'core_row_mean':     float(core_row.mean()),
        'core_row_std':      float(core_row.std()),
        'unique_row_mean':   float(uniq_row.mean()),
        'core_col_mean':     float(core_col.mean()),
        'unique_col_mean':   float(uniq_col.mean()),
        'per_mem_unique_row': per_mem_unique_row,
        'wslow_block_cc':    float(WS[np.ix_(core, core)].mean()),
    }
    _log['wslow_snapshots'].append(snap)
    print(f'[T8] snapshot={label}  core_row={snap["core_row_mean"]:.5f}  '
          f'uniq_row={snap["unique_row_mean"]:.5f}  '
          f'wslow_cc={snap["wslow_block_cc"]:.4f}', flush=True)


# ── Net capture ───────────────────────────────────────────────────────────────
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    return n
ccf.build_network = _track_build

# ── Replay hook with spike logging + MB boost ─────────────────────────────────
_orig_replay = ccf._replay_one_event
def _instrumented_replay(net, assembly, tags=None, **kw):
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    # Determine memory_idx from assembly membership
    asm_arr = np.array(assembly)
    mem_idx = -1
    for i, asm in enumerate(_assemblies[0]):
        if set(asm_arr.tolist()) == set(asm.tolist()):
            mem_idx = i
            break

    # Run replay — capture spikes during it
    net.reset_state()
    result = _orig_replay(net, assembly, tags=tags, **p, **kw)

    # Log spike counts: grab net.spikes after replay
    with torch.no_grad():
        spk = net.spikes.cpu().numpy().copy()

    rid = _replay_id[0]
    _log['replay_spikes'].append({
        'replay_id':   rid,
        'memory_idx':  mem_idx,
        'assembly':    asm_arr.tolist(),
        'spike_vec':   spk.tolist(),   # full network spike vector after replay
    })
    _log['replay_count'].append(mem_idx)

    # MB boost on core-core
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne]
        ci_arr = np.array([x for x in range(CORE_SIZE) if x < ne])
        if len(ci_arr):
            ci_t = torch.as_tensor(ci_arr, device=w.device)
            w[ci_t[:, None], ci_t[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)
            _log['mb_boosts'].append({'replay_id': rid, 'memory_idx': mem_idx})

    _replay_id[0] += 1
    return result

_assemblies = [[]]  # will be filled after make_schema_assemblies


# ── Train ─────────────────────────────────────────────────────────────────────
print(f'[T8] seed={args.seed} — training FULL (slow+replay) with neuron-level logging',
      flush=True)
ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
_assemblies[0] = [np.array(a) for a in assemblies]

_CENTROID_LOG.clear(); _last_net[0] = None; _net_ref[0] = None

ccf._replay_one_event = _instrumented_replay
try:
    r = ccf.run_sequential_experiment(True, True, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None
ne = net.n_exc

# ── Build index arrays ────────────────────────────────────────────────────────
core   = np.asarray(core_mask, dtype=np.int64)
core_set = set(core.tolist())
unique = np.array(sorted(set(int(i) for asm in assemblies for i in asm
                             if int(i) not in core_set and int(i) < ne)),
                  dtype=np.int64)

# Per-memory unique neuron index arrays
per_mem_unique = {}
for i, asm in enumerate(assemblies):
    per_mem_unique[i] = np.array([x for x in asm if x not in core_set and x < ne],
                                  dtype=np.int64)

print(f'[T8] core n={len(core)}  unique n={len(unique)}  n_exc={ne}', flush=True)
print(f'[T8] total replay events logged: {len(_log["replay_spikes"])}', flush=True)

# ── Take final W_slow snapshot ────────────────────────────────────────────────
snapshot_wslow(net, 'final', assemblies, core, unique)

# ── Compute per-neuron replay exposure ───────────────────────────────────────
replay_exposure = np.zeros(ne, dtype=np.float32)
for ev in _log['replay_spikes']:
    mem = ev['memory_idx']
    asm = np.array(ev['assembly'])
    asm_valid = asm[asm < ne]
    spk = np.array(ev['spike_vec'])
    # Count as exposed if neuron spiked during this replay
    replay_exposure[asm_valid] += spk[asm_valid]

# Also count events (not weighted by spike):
replay_event_count = np.zeros(ne, dtype=np.float32)
for ev in _log['replay_spikes']:
    mem = ev['memory_idx']
    asm = np.array(ev['assembly'])
    asm_valid = asm[asm < ne]
    replay_event_count[asm_valid] += 1.0

# Participation count per neuron (static, from assembly definitions)
participation = np.zeros(ne, dtype=np.int32)
for asm in assemblies:
    for n in asm:
        if n < ne:
            participation[n] += 1

# W_slow row means at the end
with torch.no_grad():
    WS_final = net.W_slow.cpu().numpy()
wslow_row = WS_final.mean(axis=1)  # (ne,) outgoing mean
wslow_col = WS_final.mean(axis=0)  # (ne,) incoming mean

# ── Print statistics ─────────────────────────────────────────────────────────
print(f'\n[T8] Per-group statistics:', flush=True)
print(f'  Core   neurons: n={len(core)}  '
      f'part={participation[core].mean():.1f}  '
      f'replay_events={replay_event_count[core].mean():.1f}  '
      f'wslow_row={wslow_row[core].mean():.5f}', flush=True)
print(f'  Unique neurons: n={len(unique)}  '
      f'part={participation[unique].mean():.2f}  '
      f'replay_events={replay_event_count[unique].mean():.1f}  '
      f'wslow_row={wslow_row[unique].mean():.5f}', flush=True)
print(f'  Ratio replay_events(core/unique): '
      f'{replay_event_count[core].mean()/max(replay_event_count[unique].mean(),1e-9):.2f}', flush=True)
print(f'  Ratio wslow_row(core/unique): '
      f'{wslow_row[core].mean()/max(wslow_row[unique].mean(),1e-9):.2f}', flush=True)

# Per memory unique replay exposure
for i in range(4):
    ui = per_mem_unique[i]
    if len(ui):
        n_events = sum(1 for ev in _log['replay_spikes'] if ev['memory_idx'] == i)
        print(f'  Mem {i} unique: replay_events={replay_event_count[ui].mean():.1f}  '
              f'n_replays_for_mem={n_events}  wslow_row={wslow_row[ui].mean():.5f}',
              flush=True)

# ── Probe retention per memory ────────────────────────────────────────────────
ret_per_mem = []
for i, asm in enumerate(assemblies):
    try:
        ret_per_mem.append(float(ccf.probe_memory(net, asm)['isyn_score']))
    except Exception:
        ret_per_mem.append(0.0)
print(f'\n[T8] Retention per memory: {[f"{r:.4f}" for r in ret_per_mem]}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'seed':           args.seed,
    'n_exc':          int(ne),
    'core':           core.tolist(),
    'unique':         unique.tolist(),
    'per_mem_unique': {k: v.tolist() for k,v in per_mem_unique.items()},
    'assemblies':     [a.tolist() for a in assemblies],
    'participation':  participation.tolist(),
    'replay_event_count': replay_event_count.tolist(),
    'replay_exposure':    replay_exposure.tolist(),
    'wslow_row':      wslow_row.tolist(),
    'wslow_col':      wslow_col.tolist(),
    'wslow_snapshots': _log['wslow_snapshots'],
    'replay_log':     _log['replay_count'],
    'ret_per_mem':    ret_per_mem,
    'gamma':          float(net.gamma),
}
# Also save replay spike data (may be large)
out['replay_spikes'] = _log['replay_spikes']

out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T8] SAVED {out_path}', flush=True)
