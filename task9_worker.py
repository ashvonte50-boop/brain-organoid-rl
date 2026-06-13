"""
TASK 9 WORKER — Robustness and Generalization of Schema-Core Mechanism
=======================================================================
Single worker handles all sweeps via CLI arguments:

  --seed      int    random seed
  --n_mem     int    number of memories (2/4/6/8)
  --core_size int    core overlap size (0/10/20/40/80)
  --replay    0|1    whether to do replay (0=no, 1=yes)

Measures:
  retention_mean / per_memory
  retrieval_mean
  Wslow[cc], Wslow[uc], Wslow[uu]
  replay_count per memory
  schema_strength = Wslow[cc] - Wslow[uc]
"""
import os, sys, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('--seed',      type=int, required=True)
parser.add_argument('--n_mem',     type=int, default=4)
parser.add_argument('--core_size', type=int, default=20)
parser.add_argument('--replay',    type=int, default=1, choices=[0,1])
args = parser.parse_args()

COND_NAME = f'n{args.n_mem}_c{args.core_size}_r{args.replay}'
UNIQUE_SIZE_PER_MEM = 20  # fixed

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_experiments import make_schema_assemblies
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task9'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Net capture + replay hooks ─────────────────────────────────────────────────
_net_ref     = [None]
_replay_log  = []   # list of memory_idx per replay event
_core_ref    = [None]  # will be set after assembly creation

_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    return n
ccf.build_network = _track_build

_orig_replay = ccf._replay_one_event
def _wrapped_replay(net, assembly, tags=None, **kw):
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    result = _orig_replay(net, assembly, tags=tags, **p, **kw)

    # Determine which memory this is
    asm_set = set(int(x) for x in assembly)
    mem_idx = -1
    for i, asm in enumerate(_assemblies_ref[0]):
        if set(int(x) for x in asm) == asm_set:
            mem_idx = i; break
    _replay_log.append(mem_idx)

    # MB boost on core neurons only if core exists
    core_idx = _core_ref[0]
    if core_idx is not None and len(core_idx) > 0:
        with torch.no_grad():
            ne = net.n_exc
            w  = net.W.data[:ne, :ne]
            ci_t = torch.as_tensor(core_idx, device=w.device, dtype=torch.long)
            w[ci_t[:, None], ci_t[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)
    return result

_assemblies_ref = [[]]

# ── Train ─────────────────────────────────────────────────────────────────────
print(f'[T9] {COND_NAME} seed={args.seed}  n_mem={args.n_mem}  '
      f'core={args.core_size}  replay={args.replay}', flush=True)

ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)

# Build assemblies — handle core_size=0 gracefully
total_needed = args.core_size + args.n_mem * UNIQUE_SIZE_PER_MEM
if total_needed > 400:
    print(f'[T9] WARNING: need {total_needed} neurons, capping to 400', flush=True)
    unique_size = max(1, (400 - args.core_size) // max(args.n_mem, 1))
else:
    unique_size = UNIQUE_SIZE_PER_MEM

assemblies, core_mask = make_schema_assemblies(
    args.n_mem, args.core_size, unique_size
)
_assemblies_ref[0] = [np.array(a) for a in assemblies]
core = np.asarray(core_mask, dtype=np.int64)
_core_ref[0] = core if len(core) > 0 else None

print(f'[T9] assemblies={len(assemblies)} asm_size={len(assemblies[0])} '
      f'core_n={len(core)} unique_size={unique_size}', flush=True)

_replay_log.clear()
_CENTROID_LOG.clear()
_last_net[0] = None; _net_ref[0] = None

use_replay_flag = bool(args.replay)
ccf._replay_one_event = _wrapped_replay if use_replay_flag else _orig_replay

try:
    r = ccf.run_sequential_experiment(
        True, use_replay_flag, assemblies, args.seed, ablation={}
    )
finally:
    ccf._replay_one_event = _orig_replay

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None, "Network not captured"
ne = net.n_exc

# ── Build index sets ──────────────────────────────────────────────────────────
core_set = set(core.tolist())
unique_all = np.array(sorted(set(
    int(i) for asm in assemblies for i in asm
    if int(i) not in core_set and int(i) < ne
)), dtype=np.int64)

core_idx  = torch.as_tensor(core,        device=net.W.device, dtype=torch.long)
uniq_idx  = torch.as_tensor(unique_all,  device=net.W.device, dtype=torch.long)

# ── Measure weight blocks ─────────────────────────────────────────────────────
has_slow = hasattr(net, 'W_slow') and net.slow_enabled

with torch.no_grad():
    W  = net.W.data[:ne, :ne].cpu().numpy()
    WS = net.W_slow.cpu().numpy() if has_slow else np.zeros((ne,ne))

def block_mean(M, idx_r, idx_c):
    if len(idx_r) == 0 or len(idx_c) == 0: return 0.0
    return float(M[np.ix_(idx_r, idx_c)].mean())

core_np  = core.tolist()  if len(core) else []
uniq_np  = unique_all.tolist()

Wcc  = block_mean(W,  core_np, core_np)
Wuc  = block_mean(W,  uniq_np, core_np)
Wuu  = block_mean(W,  uniq_np, uniq_np)
WScc = block_mean(WS, core_np, core_np)
WSuc = block_mean(WS, uniq_np, core_np)
WSuu = block_mean(WS, uniq_np, uniq_np)

schema_strength = WScc - WSuc  # positive = core stronger than cross-block

# ── Probe retention & retrieval ───────────────────────────────────────────────
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

# ── Per-memory replay counts ──────────────────────────────────────────────────
from collections import Counter
mem_replay_counts = Counter(_replay_log)
replay_per_mem = [mem_replay_counts.get(i, 0) for i in range(args.n_mem)]
core_replay_total = len(_replay_log)  # core fires in ALL replay events

print(f'[T9] {COND_NAME} done:  replay_total={core_replay_total}  '
      f'replay_per_mem={replay_per_mem}', flush=True)
print(f'[T9] {COND_NAME}  Wcc={Wcc:.4f}  WScc={WScc:.4f}  WSuc={WSuc:.4f}  '
      f'WSuu={WSuu:.4f}  S={schema_strength:.4f}', flush=True)
print(f'[T9] {COND_NAME}  Ret={ret_scores.mean():.4f}  '
      f'Retr={retr_scores.mean():.4f}  per_mem={[f"{r:.3f}" for r in ret_scores]}',
      flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'seed':              args.seed,
    'n_mem':             args.n_mem,
    'core_size':         args.core_size,
    'replay':            args.replay,
    'cond_name':         COND_NAME,
    'n_exc':             int(ne),
    'has_slow':          has_slow,
    'gamma':             float(getattr(net, 'gamma', 0.0)),
    'Wcc': Wcc, 'Wuc': Wuc, 'Wuu': Wuu,
    'WScc': WScc, 'WSuc': WSuc, 'WSuu': WSuu,
    'schema_strength':   float(schema_strength),
    'retention_mean':    float(ret_scores.mean()),
    'retrieval_mean':    float(retr_scores.mean()),
    'retention_per_mem': ret_scores.tolist(),
    'retrieval_per_mem': retr_scores.tolist(),
    'replay_per_mem':    replay_per_mem,
    'core_replay_total': core_replay_total,
    'assemblies':        [a.tolist() for a in assemblies],
    'core_mask':         core.tolist(),
    'unique_size':       int(unique_size),
}

fname = f'T9_{COND_NAME}_seed{args.seed}.pkl'
out_path = os.path.join(OUT_DIR, fname)
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T9] SAVED {out_path}', flush=True)
