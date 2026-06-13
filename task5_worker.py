"""
TASK 5 WORKER — Causal Role of Wcc
===================================
Trains ONE seed (full training + replay), then applies 4 post-hoc interventions
to the SAME trained network and re-probes each:

  FULL        : identity (no change)
  WCC_WEAKEN  : core-core weights *= 0.5
  WCC_DESTROY : core-core weights = 0
  WCC_ENHANCE : core-core weights *= 1.5

All 4 conditions share the identical trained network, so any retention
difference is causally attributable to the Wcc edit alone. Only the core-core
block W[0:20, 0:20] is modified; all other weights untouched.
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T5')
args = parser.parse_args()

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task5'
os.makedirs(OUT_DIR, exist_ok=True)

CORE = SCHEMA_CORE_SIZE   # 20

INTERVENTIONS = {
    'FULL':        1.0,    # identity
    'WCC_WEAKEN':  0.5,
    'WCC_DESTROY': 0.0,
    'WCC_ENHANCE': 1.5,
}

# ── net capture + replay wrapper (standard FULL: MB boost, replay on) ─────
_net_ref = [None]
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    return n
ccf.build_network = _track_build

_replay_count = [0]
_orig_replay = ccf._replay_one_event
def _wrapped_replay(net, assembly, tags=None, **kw):
    _replay_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    result = _orig_replay(net, assembly, tags=tags, **p, **kw)
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne]
        ci = np.array([x for x in range(CORE_SIZE) if x < ne])
        if len(ci):
            ci_t = torch.as_tensor(ci, device=w.device)
            w[ci_t[:, None], ci_t[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)
    return result


def measure(net, assemblies, core_mask):
    """Probe all memories; compute Wcc/Wuc/Wuu/S1/retention/retrieval."""
    core = np.asarray(core_mask)
    with torch.no_grad():
        W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
    Wcc = float(W[np.ix_(core, core)].mean())
    uc_list, uu_list = [], []
    for asm in assemblies:
        uniq = np.array([i for i in asm if i not in core and i < W.shape[0]])
        if len(uniq):
            uc_list.append(W[np.ix_(uniq, core)].mean())
            uu_list.append(W[np.ix_(uniq, uniq)].mean())
    Wuc = float(np.mean(uc_list)) if uc_list else 1e-9
    Wuu = float(np.mean(uu_list)) if uu_list else 1e-9

    # Retention via probe isyn_score (matches all prior tasks)
    ret_scores = []
    for asm in assemblies:
        try:
            ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret_scores.append(0.0)
    ret_scores = np.nan_to_num(ret_scores, nan=0.0)

    # Retrieval accuracy via completion_accuracy
    retr = []
    for asm in assemblies:
        try:
            retr.append(float(ccf.completion_accuracy(net, asm)['completion_frac']))
        except Exception:
            retr.append(0.0)
    retr = np.nan_to_num(retr, nan=0.0)

    return {
        'Wcc': Wcc, 'Wuc': Wuc, 'Wuu': Wuu, 'S1': Wcc - Wuc,
        'retention_per_memory': ret_scores.tolist(),
        'retention_mean': float(ret_scores.mean()),
        'retrieval_per_memory': retr.tolist(),
        'retrieval_mean': float(retr.mean()),
    }


# ── Train once ───────────────────────────────────────────────────────────
print(f'[T5] seed={args.seed} — training (shared across 4 conditions)', flush=True)
ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

_replay_count[0] = 0
_CENTROID_LOG.clear()
_last_net[0] = None
_net_ref[0] = None

ccf._replay_one_event = _wrapped_replay
try:
    r = ccf.run_sequential_experiment(True, True, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None

core = np.asarray(core_mask)
core_idx = torch.as_tensor(core, device=net.W.device, dtype=torch.long)

# Snapshot the trained weights
with torch.no_grad():
    W_trained = net.W.data.clone()

# ── Apply each intervention post-hoc ─────────────────────────────────────
results = {}
for cond, factor in INTERVENTIONS.items():
    with torch.no_grad():
        # Restore trained weights
        net.W.data.copy_(W_trained)
        # Apply intervention to core-core block ONLY
        net.W.data[core_idx[:, None], core_idx[None, :]] *= factor
        # (DESTROY: factor=0 -> exactly zero. No clamp needed; all >=0.)
    m = measure(net, assemblies, core_mask)
    m['intervention_factor'] = factor
    m['replay_events'] = int(_replay_count[0])
    results[cond] = m
    print(f'[T5] {cond:<12s} Wcc={m["Wcc"]:.4f} S1={m["S1"]:.4f} '
          f'Ret={m["retention_mean"]:.4f} Retr={m["retrieval_mean"]:.4f}', flush=True)

# Restore trained weights (cleanliness)
with torch.no_grad():
    net.W.data.copy_(W_trained)

out = {
    'seed': args.seed,
    'replay_events': int(_replay_count[0]),
    'conditions': results,
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': core.tolist(),
}
out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T5] SAVED {out_path}', flush=True)
