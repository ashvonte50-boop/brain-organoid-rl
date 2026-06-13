"""
TASK 6 WORKER — Identify the True Replay-Protected Memory Substrate
===================================================================
Trains ONE FULL network per seed (identical protocol to Task 5), then applies
6 post-hoc weight-block destructions to the SAME trained network and re-probes
each.  Any retention difference is causally attributable to the destroyed block.

Interventions (on the trained excitatory weight matrix W[:n_exc,:n_exc]):

  CONTROL              : no modification (= FULL)
  DESTROY_WUC          : zero unique<->core block (both directions)
  DESTROY_WUU          : zero unique<->unique block
  DESTROY_WUC_WUU      : zero Wuc AND Wuu (Wcc + background kept)
  DESTROY_ALL_NON_CORE : zero everything EXCEPT the core<->core block
  DESTROY_ALL          : zero the entire excitatory block (sanity floor)

Goal: find which destruction reproduces the replay-removal phenotype
(~87% retention loss), i.e. which block actually stores the memory.
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T6')
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

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task6'
os.makedirs(OUT_DIR, exist_ok=True)

CORE = SCHEMA_CORE_SIZE  # 20
CONDS = ['CONTROL', 'DESTROY_WUC', 'DESTROY_WUU', 'DESTROY_WUC_WUU',
         'DESTROY_ALL_NON_CORE', 'DESTROY_ALL']


# ── net capture + replay wrapper (standard FULL: MB boost, replay on) ─────────
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

    ret_scores = []
    for asm in assemblies:
        try:
            ret_scores.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret_scores.append(0.0)
    ret_scores = np.nan_to_num(ret_scores, nan=0.0)

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


# ── Train once (shared across all 6 interventions) ───────────────────────────
print(f'[T6] seed={args.seed} — training FULL (shared across 6 interventions)', flush=True)
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
ne = net.n_exc

# ── Build core / unique index sets ───────────────────────────────────────────
core = np.asarray(core_mask)
core_set = set(int(x) for x in core)
unique = sorted(set(int(i) for asm in assemblies for i in asm
                    if int(i) not in core_set and int(i) < ne))
unique = np.array(unique, dtype=np.int64)

core_idx = torch.as_tensor(core, device=net.W.device, dtype=torch.long)
uniq_idx = torch.as_tensor(unique, device=net.W.device, dtype=torch.long)

print(f'[T6] core n={len(core)}  unique n={len(unique)}  n_exc={ne}', flush=True)

# Snapshot trained weights
with torch.no_grad():
    W_trained = net.W.data.clone()
    cc_block_trained = W_trained[core_idx[:, None], core_idx[None, :]].clone()


def apply_intervention(cond):
    """Restore trained weights, then zero the relevant block(s)."""
    with torch.no_grad():
        net.W.data.copy_(W_trained)
        W = net.W.data
        if cond == 'CONTROL':
            pass
        elif cond == 'DESTROY_WUC':
            W[uniq_idx[:, None], core_idx[None, :]] = 0.0
            W[core_idx[:, None], uniq_idx[None, :]] = 0.0
        elif cond == 'DESTROY_WUU':
            W[uniq_idx[:, None], uniq_idx[None, :]] = 0.0
        elif cond == 'DESTROY_WUC_WUU':
            W[uniq_idx[:, None], core_idx[None, :]] = 0.0
            W[core_idx[:, None], uniq_idx[None, :]] = 0.0
            W[uniq_idx[:, None], uniq_idx[None, :]] = 0.0
        elif cond == 'DESTROY_ALL_NON_CORE':
            # Zero entire excitatory block, then restore ONLY core-core
            W[:ne, :ne] = 0.0
            W[core_idx[:, None], core_idx[None, :]] = cc_block_trained
        elif cond == 'DESTROY_ALL':
            W[:ne, :ne] = 0.0
        else:
            raise ValueError(cond)


# ── Apply each intervention post-hoc ─────────────────────────────────────────
results = {}
for cond in CONDS:
    apply_intervention(cond)
    m = measure(net, assemblies, core_mask)
    m['replay_events'] = int(_replay_count[0])
    results[cond] = m
    print(f'[T6] {cond:<22s} Wcc={m["Wcc"]:.4f} Wuc={m["Wuc"]:.4f} Wuu={m["Wuu"]:.4f} '
          f'S1={m["S1"]:.4f} Ret={m["retention_mean"]:.4f} Retr={m["retrieval_mean"]:.4f}',
          flush=True)

# Restore trained weights (cleanliness)
with torch.no_grad():
    net.W.data.copy_(W_trained)

out = {
    'seed': args.seed,
    'replay_events': int(_replay_count[0]),
    'conditions': results,
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': core.tolist(),
    'unique_idx': unique.tolist(),
    'n_exc': int(ne),
}
out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T6] SAVED {out_path}', flush=True)
