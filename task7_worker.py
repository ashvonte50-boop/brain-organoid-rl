"""
TASK 7 WORKER — Identify the True Memory Substrate
====================================================
Hypothesis: Memory is stored in W_slow (slow synaptic consolidation state),
not in the fast weight matrix W.  Task 6 interventions only zeroed W, leaving
W_slow intact — hence no retention collapse was observed.

This worker:
  1. Trains ONE FULL network per seed (use_slow=True, replay=True)
  2. Snapshots BOTH net.W and net.W_slow after post-training rest
  3. Applies 10 post-hoc interventions to the SAME trained network
  4. Probes retention (isyn_score) for each condition

Conditions:
  CONTROL           : no modification
  DESTROY_W_ALL     : zero net.W[:n_exc,:n_exc]  (replicates T6 DESTROY_ALL)
  DESTROY_WSLOW_ALL : zero net.W_slow             *** KEY TEST ***
  DESTROY_BOTH      : zero W and W_slow           (sanity floor)
  DESTROY_WSLOW_CC  : zero W_slow[core,core]
  DESTROY_WSLOW_UC  : zero W_slow[unique,core] + W_slow[core,unique]
  DESTROY_WSLOW_UU  : zero W_slow[unique,unique]
  DESTROY_WSLOW_NON_CC : keep only W_slow[cc], zero the rest
  WSLOW_ONLY        : zero W, keep W_slow  (isolates W_slow contribution)
  W_ONLY            : zero W_slow, keep W  (isolates W contribution)
"""
import os, sys, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T7')
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

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task7'
os.makedirs(OUT_DIR, exist_ok=True)

CORE = SCHEMA_CORE_SIZE  # 20

CONDS = [
    'CONTROL',
    'DESTROY_W_ALL',
    'DESTROY_WSLOW_ALL',
    'DESTROY_BOTH',
    'WSLOW_ONLY',
    'W_ONLY',
    'DESTROY_WSLOW_CC',
    'DESTROY_WSLOW_UC',
    'DESTROY_WSLOW_UU',
    'DESTROY_WSLOW_NON_CC',
]


# ── Net capture + replay wrapper (identical to Task 6) ────────────────────────
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
    """Probe all memories; return retention and weight block stats."""
    core = np.asarray(core_mask)
    with torch.no_grad():
        W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
        has_slow = hasattr(net, 'W_slow') and net.slow_enabled
        WS = net.W_slow.cpu().numpy() if has_slow else np.zeros_like(W)

    Wcc  = float(W[np.ix_(core, core)].mean())
    WScc = float(WS[np.ix_(core, core)].mean()) if has_slow else 0.0

    uc_list, uu_list, wsuc_list, wsuu_list = [], [], [], []
    for asm in assemblies:
        uniq = np.array([i for i in asm if i not in set(core.tolist()) and i < W.shape[0]])
        if len(uniq):
            uc_list.append(W[np.ix_(uniq, core)].mean())
            uu_list.append(W[np.ix_(uniq, uniq)].mean())
            if has_slow:
                wsuc_list.append(WS[np.ix_(uniq, core)].mean())
                wsuu_list.append(WS[np.ix_(uniq, uniq)].mean())

    Wuc  = float(np.mean(uc_list))  if uc_list  else 0.0
    Wuu  = float(np.mean(uu_list))  if uu_list  else 0.0
    WSuc = float(np.mean(wsuc_list)) if wsuc_list else 0.0
    WSuu = float(np.mean(wsuu_list)) if wsuu_list else 0.0

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
        'Wcc': Wcc, 'Wuc': Wuc, 'Wuu': Wuu,
        'WScc': WScc, 'WSuc': WSuc, 'WSuu': WSuu,
        'S1': Wcc - Wuc,
        'retention_per_memory': ret_scores.tolist(),
        'retention_mean': float(ret_scores.mean()),
        'retrieval_per_memory': retr.tolist(),
        'retrieval_mean': float(retr.mean()),
    }


# ── Train once (FULL: slow=True, replay=True) ─────────────────────────────────
print(f'[T7] seed={args.seed} — training FULL (slow+replay)', flush=True)
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
assert net is not None, "Network not captured"
ne = net.n_exc

# ── Build core / unique index sets ────────────────────────────────────────────
core = np.asarray(core_mask)
core_set = set(int(x) for x in core)
unique = sorted(set(int(i) for asm in assemblies for i in asm
                    if int(i) not in core_set and int(i) < ne))
unique = np.array(unique, dtype=np.int64)

core_idx = torch.as_tensor(core, device=net.W.device, dtype=torch.long)
uniq_idx = torch.as_tensor(unique, device=net.W.device, dtype=torch.long)

print(f'[T7] core n={len(core)}  unique n={len(unique)}  n_exc={ne}', flush=True)

has_slow = hasattr(net, 'W_slow') and net.slow_enabled
print(f'[T7] slow_enabled={has_slow}  gamma={getattr(net,"gamma",None)}', flush=True)

# ── Snapshot trained state ────────────────────────────────────────────────────
with torch.no_grad():
    W_trained = net.W.data.clone()
    WS_trained = net.W_slow.clone() if has_slow else None

# Log weight magnitudes for diagnostics
with torch.no_grad():
    w_sub  = W_trained[:ne, :ne]
    ws_sub = WS_trained if WS_trained is not None else torch.zeros(ne, ne)
    w_mean   = float(w_sub.mean())
    ws_mean  = float(ws_sub.mean())
    w_norm   = float(w_sub.norm())
    ws_norm  = float(ws_sub.norm())
print(f'[T7] W mean={w_mean:.4f} norm={w_norm:.2f}  '
      f'W_slow mean={ws_mean:.4f} norm={ws_norm:.2f}', flush=True)


def apply_intervention(cond):
    """Restore trained state, then zero the requested components."""
    with torch.no_grad():
        net.W.data.copy_(W_trained)
        if has_slow:
            net.W_slow.copy_(WS_trained)
        W  = net.W.data
        WS = net.W_slow if has_slow else None

        if cond == 'CONTROL':
            pass

        elif cond == 'DESTROY_W_ALL':
            # Zero entire fast excitatory block (mirrors T6 DESTROY_ALL)
            W[:ne, :ne] = 0.0

        elif cond == 'DESTROY_WSLOW_ALL':
            # Zero entire slow weight matrix — KEY TEST
            if WS is not None:
                WS.zero_()

        elif cond == 'DESTROY_BOTH':
            # Zero both — sanity floor
            W[:ne, :ne] = 0.0
            if WS is not None:
                WS.zero_()

        elif cond == 'WSLOW_ONLY':
            # Zero fast W, keep W_slow — isolates W_slow contribution
            W[:ne, :ne] = 0.0

        elif cond == 'W_ONLY':
            # Zero W_slow, keep fast W — isolates W contribution
            if WS is not None:
                WS.zero_()

        elif cond == 'DESTROY_WSLOW_CC':
            if WS is not None:
                WS[core_idx[:, None], core_idx[None, :]] = 0.0

        elif cond == 'DESTROY_WSLOW_UC':
            if WS is not None:
                WS[uniq_idx[:, None], core_idx[None, :]] = 0.0
                WS[core_idx[:, None], uniq_idx[None, :]] = 0.0

        elif cond == 'DESTROY_WSLOW_UU':
            if WS is not None:
                WS[uniq_idx[:, None], uniq_idx[None, :]] = 0.0

        elif cond == 'DESTROY_WSLOW_NON_CC':
            # Keep only W_slow[cc], zero everything else in W_slow
            if WS is not None:
                cc_block = WS[core_idx[:, None], core_idx[None, :]].clone()
                WS.zero_()
                WS[core_idx[:, None], core_idx[None, :]] = cc_block

        else:
            raise ValueError(f'Unknown condition: {cond}')


# ── Run all conditions ─────────────────────────────────────────────────────────
results = {}
for cond in CONDS:
    apply_intervention(cond)
    m = measure(net, assemblies, core_mask)
    m['replay_events'] = int(_replay_count[0])
    results[cond] = m
    print(f'[T7] {cond:<22s} '
          f'Wcc={m["Wcc"]:.4f} WScc={m["WScc"]:.4f} '
          f'Ret={m["retention_mean"]:.4f} Retr={m["retrieval_mean"]:.4f}',
          flush=True)

# Restore weights
with torch.no_grad():
    net.W.data.copy_(W_trained)
    if has_slow and WS_trained is not None:
        net.W_slow.copy_(WS_trained)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'seed': args.seed,
    'replay_events': int(_replay_count[0]),
    'conditions': results,
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': core.tolist(),
    'unique_idx': unique.tolist(),
    'n_exc': int(ne),
    'has_slow': has_slow,
    'gamma': float(getattr(net, 'gamma', 0.0)),
    'W_trained_norm': float(w_norm),
    'WS_trained_norm': float(ws_norm),
    'W_trained_mean': float(w_mean),
    'WS_trained_mean': float(ws_mean),
}
out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T7] SAVED {out_path}', flush=True)
