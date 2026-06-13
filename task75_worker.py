"""
TASK 7.5 WORKER — Sufficiency Test of W_slow[cc]
==================================================
Trains ONE network per seed (identical to T7), then applies 6 post-hoc
interventions that isolate individual W_slow block combinations with W=0.

CONDITIONS:
  CONTROL          : no modification
  WSLOW_CC_ONLY    : W=0, keep only W_slow[cc]       *** KEY SUFFICIENCY TEST ***
  WSLOW_CC_PLUS_UC : W=0, keep W_slow[cc] + W_slow[uc]
  WSLOW_CC_PLUS_UU : W=0, keep W_slow[cc] + W_slow[uu]
  WSLOW_UC_ONLY    : W=0, keep only W_slow[uc]        (falsification control)
  WSLOW_UU_ONLY    : W=0, keep only W_slow[uu]        (falsification control)
"""
import os, sys, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T75')
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

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task75'
os.makedirs(OUT_DIR, exist_ok=True)

CONDS = [
    'CONTROL',
    'WSLOW_CC_ONLY',
    'WSLOW_CC_PLUS_UC',
    'WSLOW_CC_PLUS_UU',
    'WSLOW_UC_ONLY',
    'WSLOW_UU_ONLY',
]

# ── Net capture + replay wrapper (identical to T7) ────────────────────────────
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
    core = np.asarray(core_mask)
    with torch.no_grad():
        W  = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
        WS = net.W_slow.cpu().numpy() if (hasattr(net,'W_slow') and net.slow_enabled) else np.zeros_like(W)
    Wcc = float(W[np.ix_(core, core)].mean())
    WScc = float(WS[np.ix_(core, core)].mean())
    uc_w, uu_w, uc_ws, uu_ws = [], [], [], []
    for asm in assemblies:
        uniq = np.array([i for i in asm if i not in set(core.tolist()) and i < W.shape[0]])
        if len(uniq):
            uc_w.append(W[np.ix_(uniq, core)].mean())
            uu_w.append(W[np.ix_(uniq, uniq)].mean())
            uc_ws.append(WS[np.ix_(uniq, core)].mean())
            uu_ws.append(WS[np.ix_(uniq, uniq)].mean())
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
        'Wcc':  Wcc,  'Wuc':  float(np.mean(uc_w))  if uc_w  else 0.0,
        'Wuu':  float(np.mean(uu_w))  if uu_w  else 0.0,
        'WScc': WScc, 'WSuc': float(np.mean(uc_ws)) if uc_ws else 0.0,
        'WSuu': float(np.mean(uu_ws)) if uu_ws else 0.0,
        'S1':   Wcc - (float(np.mean(uc_w)) if uc_w else 0.0),
        'retention_per_memory': ret_scores.tolist(),
        'retention_mean': float(ret_scores.mean()),
        'retrieval_per_memory': retr.tolist(),
        'retrieval_mean': float(retr.mean()),
    }


# ── Train ─────────────────────────────────────────────────────────────────────
print(f'[T75] seed={args.seed} — training FULL (slow+replay)', flush=True)
ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

_replay_count[0] = 0
_CENTROID_LOG.clear(); _last_net[0] = None; _net_ref[0] = None

ccf._replay_one_event = _wrapped_replay
try:
    r = ccf.run_sequential_experiment(True, True, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None
ne = net.n_exc

core = np.asarray(core_mask)
core_set = set(int(x) for x in core)
unique = np.array(sorted(set(int(i) for asm in assemblies for i in asm
                             if int(i) not in core_set and int(i) < ne)), dtype=np.int64)

core_idx = torch.as_tensor(core,   device=net.W.device, dtype=torch.long)
uniq_idx = torch.as_tensor(unique, device=net.W.device, dtype=torch.long)

print(f'[T75] core n={len(core)}  unique n={len(unique)}  n_exc={ne}', flush=True)
print(f'[T75] gamma={net.gamma:.2f}  W_norm={net.W.data.norm():.2f}  '
      f'WS_norm={net.W_slow.norm():.2f}', flush=True)

with torch.no_grad():
    W_trained  = net.W.data.clone()
    WS_trained = net.W_slow.clone()


def apply(cond):
    with torch.no_grad():
        net.W.data.copy_(W_trained)
        net.W_slow.copy_(WS_trained)
        W  = net.W.data
        WS = net.W_slow

        if cond == 'CONTROL':
            return

        # All non-CONTROL conditions: zero fast W entirely
        W[:ne, :ne] = 0.0

        cc = WS[core_idx[:, None], core_idx[None, :]].clone()
        uc_row = WS[uniq_idx[:, None], core_idx[None, :]].clone()
        cu_row = WS[core_idx[:, None], uniq_idx[None, :]].clone()
        uu = WS[uniq_idx[:, None], uniq_idx[None, :]].clone()

        WS.zero_()

        if cond == 'WSLOW_CC_ONLY':
            WS[core_idx[:, None], core_idx[None, :]] = cc

        elif cond == 'WSLOW_CC_PLUS_UC':
            WS[core_idx[:, None], core_idx[None, :]] = cc
            WS[uniq_idx[:, None], core_idx[None, :]] = uc_row
            WS[core_idx[:, None], uniq_idx[None, :]] = cu_row

        elif cond == 'WSLOW_CC_PLUS_UU':
            WS[core_idx[:, None], core_idx[None, :]] = cc
            WS[uniq_idx[:, None], uniq_idx[None, :]] = uu

        elif cond == 'WSLOW_UC_ONLY':
            WS[uniq_idx[:, None], core_idx[None, :]] = uc_row
            WS[core_idx[:, None], uniq_idx[None, :]] = cu_row

        elif cond == 'WSLOW_UU_ONLY':
            WS[uniq_idx[:, None], uniq_idx[None, :]] = uu

        else:
            raise ValueError(cond)


# ── Probe all conditions ───────────────────────────────────────────────────────
results = {}
for cond in CONDS:
    apply(cond)
    m = measure(net, assemblies, core_mask)
    results[cond] = m
    print(f'[T75] {cond:<22s} '
          f'WScc={m["WScc"]:.4f} WSuc={m["WSuc"]:.4f} WSuu={m["WSuu"]:.4f} '
          f'Ret={m["retention_mean"]:.4f} Retr={m["retrieval_mean"]:.4f}',
          flush=True)

with torch.no_grad():
    net.W.data.copy_(W_trained)
    net.W_slow.copy_(WS_trained)

out = {
    'seed': args.seed,
    'conditions': results,
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': core.tolist(),
    'unique_idx': unique.tolist(),
    'n_exc': int(ne),
    'gamma': float(net.gamma),
}
out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'[T75] SAVED {out_path}', flush=True)
