"""
TASK 3 WORKER — Schema Formation Dynamics
==========================================
Captures Wcc, Wuc, S1, RS at every natural training checkpoint:
  baseline → post_encode_A → post_replay_A → post_encode_B → ...
  → post_replay_C → post_encode_D → final

Retention trajectory extracted from result retention_matrix (already
computed inside run_sequential_experiment at zero extra cost).

Conditions: FULL (use_replay=True), NO_REPLAY (use_replay=False)
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('cname')
parser.add_argument('si',   type=int)
parser.add_argument('seed', type=int)
parser.add_argument('--prefix',     default='T3')
parser.add_argument('--use_replay', type=int, default=1)
parser.add_argument('--boost_scale',type=float, default=1.3)
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

from _distortion_paper import compute_real_schema_index
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task3'
os.makedirs(OUT_DIR, exist_ok=True)

# ── net capture ──────────────────────────────────────────────────────
_net_capture = [None]
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_capture[0] = n
    return n
ccf.build_network = _track_build

# ── replay counter + boost wrapper ──────────────────────────────────
_replay_count = [0]
_assemblies_ref = [None]
_core_mask_ref  = [None]
_orig_replay = ccf._replay_one_event

def _wrapped_replay(net, assembly, tags=None, **kw):
    _replay_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    result = _orig_replay(net, assembly, tags=tags, **p, **kw)
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne]
        if args.boost_scale != 1.0:
            ci = np.array([int(x) for x in range(CORE_SIZE) if x < ne])
            if len(ci):
                ci_t = torch.as_tensor(ci, device=w.device)
                w[ci_t[:, None], ci_t[None, :]] *= args.boost_scale
                w.clamp_(0.0, net.w_max)
    return result

# ── checkpoint snapshot function ────────────────────────────────────
_trajectory = []   # list of dicts, one per checkpoint

def _snapshot(label, j, net):
    """Compute Wcc, Wuc, S1, RS from current weight state."""
    if net is None:
        return
    assemblies = _assemblies_ref[0]
    core_mask  = _core_mask_ref[0]
    if assemblies is None or core_mask is None:
        return

    with torch.no_grad():
        W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()

    core = np.asarray(core_mask)
    Wcc  = float(W[np.ix_(core, core)].mean())
    uc_list = []
    for asm in assemblies:
        uniq = np.array([i for i in asm if i not in core and i < W.shape[0]])
        if len(uniq):
            uc_list.append(W[np.ix_(uniq, core)].mean())
    Wuc = float(np.mean(uc_list)) if uc_list else 1e-9
    S1  = Wcc - Wuc
    RS  = (Wcc - Wuc) / (Wcc + Wuc + 1e-9)

    _trajectory.append({
        'label': label,
        'j': int(j),
        'Wcc': Wcc, 'Wuc': Wuc, 'S1': S1, 'RS': RS,
    })

# ── hooks ────────────────────────────────────────────────────────────
_orig_baseline = ccf._EXPERIMENT_HOOKS.get('baseline')
_orig_encode   = ccf._EXPERIMENT_HOOKS.get('post_encode')
_orig_replay_h = ccf._EXPERIMENT_HOOKS.get('post_replay')
_orig_final    = ccf._EXPERIMENT_HOOKS.get('final')

def _h_baseline(net, assemblies, n_mem, j=-1, **kw):
    if _orig_baseline: _orig_baseline(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot('baseline', -1, net)

def _h_encode(net, assemblies, n_mem, j, **kw):
    if _orig_encode: _orig_encode(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot(f'post_encode_{j}', j, net)

def _h_replay(net, assemblies, n_mem, j, **kw):
    if _orig_replay_h: _orig_replay_h(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot(f'post_replay_{j}', j, net)

def _h_final(net, assemblies, n_mem, **kw):
    if _orig_final: _orig_final(net=net, assemblies=assemblies, n_mem=n_mem, **kw)
    _snapshot('final', n_mem-1, net)

ccf.register_hook('baseline',    _h_baseline)
ccf.register_hook('post_encode', _h_encode)
ccf.register_hook('post_replay', _h_replay)
ccf.register_hook('final',       _h_final)

# ── main ─────────────────────────────────────────────────────────────
print(f'[T3] cname={args.cname} seed={args.seed} '
      f'use_replay={bool(args.use_replay)} boost={args.boost_scale}', flush=True)

USE_REPLAY = bool(args.use_replay)

ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
_assemblies_ref[0] = assemblies
_core_mask_ref[0]  = core_mask
_trajectory.clear()
_replay_count[0] = 0
_last_net[0] = None
_net_capture[0] = None

if USE_REPLAY:
    ccf._replay_one_event = _wrapped_replay

try:
    r = ccf.run_sequential_experiment(True, USE_REPLAY, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay

net = _net_capture[0] if _net_capture[0] is not None else _last_net[0]
assert net is not None, 'net not captured'

if not USE_REPLAY:
    assert _replay_count[0] == 0, f'Expected 0 replay events, got {_replay_count[0]}'

# Final metrics
fs  = np.nan_to_num(r['final_scores'], nan=0.0)
ret_matrix = r.get('retention_matrix', None)  # shape (n_mem, n_mem)

# Retention trajectory: ret_matrix[i, j] = retention of memory i after training j
# At stage j: mean retention of memories 0..j = mean(diag(ret_matrix[:j+1, j]))
ret_traj = []
if ret_matrix is not None and len(ret_matrix) > 0:
    rm = np.array(ret_matrix)
    for j in range(rm.shape[1]):
        vals = [rm[i, j] for i in range(j+1) if np.isfinite(rm[i, j])]
        ret_traj.append({'j': j, 'retention_mean': float(np.mean(vals)) if vals else 0.0,
                         'retention_per_memory': [float(rm[i, j]) for i in range(j+1)]})

out = {
    'cname':        args.cname,
    'seed':         args.seed,
    'use_replay':   USE_REPLAY,
    'boost_scale':  args.boost_scale,
    'replay_events': int(_replay_count[0]),
    'trajectory':   _trajectory,        # Wcc/Wuc/S1/RS at each hook
    'ret_traj':     ret_traj,           # retention at each stage j
    'final_scores': fs.tolist(),
    'retention_mean': float(np.mean(fs)),
    'retention_matrix': (r['retention_matrix'].tolist()
                         if hasattr(r.get('retention_matrix'), 'tolist')
                         else r.get('retention_matrix')),
    'assemblies':   [a.tolist() for a in assemblies],
    'core_mask':    np.asarray(core_mask).tolist(),
}

out_path = os.path.join(OUT_DIR, f'{args.prefix}_{args.cname}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

# Final weight snapshot
with torch.no_grad():
    W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
core = np.asarray(core_mask)
Wcc_final = float(W[np.ix_(core, core)].mean())
uc = [W[np.ix_(np.array([i for i in a if i not in core and i<W.shape[0]]),core)].mean()
      for a in assemblies if len([i for i in a if i not in core and i<W.shape[0]])>0]
Wuc_final = float(np.mean(uc)) if uc else 1e-9

print(f'[T3] SAVED {out_path}', flush=True)
print(f'[T3] Wcc={Wcc_final:.4f} S1={(Wcc_final-Wuc_final):.4f} '
      f'Ret={float(np.mean(fs)):.4f} rep={_replay_count[0]} '
      f'checkpoints={len(_trajectory)}', flush=True)
