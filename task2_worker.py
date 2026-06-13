"""
TASK 2 WORKER — REPLAY NECESSITY TEST
======================================
Runs ONE seed under ONE condition. Supports use_replay=True/False.

For NO_REPLAY conditions:
  - Wrapper around _replay_one_event still installed but should NEVER fire
  - Net captured via build_network wrapper (since replay wrapper won't fire)
  - Replay event count must be exactly zero — asserted

Tracks:
  - Final REAL_SCHEMA, DAI_core, Retention
  - RS evolution over training (after each memory's post_replay hook)
  - Final weight matrix (N_EXC x N_EXC)
  - Replay event count
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('cname')
parser.add_argument('si',   type=int)
parser.add_argument('seed', type=int)
parser.add_argument('abl_json')
parser.add_argument('--prefix',      default='T2')
parser.add_argument('--use_replay',  type=int, default=1, choices=[0, 1])
parser.add_argument('--boost_scale', type=float, default=1.3)
parser.add_argument('--coh_thr',     type=float, default=None)
args = parser.parse_args()

USE_REPLAY = bool(args.use_replay)

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
if args.coh_thr is not None:
    ccf.REPLAY_COHERENCE_THR = args.coh_thr

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

from _distortion_paper import (
    compute_directional_alignment, compute_real_schema_index,
)
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task2'
os.makedirs(OUT_DIR, exist_ok=True)


# ── Build-network wrap: captures net regardless of replay ────────────────
_net_capture = [None]
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_capture[0] = n
    return n
ccf.build_network = _track_build

# ── Replay-event counter + optional core-boost ──────────────────────────
_replay_event_count = [0]
_orig_replay = ccf._replay_one_event
def _wrapped_replay(net, assembly, tags=None, **kw):
    _replay_event_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    # Snapshot centroids before
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne].cpu().numpy()
        cb = {i: w[np.ix_([int(x) for x in asm if 0 <= int(x) < ne],
                          [int(x) for x in asm if 0 <= int(x) < ne])].mean(axis=1)
              for i, asm in enumerate(_assemblies_ref[0])
              if any(0 <= int(x) < ne for x in asm)}
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
        w_np = w.cpu().numpy()
        ca = {i: w_np[np.ix_([int(x) for x in asm if 0 <= int(x) < ne],
                             [int(x) for x in asm if 0 <= int(x) < ne])].mean(axis=1)
              for i, asm in enumerate(_assemblies_ref[0])
              if any(0 <= int(x) < ne for x in asm)}
    _CENTROID_LOG.append({
        'replay_id':       kw.get('burst_id', 0) * 1000 + kw.get('event_id', 0),
        'memory_idx':      kw.get('assembly_idx', -1),
        'centroid_before': {k: v.tolist() for k, v in cb.items()},
        'centroid_after':  {k: v.tolist() for k, v in ca.items()},
    })
    return result

# ── RS-evolution hook: wrap existing post_replay/post_encode hooks ──────
_rs_evolution = []   # list of (stage, j, RS, weight_mean)

_assemblies_ref = [None]
_core_mask_ref  = [None]

def _snapshot_rs(stage, j, net, assemblies, core_mask):
    if net is None:
        return
    rs = compute_real_schema_index(net, assemblies, core_mask)
    with torch.no_grad():
        wm = float(net.W.data[:net.n_exc, :net.n_exc].mean().item())
    _rs_evolution.append({'stage': stage, 'j': int(j), 'rs': float(rs), 'w_mean': wm})

# Wrap existing schema_core hooks
_orig_baseline = ccf._EXPERIMENT_HOOKS.get('baseline')
_orig_encode   = ccf._EXPERIMENT_HOOKS.get('post_encode')
_orig_replay_h = ccf._EXPERIMENT_HOOKS.get('post_replay')
_orig_final    = ccf._EXPERIMENT_HOOKS.get('final')

def _h_baseline(net, assemblies, n_mem, j=-1, **kw):
    if _orig_baseline: _orig_baseline(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot_rs('baseline', j, net, _assemblies_ref[0], _core_mask_ref[0])

def _h_encode(net, assemblies, n_mem, j, **kw):
    if _orig_encode: _orig_encode(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot_rs('post_encode', j, net, _assemblies_ref[0], _core_mask_ref[0])

def _h_replay(net, assemblies, n_mem, j, **kw):
    if _orig_replay_h: _orig_replay_h(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot_rs('post_replay', j, net, _assemblies_ref[0], _core_mask_ref[0])

def _h_final(net, assemblies, n_mem, **kw):
    if _orig_final: _orig_final(net=net, assemblies=assemblies, n_mem=n_mem, **kw)
    _snapshot_rs('final', n_mem - 1, net, _assemblies_ref[0], _core_mask_ref[0])

ccf.register_hook('baseline',    _h_baseline)
ccf.register_hook('post_encode', _h_encode)
ccf.register_hook('post_replay', _h_replay)
ccf.register_hook('final',       _h_final)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
abl_dict = json.loads(args.abl_json)
print(f'[task2_worker] cname={args.cname} si={args.si} seed={args.seed} '
      f'use_replay={USE_REPLAY} boost={args.boost_scale} abl={args.abl_json}',
      flush=True)

ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
_assemblies_ref[0] = assemblies
_core_mask_ref[0]  = core_mask

_replay_event_count[0] = 0
_CENTROID_LOG.clear()
_rs_evolution.clear()
_last_net[0]     = None
_net_capture[0]  = None

# Install replay wrapper only when we want replay
if USE_REPLAY:
    ccf._replay_one_event = _wrapped_replay

try:
    r = ccf.run_sequential_experiment(
        True, USE_REPLAY, assemblies, args.seed, ablation=abl_dict,
    )
finally:
    ccf._replay_one_event = _orig_replay

# Net: prefer build-network capture (works for both replay and no-replay)
net = _net_capture[0] if _net_capture[0] is not None else _last_net[0]
assert net is not None, 'FATAL: net was never captured'

# Verify replay count
if not USE_REPLAY:
    assert _replay_event_count[0] == 0, (
        f'FATAL: USE_REPLAY=False but {_replay_event_count[0]} replay events fired'
    )
print(f'[task2_worker] replay_events={_replay_event_count[0]}', flush=True)

rs  = compute_real_schema_index(net, assemblies, core_mask)
fs  = np.nan_to_num(r['final_scores'], nan=0.0)
dall = compute_directional_alignment(list(_CENTROID_LOG), n_mem=4, core_size=CORE_SIZE)

# Final W matrix (full N_EXC x N_EXC)
with torch.no_grad():
    W_final = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy().copy()

out_dict = {
    'cname':           args.cname,
    'seed':            args.seed,
    'use_replay':      USE_REPLAY,
    'boost_scale':     args.boost_scale,
    'ablation':        abl_dict,
    'replay_events':   int(_replay_event_count[0]),
    'real_schema':     float(rs),
    'dai_core':        float(dall.get('mean_core', 0.0)),
    'dai_unique':      float(dall.get('mean_unique', 0.0)),
    'final_scores':    fs.tolist(),
    'retention_A':     float(fs[0]),
    'retention_B':     float(fs[1]),
    'retention_C':     float(fs[2]),
    'retention_D':     float(fs[3]),
    'retention_mean':  float(np.mean(fs)),
    'rs_evolution':    list(_rs_evolution),
    'W_final':         W_final,
    'core_mask':       np.asarray(core_mask).tolist(),
    'assemblies':      [a.tolist() for a in assemblies],
}

out_path = os.path.join(OUT_DIR, f'{args.prefix}_{args.cname}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out_dict, f)

print(f'[task2_worker] SAVED -> {out_path}', flush=True)
print(f'[task2_worker] RS={rs:.4f}  DAI={out_dict["dai_core"]:.4f}  '
      f'Ret_mean={out_dict["retention_mean"]:.4f}  replay_events={_replay_event_count[0]}',
      flush=True)
