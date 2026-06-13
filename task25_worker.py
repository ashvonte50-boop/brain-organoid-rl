"""
TASK 2.5 WORKER — CORE NECESSITY TEST
======================================
Runs ONE seed under ONE condition for the core-necessity ablation.

Conditions supported (via --intervention flag):
  FULL                — baseline (matches Task 2 FULL exactly)
  NO_CORE_STIM        — train_one_memory + _replay_one_event: filter core
                        neurons out of the stimulated assembly. Core neurons
                        exist but are never directly driven. Tests whether
                        direct co-activation of core is necessary for the
                        RS asymmetry to form.
  HALF_STIM           — confound control: train with STIM_STRENGTH * 0.5.
                        Matches total injected current of NO_CORE_STIM
                        (20*1.0 == 40*0.5) without removing core.

Same network size, training schedule, and replay schedule across all
conditions. Only the stim driving the relevant neurons changes.

Saves: RS (true core), RS (random permuted core), Retention, replay events,
RS evolution, final W matrix.
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('cname')
parser.add_argument('si',   type=int)
parser.add_argument('seed', type=int)
parser.add_argument('abl_json')
parser.add_argument('--prefix',       default='T25')
parser.add_argument('--intervention', choices=['FULL', 'NO_CORE_STIM', 'HALF_STIM'],
                                       default='FULL')
parser.add_argument('--use_replay',   type=int, default=1, choices=[0, 1])
parser.add_argument('--boost_scale',  type=float, default=1.3)
args = parser.parse_args()

USE_REPLAY   = bool(args.use_replay)
INTERVENTION = args.intervention

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

from _distortion_paper import (
    compute_directional_alignment, compute_real_schema_index,
)
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task25'
os.makedirs(OUT_DIR, exist_ok=True)


# ── Build-network wrap: captures net regardless of replay ────────────────
_net_capture = [None]
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_capture[0] = n
    return n
ccf.build_network = _track_build


# ── Intervention 1: NO_CORE_STIM — filter core out of stimulated assembly
_CORE_INDICES_SET = set(range(SCHEMA_CORE_SIZE))  # {0..19}

_orig_train  = ccf.train_one_memory
_orig_replay = ccf._replay_one_event
_orig_stim_strength = ccf.STIM_STRENGTH

def _filter_core(assembly):
    """Return assembly array with core indices removed."""
    asm = np.asarray(assembly)
    mask = np.array([int(n) not in _CORE_INDICES_SET for n in asm])
    return asm[mask]


def _train_no_core(net, assembly, tags=None, n_presentations=None, prev_assembly=None):
    """Drop-in for train_one_memory that excludes core neurons from stimulation."""
    filtered = _filter_core(assembly)
    kwargs = {}
    if n_presentations is not None:
        kwargs['n_presentations'] = n_presentations
    if prev_assembly is not None:
        # Also filter prev_assembly so chain-STDP doesn't pull core in via that path
        kwargs['prev_assembly'] = _filter_core(prev_assembly)
    return _orig_train(net, filtered, tags=tags, **kwargs)


# ── Replay event counter + optional core-boost (matches task2_worker) ──
_replay_event_count = [0]
_assemblies_ref = [None]
_core_mask_ref  = [None]

def _wrapped_replay_full_asm(net, assembly, tags=None, **kw):
    """FULL-condition replay wrapper (no core filtering)."""
    _replay_event_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
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


def _wrapped_replay_no_core(net, assembly, tags=None, **kw):
    """NO_CORE_STIM-condition replay wrapper: pass a core-filtered assembly to
    the underlying replay function so cue / target_mask exclude core neurons.

    Centroid log still uses the FULL assemblies for measurement consistency
    with FULL condition.
    """
    _replay_event_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    filtered = _filter_core(assembly)
    if len(filtered) < p['cue_size']:
        # Skip events that would have no cue after filtering
        return None
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne].cpu().numpy()
        cb = {i: w[np.ix_([int(x) for x in asm if 0 <= int(x) < ne],
                          [int(x) for x in asm if 0 <= int(x) < ne])].mean(axis=1)
              for i, asm in enumerate(_assemblies_ref[0])
              if any(0 <= int(x) < ne for x in asm)}
    result = _orig_replay(net, filtered, tags=tags, **p, **kw)
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


# ── RS-evolution hook ────────────────────────────────────────────────────
_rs_evolution = []
_orig_baseline = ccf._EXPERIMENT_HOOKS.get('baseline')
_orig_encode   = ccf._EXPERIMENT_HOOKS.get('post_encode')
_orig_replay_h = ccf._EXPERIMENT_HOOKS.get('post_replay')
_orig_final    = ccf._EXPERIMENT_HOOKS.get('final')

def _snapshot_rs(stage, j, net, assemblies, core_mask):
    if net is None:
        return
    rs = compute_real_schema_index(net, assemblies, core_mask)
    with torch.no_grad():
        wm = float(net.W.data[:net.n_exc, :net.n_exc].mean().item())
    _rs_evolution.append({'stage': stage, 'j': int(j), 'rs': float(rs), 'w_mean': wm})

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
# Install intervention
# ──────────────────────────────────────────────────────────────────────────
print(f'[task25_worker] cname={args.cname} si={args.si} seed={args.seed} '
      f'intervention={INTERVENTION} use_replay={USE_REPLAY} '
      f'boost={args.boost_scale}', flush=True)

if INTERVENTION == 'NO_CORE_STIM':
    ccf.train_one_memory = _train_no_core
    print(f'[task25_worker] INSTALLED: train_one_memory filters core ({len(_CORE_INDICES_SET)} indices)', flush=True)
elif INTERVENTION == 'HALF_STIM':
    ccf.STIM_STRENGTH = _orig_stim_strength * 0.5
    print(f'[task25_worker] STIM_STRENGTH: {_orig_stim_strength} -> {ccf.STIM_STRENGTH}', flush=True)
# else: FULL — no changes

abl_dict = json.loads(args.abl_json)

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

# Install replay wrapper if replay is enabled
if USE_REPLAY:
    if INTERVENTION == 'NO_CORE_STIM':
        ccf._replay_one_event = _wrapped_replay_no_core
    else:
        ccf._replay_one_event = _wrapped_replay_full_asm

try:
    r = ccf.run_sequential_experiment(
        True, USE_REPLAY, assemblies, args.seed, ablation=abl_dict,
    )
finally:
    ccf._replay_one_event  = _orig_replay
    ccf.train_one_memory   = _orig_train
    ccf.STIM_STRENGTH      = _orig_stim_strength

net = _net_capture[0] if _net_capture[0] is not None else _last_net[0]
assert net is not None, 'FATAL: net was never captured'

if not USE_REPLAY:
    assert _replay_event_count[0] == 0, (
        f'FATAL: USE_REPLAY=False but {_replay_event_count[0]} replay events fired'
    )

# Core integrity: how often did core neurons fire / how potentiated are they?
with torch.no_grad():
    W_final = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy().copy()

core_idx = np.asarray(core_mask)
W_core_core = W_final[np.ix_(core_idx, core_idx)].mean()
W_unique_to_core_list = []
for asm in assemblies:
    unique = [i for i in asm if i not in core_idx and i < W_final.shape[0]]
    if unique:
        W_unique_to_core_list.append(W_final[np.ix_(unique, core_idx)].mean())
W_unique_to_core = float(np.mean(W_unique_to_core_list)) if W_unique_to_core_list else 0.0

rs_true = compute_real_schema_index(net, assemblies, core_mask)

# RS with permuted core: pick 20 random indices NOT in the true core
rng = np.random.default_rng(args.seed)
non_core = np.array([i for i in range(net.n_exc) if i not in core_idx])
permuted_core = rng.choice(non_core, size=len(core_idx), replace=False)
rs_permuted = compute_real_schema_index(net, assemblies, permuted_core)

fs  = np.nan_to_num(r['final_scores'], nan=0.0)
dall = compute_directional_alignment(list(_CENTROID_LOG), n_mem=4, core_size=CORE_SIZE)

out_dict = {
    'cname':              args.cname,
    'seed':               args.seed,
    'intervention':       INTERVENTION,
    'use_replay':         USE_REPLAY,
    'boost_scale':        args.boost_scale,
    'ablation':           abl_dict,
    'replay_events':      int(_replay_event_count[0]),
    'real_schema':        float(rs_true),
    'real_schema_permuted': float(rs_permuted),
    'W_core_core_mean':   float(W_core_core),
    'W_unique_to_core_mean': float(W_unique_to_core),
    'dai_core':           float(dall.get('mean_core', 0.0)),
    'dai_unique':         float(dall.get('mean_unique', 0.0)),
    'final_scores':       fs.tolist(),
    'retention_A':        float(fs[0]),
    'retention_B':        float(fs[1]),
    'retention_C':        float(fs[2]),
    'retention_D':        float(fs[3]),
    'retention_mean':     float(np.mean(fs)),
    'rs_evolution':       list(_rs_evolution),
    'W_final':            W_final,
    'core_mask':          core_idx.tolist(),
    'permuted_core':      permuted_core.tolist(),
    'assemblies':         [a.tolist() for a in assemblies],
}

out_path = os.path.join(OUT_DIR, f'{args.prefix}_{args.cname}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out_dict, f)

print(f'[task25_worker] SAVED -> {out_path}', flush=True)
print(f'[task25_worker] RS_true={rs_true:.4f}  RS_perm={rs_permuted:.4f}  '
      f'W[c,c]={W_core_core:.4f}  W[u,c]={W_unique_to_core:.4f}  '
      f'Ret_mean={float(np.mean(fs)):.4f}  rep={_replay_event_count[0]}',
      flush=True)
