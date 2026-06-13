"""
TASK 4 WORKER — Mechanism Discovery
====================================
Heavily instrumented worker that captures, during inter-memory rest/replay:

  1. Replay-event statistics: core/unique participation, spikes/event, memory dist
  2. STDP decomposition: potentiation/depression in cc, uc, uu blocks (replay only)
  3. Spike coincidence: 100x100 coincidence accumulator during measurement windows
  4. Weight decomposition: Wcc, Wuc, Wuu, S1 at every checkpoint

Designated neurons = first 100 excitatory indices:
  core   = [0..19]    (shared by all 4 memories)
  unique = [20..99]   (20 per memory: A=20-39, B=40-59, C=60-79, D=80-99)

Conditions: FULL (use_replay=True), NO_REPLAY (use_replay=False).
torch.compile disabled to allow forward instrumentation.
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('cname')
parser.add_argument('seed', type=int)
parser.add_argument('--prefix',      default='T4')
parser.add_argument('--use_replay',  type=int, default=1)
parser.add_argument('--boost_scale', type=float, default=1.3)
args = parser.parse_args()

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
ccf.USE_TORCH_COMPILE = False   # allow forward instrumentation

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

from _distortion_paper import compute_real_schema_index
from ablation_pipeline import _CENTROID_LOG, _last_net, CORE_SIZE

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task4'
os.makedirs(OUT_DIR, exist_ok=True)

USE_REPLAY = bool(args.use_replay)
CORE = SCHEMA_CORE_SIZE          # 20
N_DESIG = 100                    # core(20) + 4*unique(20)
core_sl   = slice(0, CORE)       # 0..19
uniq_sl   = slice(CORE, N_DESIG) # 20..99

# ── Global instrumentation state ─────────────────────────────────────────
_measuring        = [False]   # True during inter-memory rest (both conditions)
_in_replay_event  = [False]   # True during a single replay event (FULL only)
_current_j        = [-1]      # which rest period we are in (0..2)

# Coincidence accumulator over the 100 designated neurons (per condition, global)
_coinc      = np.zeros((N_DESIG, N_DESIG), dtype=np.float64)
_spike_sum  = np.zeros(N_DESIG, dtype=np.float64)
_meas_steps = [0]

# Per-event spike accumulator (reset each replay event)
_event_spk  = np.zeros(N_DESIG, dtype=np.float64)
_event_steps= [0]

# Replay event records
_replay_records = []   # dicts: memory_idx, core_part, uniq_part, mean_spikes, total_spikes

# STDP decomposition — captured separately for TRAINING and REPLAY phases.
# At COH_THR=0.50 replay STDP rarely fires; training STDP builds core-core
# because core neurons participate in all 4 memories. Both matter.
def _new_stdp():
    return {'pot_cc':0.0,'dep_cc':0.0,'pot_uc':0.0,'dep_uc':0.0,
            'pot_uu':0.0,'dep_uu':0.0,'n_steps':0}
_stdp        = _new_stdp()   # replay-phase STDP
_stdp_train  = _new_stdp()   # training-phase STDP
_in_training = [False]       # True during train_one_memory

# Weight-decomposition trajectory (Wcc, Wuc, Wuu, S1 at each checkpoint)
_traj = []

_assemblies_ref = [None]
_core_mask_ref  = [None]

# ── net capture + forward/stdp instrumentation ───────────────────────────
_net_capture = [None]
_orig_build = ccf.build_network

def _install_instrumentation(net):
    orig_forward = net.forward
    orig_stdp    = net.stdp_step

    def _instr_forward(x_ext=None):
        out = orig_forward(x_ext)
        if _measuring[0]:
            with torch.no_grad():
                s = (net.spikes[:N_DESIG] > 0).float().cpu().numpy()
            _coinc.__iadd__(np.outer(s, s))
            _spike_sum.__iadd__(s)
            _meas_steps[0] += 1
            if _in_replay_event[0]:
                _event_spk.__iadd__(s)
                _event_steps[0] += 1
        return out

    def _instr_stdp():
        bucket = _stdp if _in_replay_event[0] else (_stdp_train if _in_training[0] else None)
        if bucket is not None:
            with torch.no_grad():
                W = net.W.data
                cc_b = W[core_sl, core_sl].clone()
                uc_b = W[uniq_sl, core_sl].clone()
                uu_b = W[uniq_sl, uniq_sl].clone()
            orig_stdp()
            with torch.no_grad():
                W = net.W.data
                for key, b, sl_a, sl_b in [
                    ('cc', cc_b, core_sl, core_sl),
                    ('uc', uc_b, uniq_sl, core_sl),
                    ('uu', uu_b, uniq_sl, uniq_sl)]:
                    delta = (W[sl_a, sl_b] - b)
                    bucket['pot_'+key] += float(delta[delta > 0].sum().item())
                    bucket['dep_'+key] += float(delta[delta < 0].sum().item())
                bucket['n_steps'] += 1
        else:
            orig_stdp()

    net.forward   = _instr_forward
    net.stdp_step = _instr_stdp

def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _install_instrumentation(n)
    _net_capture[0] = n
    return n
ccf.build_network = _track_build

# ── wrap rest functions to open/close measurement window ─────────────────
_orig_rest_replay = ccf.inter_memory_rest_with_replay
_orig_rest_norep  = ccf.inter_memory_rest_no_replay

def _wrap_rest_replay(net, learned_assemblies, *a, **kw):
    _current_j[0] = kw.get('rest_id', len(learned_assemblies)-1)
    _measuring[0] = True
    try:
        return _orig_rest_replay(net, learned_assemblies, *a, **kw)
    finally:
        _measuring[0] = False

def _wrap_rest_norep(net, *a, **kw):
    _measuring[0] = True
    try:
        return _orig_rest_norep(net, *a, **kw)
    finally:
        _measuring[0] = False

ccf.inter_memory_rest_with_replay = _wrap_rest_replay
ccf.inter_memory_rest_no_replay   = _wrap_rest_norep

# ── wrap train_one_memory to tag training-phase STDP ─────────────────────
_orig_train = ccf.train_one_memory
def _wrap_train(net, assembly, *a, **kw):
    _in_training[0] = True
    try:
        return _orig_train(net, assembly, *a, **kw)
    finally:
        _in_training[0] = False
ccf.train_one_memory = _wrap_train

# ── replay event wrapper (FULL only) ─────────────────────────────────────
_replay_count = [0]
_orig_replay = ccf._replay_one_event

def _wrapped_replay(net, assembly, tags=None, **kw):
    _replay_count[0] += 1
    _last_net[0] = net
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    # Reset per-event accumulator, open event window
    _event_spk[:] = 0.0
    _event_steps[0] = 0
    _in_replay_event[0] = True
    try:
        result = _orig_replay(net, assembly, tags=tags, **p, **kw)
    finally:
        _in_replay_event[0] = False

    # Record participation for this event
    core_fired = int((_event_spk[core_sl] > 0).sum())
    uniq_fired = int((_event_spk[uniq_sl] > 0).sum())
    _replay_records.append({
        'memory_idx':   int(kw.get('assembly_idx', -1)),
        'core_part':    core_fired / CORE,
        'uniq_part':    uniq_fired / (N_DESIG - CORE),
        'core_spikes':  float(_event_spk[core_sl].sum()),
        'uniq_spikes':  float(_event_spk[uniq_sl].sum()),
        'total_spikes': float(_event_spk.sum()),
        'n_steps':      int(_event_steps[0]),
    })

    # MB core boost
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

# ── checkpoint snapshot (Wcc, Wuc, Wuu, S1) ──────────────────────────────
def _snapshot(label, net):
    if net is None: return
    assemblies = _assemblies_ref[0]; core_mask = _core_mask_ref[0]
    if assemblies is None: return
    with torch.no_grad():
        W = net.W.data[:net.n_exc, :net.n_exc].cpu().numpy()
    core = np.asarray(core_mask)
    Wcc = float(W[np.ix_(core, core)].mean())
    uc_list, uu_list = [], []
    for asm in assemblies:
        uniq = np.array([i for i in asm if i not in core and i < W.shape[0]])
        if len(uniq):
            uc_list.append(W[np.ix_(uniq, core)].mean())
            uu_list.append(W[np.ix_(uniq, uniq)].mean())
    Wuc = float(np.mean(uc_list)) if uc_list else 1e-9
    Wuu = float(np.mean(uu_list)) if uu_list else 1e-9
    _traj.append({'label': label, 'Wcc': Wcc, 'Wuc': Wuc, 'Wuu': Wuu, 'S1': Wcc-Wuc})

_oh_base = ccf._EXPERIMENT_HOOKS.get('baseline')
_oh_enc  = ccf._EXPERIMENT_HOOKS.get('post_encode')
_oh_rep  = ccf._EXPERIMENT_HOOKS.get('post_replay')
_oh_fin  = ccf._EXPERIMENT_HOOKS.get('final')

def _h_base(net, assemblies, n_mem, j=-1, **kw):
    if _oh_base: _oh_base(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot('baseline', net)
def _h_enc(net, assemblies, n_mem, j, **kw):
    if _oh_enc: _oh_enc(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot(f'post_encode_{j}', net)
def _h_rep(net, assemblies, n_mem, j, **kw):
    if _oh_rep: _oh_rep(net=net, assemblies=assemblies, n_mem=n_mem, j=j, **kw)
    _snapshot(f'post_replay_{j}', net)
def _h_fin(net, assemblies, n_mem, **kw):
    if _oh_fin: _oh_fin(net=net, assemblies=assemblies, n_mem=n_mem, **kw)
    _snapshot('final', net)

ccf.register_hook('baseline', _h_base)
ccf.register_hook('post_encode', _h_enc)
ccf.register_hook('post_replay', _h_rep)
ccf.register_hook('final', _h_fin)

# ── main ─────────────────────────────────────────────────────────────────
print(f'[T4] cname={args.cname} seed={args.seed} use_replay={USE_REPLAY}', flush=True)

ccf.torch.manual_seed(args.seed)
ccf.np.random.seed(args.seed)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
_assemblies_ref[0] = assemblies
_core_mask_ref[0]  = core_mask

if USE_REPLAY:
    ccf._replay_one_event = _wrapped_replay

try:
    r = ccf.run_sequential_experiment(True, USE_REPLAY, assemblies, args.seed, ablation={})
finally:
    ccf._replay_one_event = _orig_replay
    ccf.inter_memory_rest_with_replay = _orig_rest_replay
    ccf.inter_memory_rest_no_replay = _orig_rest_norep
    ccf.train_one_memory = _orig_train

net = _net_capture[0] if _net_capture[0] is not None else _last_net[0]
assert net is not None
if not USE_REPLAY:
    assert _replay_count[0] == 0, f'Expected 0 replay, got {_replay_count[0]}'

# Normalize coincidence by steps
coinc_norm = _coinc / max(_meas_steps[0], 1)
spike_rate = _spike_sum / max(_meas_steps[0], 1)

# Coincidence block summaries (off-diagonal means)
def block_offdiag_mean(M, a0, a1, b0, b1, same):
    blk = M[a0:a1, b0:b1]
    if same:
        n = blk.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(blk[mask].mean()) if n > 1 else 0.0
    return float(blk.mean())

cc_coinc = block_offdiag_mean(coinc_norm, 0, CORE, 0, CORE, True)
uu_coinc = block_offdiag_mean(coinc_norm, CORE, N_DESIG, CORE, N_DESIG, True)
cu_coinc = block_offdiag_mean(coinc_norm, 0, CORE, CORE, N_DESIG, False)

fs = np.nan_to_num(r['final_scores'], nan=0.0)

out = {
    'cname': args.cname, 'seed': args.seed, 'use_replay': USE_REPLAY,
    'replay_events': int(_replay_count[0]),
    'replay_records': _replay_records,
    'stdp': dict(_stdp),
    'stdp_train': dict(_stdp_train),
    'coinc_cc': cc_coinc, 'coinc_uu': uu_coinc, 'coinc_cu': cu_coinc,
    'coinc_matrix': coinc_norm.tolist(),
    'spike_rate_core': float(spike_rate[core_sl].mean()),
    'spike_rate_uniq': float(spike_rate[uniq_sl].mean()),
    'meas_steps': int(_meas_steps[0]),
    'trajectory': _traj,
    'final_scores': fs.tolist(),
    'retention_mean': float(np.mean(fs)),
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': np.asarray(core_mask).tolist(),
}

out_path = os.path.join(OUT_DIR, f'{args.prefix}_{args.cname}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

# Summary print
n_ev = len(_replay_records)
mean_core_part = np.mean([r['core_part'] for r in _replay_records]) if _replay_records else 0
mean_uniq_part = np.mean([r['uniq_part'] for r in _replay_records]) if _replay_records else 0
print(f'[T4] SAVED {out_path}', flush=True)
print(f'[T4] events={n_ev} core_part={mean_core_part:.3f} uniq_part={mean_uniq_part:.3f} '
      f'coinc_cc={cc_coinc:.3f} coinc_uu={uu_coinc:.3f} '
      f'pot_cc={_stdp["pot_cc"]:.3f} pot_uc={_stdp["pot_uc"]:.3f} pot_uu={_stdp["pot_uu"]:.3f} '
      f'Ret={float(np.mean(fs)):.4f}', flush=True)
