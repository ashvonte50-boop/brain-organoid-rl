"""
TASK 5.5 WORKER — Formation-time Causal Test of Wcc
=====================================================
Trains a SEPARATE network per condition with intervention active DURING
all training and replay (formation-time causality):

  FULL             : standard training, no intervention
  WCC_FROZEN       : core-core block restored to init values after every STDP step
  WCC_CLAMPED_ZERO : core-core block zeroed after every STDP step
  WCC_NO_STDP      : plastic_mask zeros out core-core pairs (no STDP to that block)

This directly tests whether Wcc must GROW during learning to produce
replay-protected retention — the question Task 5 (post-hoc) could not answer.
"""
import os, sys, json, pickle, argparse
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('--prefix', default='T55')
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

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task55'
os.makedirs(OUT_DIR, exist_ok=True)

CORE = SCHEMA_CORE_SIZE  # 20
CONDS = ['FULL', 'WCC_FROZEN', 'WCC_CLAMPED_ZERO', 'WCC_NO_STDP']


# ── Helper: build a guard that enforces a formation-time intervention ─────────

def _make_stdp_guard(net, mode, core_idx, W_init_cc):
    """
    Return a wrapper around net.stdp_step that enforces formation-time Wcc
    constraint.  Called ONCE per condition before training starts.
    """
    _orig_stdp = net.stdp_step

    if mode == 'FROZEN':
        def _guarded():
            _orig_stdp()
            with torch.no_grad():
                net.W.data[core_idx[:, None], core_idx[None, :]] = W_init_cc
        return _guarded

    elif mode == 'CLAMPED_ZERO':
        def _guarded():
            _orig_stdp()
            with torch.no_grad():
                net.W.data[core_idx[:, None], core_idx[None, :]] = 0.0
        return _guarded

    elif mode == 'NO_STDP':
        # Zero out plastic_mask for core-core pairs before training begins;
        # no per-step overhead needed.
        with torch.no_grad():
            net.plastic_mask[core_idx[:, None], core_idx[None, :]] = 0.0
        return _orig_stdp  # unmodified step is fine — mask prevents updates

    else:  # FULL — no-op
        return _orig_stdp


# ── Net capture wrappers (shared logic identical to task5_worker) ─────────────

_net_ref   = [None]
_orig_build = ccf.build_network

def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    return n

ccf.build_network = _track_build


def _make_replay_wrapper(net, mode, core_idx, W_init_cc):
    """
    Wrap _replay_one_event so that the core-boost AND the formation constraint
    are both applied consistently during replay.
    """
    _orig_replay = ccf._replay_one_event
    _replay_count = [0]

    def _wrapped(net_inner, assembly, tags=None, **kw):
        _replay_count[0] += 1
        _last_net[0] = net_inner
        p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
        result = _orig_replay(net_inner, assembly, tags=tags, **p, **kw)
        # Core boost (same as task5 and ablation_pipeline)
        with torch.no_grad():
            ne = net_inner.n_exc
            w = net_inner.W.data[:ne, :ne]
            ci = core_idx[core_idx < ne]
            if len(ci):
                w[ci[:, None], ci[None, :]] *= 1.3
                w.clamp_(0.0, net_inner.w_max)
            # Enforce formation constraint post-boost
            if mode == 'FROZEN':
                net_inner.W.data[core_idx[:, None], core_idx[None, :]] = W_init_cc
            elif mode == 'CLAMPED_ZERO':
                net_inner.W.data[core_idx[:, None], core_idx[None, :]] = 0.0
        return result

    return _wrapped, _replay_count


# ── Measure function (identical to task5_worker) ──────────────────────────────

def measure(net, assemblies, core_mask):
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


# ── Train each condition independently ───────────────────────────────────────

print(f'[T55] seed={args.seed} — formation-time causal test (4 conditions)', flush=True)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
core = np.asarray(core_mask)

results = {}

_COND_MODES = {
    'FULL':            'FULL',
    'WCC_FROZEN':      'FROZEN',
    'WCC_CLAMPED_ZERO':'CLAMPED_ZERO',
    'WCC_NO_STDP':     'NO_STDP',
}

for cond in CONDS:
    mode = _COND_MODES[cond]
    print(f'\n[T55] === {cond} (mode={mode}) ===', flush=True)

    # Fresh seed for each condition (same seed -> same network init)
    ccf.torch.manual_seed(args.seed)
    ccf.np.random.seed(args.seed)
    _net_ref[0] = None
    _CENTROID_LOG.clear()
    _last_net[0] = None

    # Force build_network to be called by run_sequential_experiment
    # by re-installing our tracking wrapper
    ccf.build_network = _track_build

    # First, do a dummy build to capture the initial network weights before
    # training, so we know what W_init_cc looks like
    _dummy = ccf.build_network(use_slow=True)
    core_idx = torch.as_tensor(core, device=_dummy.W.device, dtype=torch.long)
    with torch.no_grad():
        W_init_cc = _dummy.W.data[core_idx[:, None], core_idx[None, :]].clone()
    del _dummy

    # Re-seed so run_sequential_experiment builds the same network
    ccf.torch.manual_seed(args.seed)
    ccf.np.random.seed(args.seed)
    _net_ref[0] = None

    # Install replay wrapper with formation constraint
    _orig_replay = ccf._replay_one_event
    _wrapped_replay_fn, _replay_count_box = _make_replay_wrapper(
        None, mode, core_idx, W_init_cc)
    ccf._replay_one_event = _wrapped_replay_fn

    # We need to intercept net.stdp_step AFTER build_network is called.
    # Strategy: override build_network to install the guard right after build.
    _orig_build_inner = _orig_build

    def _build_with_guard(use_slow=False, _mode=mode, _cidx=core_idx, _wicc=W_init_cc):
        n = _orig_build_inner(use_slow=use_slow)
        _net_ref[0] = n
        # Install formation-time guard on this network
        n.stdp_step = _make_stdp_guard(n, _mode, _cidx, _wicc)
        return n

    ccf.build_network = _build_with_guard

    try:
        r = ccf.run_sequential_experiment(True, True, assemblies, args.seed, ablation={})
    finally:
        ccf._replay_one_event = _orig_replay
        ccf.build_network = _track_build  # restore for next iteration

    net = _net_ref[0]
    assert net is not None, f'Network was not captured for {cond}'

    m = measure(net, assemblies, core_mask)
    m['replay_events'] = int(_replay_count_box[0])
    results[cond] = m

    print(f'[T55] {cond:<16s} Wcc={m["Wcc"]:.4f} Wuc={m["Wuc"]:.4f} Wuu={m["Wuu"]:.4f} '
          f'S1={m["S1"]:.4f} Ret={m["retention_mean"]:.4f} Retr={m["retrieval_mean"]:.4f}',
          flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'seed': args.seed,
    'conditions': results,
    'assemblies': [a.tolist() for a in assemblies],
    'core_mask': core.tolist(),
}
out_path = os.path.join(OUT_DIR, f'{args.prefix}_seed{args.seed}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'\n[T55] SAVED {out_path}', flush=True)
