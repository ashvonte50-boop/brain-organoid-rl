"""
Serial-position-effect worker.
Wraps compare_catastrophic_forgetting.py (copied as ccf_module.py) to:
  - Run encoding sequences with controllable replay-event count
  - Capture per-memory isyn_score, W_slow, W_fast, assembly_w_fast
  - Provide a gamma=0 (fast-only) probe variant

NEVER writes to the parent project. All outputs go under
~/serial_position_experiment/.
"""
import os, sys, gc
import numpy as np
import torch

# Set DEV_MODE before importing — it's read at import time
os.environ['DEV_MODE'] = '1'

# Make sure this directory is on the path and the parent isn't (avoid mixed import)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ccf_module as ccf

# Default replay-events-per-rest under DEV_MODE
DEV_DEFAULT_REPLAY = 15

# ─────────────────────────────────────────────────────────────────────────────
# Hook: capture network state at end of trial
# ─────────────────────────────────────────────────────────────────────────────
_CAPTURED = {}

def _final_hook(net=None, assemblies=None, n_mem=None, **kwargs):
    """Pull W_fast, W_slow per memory + per assembly_w_fast. Keep net for later probes."""
    if net is None or assemblies is None:
        return

    W_fast_t = net.W.data
    has_slow = hasattr(net, 'W_slow') and net.slow_enabled

    w_fast = []
    w_slow = []
    asm_w_fast = []

    for asm in assemblies:
        asm_exc = asm[asm < ccf.N_EXC]
        idx = np.ix_(asm_exc, asm_exc)
        sub_f = W_fast_t.cpu().numpy()[idx]
        mask = sub_f > 0
        w_fast.append(float(sub_f[mask].mean()) if mask.any() else 0.0)

        if has_slow:
            sub_s = net.W_slow.cpu().numpy()[idx]
            mask_s = sub_s > 0
            w_slow.append(float(sub_s[mask_s].mean()) if mask_s.any() else 0.0)
        else:
            w_slow.append(0.0)

        # Assembly-specific W_fast: mean of weights from assembly's "unique" neurons
        # to its core neurons (here: assembly[5:] -> assembly[:5], or all-to-core for
        # the no-core 4-memory layout). Using non-cued -> cued as a proxy.
        cue_size = min(ccf.CUE_SIZE, len(asm))
        cued = asm[:cue_size]
        non_cued = asm[cue_size:]
        cued_exc = cued[cued < ccf.N_EXC]
        nc_exc = non_cued[non_cued < ccf.N_EXC]
        if len(cued_exc) > 0 and len(nc_exc) > 0:
            sub_a = W_fast_t.cpu().numpy()[np.ix_(nc_exc, cued_exc)]
            mask_a = sub_a > 0
            asm_w_fast.append(float(sub_a[mask_a].mean()) if mask_a.any() else 0.0)
        else:
            asm_w_fast.append(0.0)

    _CAPTURED['w_fast'] = w_fast
    _CAPTURED['w_slow'] = w_slow
    _CAPTURED['assembly_w_fast'] = asm_w_fast
    _CAPTURED['net'] = net
    _CAPTURED['assemblies'] = assemblies

# Register hook ONCE (idempotent)
if 'final' not in getattr(ccf, '_HOOKS', {}) or _final_hook not in ccf._HOOKS.get('final', []):
    ccf.register_hook('final', _final_hook)

# ─────────────────────────────────────────────────────────────────────────────
# Trial entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_single_timepoint(seed, consolidation_fraction, n_memories=4,
                          overlap_frac=0.0, use_gamma0_probe=False,
                          base_replay=None):
    """
    Run one trial with replay events scaled by consolidation_fraction.
    consolidation_fraction in [0.0, 1.0] — 0.0 means NO replay events during
    inter-memory rests, 1.0 means full count.

    Returns a dict with:
        retention   : list of isyn_score per memory (gamma=net.gamma)
        retention_g0: list of isyn_score per memory probed with gamma=0 (if asked)
        w_slow      : list per memory
        w_fast      : list per memory
        assembly_w_fast: list per memory
    """
    if base_replay is None:
        base_replay = DEV_DEFAULT_REPLAY

    n_events = int(round(consolidation_fraction * base_replay))
    # Override module-level constant
    ccf._N_REPLAY_EVENTS = n_events

    # Build assemblies. For 4 memories with overlap=0 the default helper works.
    # For n_memories>4 we need a custom assembly list.
    if n_memories <= 4:
        assemblies = ccf.make_overlapping_assemblies(
            n_memories=n_memories,
            assembly_size=ccf.ASSEMBLY_SIZE,
            overlap_frac=overlap_frac,
        )
    else:
        # Custom n-memory chain inside memory module pool
        exc_per_module = ccf.N_EXC // ccf.N_MODULES
        mm_start = ccf.MEMORY_MODULE * exc_per_module
        a_size = ccf.ASSEMBLY_SIZE
        n_overlap = int(round(overlap_frac * a_size))
        step = a_size - n_overlap
        # ensure all memories fit in the module
        needed = mm_start + (n_memories - 1) * step + a_size
        if needed > mm_start + exc_per_module:
            # shrink assembly size
            a_size = max(15, (exc_per_module - n_overlap) // n_memories + n_overlap)
            step = a_size - n_overlap
        assemblies = [
            np.arange(mm_start + m * step, mm_start + m * step + a_size, dtype=int)
            for m in range(n_memories)
        ]

    _CAPTURED.clear()

    # Run trial: Slow+Replay condition (the regime where all mechanisms are on)
    # use_slow=True, use_replay=True; if frac=0 we get Slow+NoReplay in effect
    # (because n_events=0 means no events fire even though use_replay=True).
    result = ccf.run_sequential_experiment(
        use_slow=True,
        use_replay=True,
        assemblies=assemblies,
        trial_seed=seed,
        prioritize='interference_aware',
        verbose=False,
    )

    out = {
        'retention': result['final_scores'].tolist(),
        'w_slow': _CAPTURED.get('w_slow', [0.0] * n_memories),
        'w_fast': _CAPTURED.get('w_fast', [0.0] * n_memories),
        'assembly_w_fast': _CAPTURED.get('assembly_w_fast', [0.0] * n_memories),
        'retention_g0': None,
    }

    if use_gamma0_probe:
        net = _CAPTURED.get('net')
        if net is not None and hasattr(net, 'gamma'):
            orig_gamma = net.gamma
            try:
                net.gamma = 0.0
                ccf.clear_probe_cache()
                g0_scores = []
                for asm in _CAPTURED['assemblies']:
                    g0_scores.append(
                        float(ccf.probe_memory(net, asm)['isyn_score'])
                    )
                out['retention_g0'] = g0_scores
            finally:
                net.gamma = orig_gamma

    gc.collect()
    return out


def quick_sanity():
    """Smoke test: seed=42 FULL run should give retention ≈ 0.286 (baseline)."""
    out = run_single_timepoint(seed=42, consolidation_fraction=1.0, n_memories=4)
    print(f"[SANITY] retention = {out['retention']}")
    print(f"[SANITY] w_fast    = {out['w_fast']}")
    print(f"[SANITY] w_slow    = {out['w_slow']}")
    return out


if __name__ == '__main__':
    quick_sanity()
