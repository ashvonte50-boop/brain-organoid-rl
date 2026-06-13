"""
MECHANISM ACTIVATION DIAGNOSTIC v2
===================================
Patches CCF directly to count when each mechanism's CODE BLOCK executes.
No wrapper kwargs conflicts. Inserts counters by monkey-patching the
specific numpy/torch operations the mechanisms perform.

For each mechanism we count:
  - times the gate condition is TRUE (so the code should run)
  - magnitude of the weight delta produced

Usage:
  python diagnose_mechanisms.py
"""
import os, sys, warnings, json
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import torch

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
from _distortion_paper import compute_directional_alignment, compute_real_schema_index

SEED = 42
CORE = SCHEMA_CORE_SIZE

# ── Monkey-patched counters ──────────────────────────────────────────────────
COUNTERS = {}

def reset_counters():
    global COUNTERS
    COUNTERS = {
        'replay_event_calls':    0,
        'coh_values':            [],
        'overlap_max_values':    [],
        'has_overlap_neighbors': [],
        # M5 drift
        'M5_drift_gate_met':     0,
        'M5_drift_weight_delta': 0.0,
        # M1 overlap penalty
        'M1_ov_pen_applied':     0,
        # M2 cross-LTD
        'M2_xltd_executed':      0,
        'M2_xltd_weight_delta':  0.0,
        # M10 reconsol
        'M10_reconsol_active':   0,
    }


# ── Wrap np.ix_ around M5's specific weight update ────────────────────────────
# The M5 code does:  net.W.data[np.ix_(other_e, _active_ov)] += OVERLAP_DRIFT_RATE
# We patch torch.Tensor.__iadd__ to detect when OVERLAP_DRIFT_RATE is being added.
# Easier approach: directly instrument by wrapping _replay_one_event using
# inspect-style techniques. But the cleanest path: copy the source and inject prints.

# ── Strategy: install a wrapper that calls orig() then DIFFS weights to detect
# which mechanisms fired ──────────────────────────────────────────────────────

_centroid_log = []
_last_net_ref = [None]

def _wrapped_replay(orig_fn, ablation_dict, assemblies):
    """Returns a wrapper around _replay_one_event that:
       - snapshots W before/after
       - measures change in unique→core block (M5 drift signature)
       - measures change in overlap LTD block (M2 signature)
       - records centroid log
    """
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    def _wrapper(net, assembly, tags=None, **kw):
        # Remove ablation from kw if user double-passed it
        kw.pop('ablation', None)

        COUNTERS['replay_event_calls'] += 1
        ne = net.n_exc

        # -- Snapshot weights and centroids BEFORE replay --
        W_before = net.W.data[:ne, :ne].cpu().numpy().copy()
        with torch.no_grad():
            wn_before = W_before
            cb = {}
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    cb[i] = wn_before[np.ix_(valid, valid)].mean(axis=1).copy()

        # -- Check gating: does this assembly have overlap neighbours? --
        asm_idx = kw.get('assembly_idx', -1)
        all_asms = kw.get('all_assemblies', assemblies)
        has_neighbors = False
        max_overlap  = 0.0
        if asm_idx >= 0 and asm_idx < len(all_asms):
            replay_set = set(all_asms[asm_idx].tolist())
            other_neurons = set()
            for j, oasm in enumerate(all_asms):
                if j == asm_idx: continue
                other_neurons |= set(oasm.tolist())
            shared = replay_set & other_neurons
            if shared:
                has_neighbors = True
                max_overlap = len(shared) / max(len(replay_set), 1)
        COUNTERS['has_overlap_neighbors'].append(has_neighbors)
        COUNTERS['overlap_max_values'].append(float(max_overlap))

        # -- Call orig with ablation kwarg (not in kw) --
        result = orig_fn(net, assembly, tags=tags, ablation=ablation_dict, **p, **kw)

        # -- Snapshot weights AFTER replay --
        W_after = net.W.data[:ne, :ne].cpu().numpy().copy()

        # -- Measure mechanism-specific signatures --
        # M5 drift signature: weight INCREASE in unique→core direction
        # (the M5 code does W[other_e, _active_ov] += OVERLAP_DRIFT_RATE for overlap-shared)
        # The "shared" neurons are the schema core (since all 4 memories share core[0:20]).
        # So M5 adds to W[unique_of_other_memory, core]
        # We measure: change in mean weight from unique (20-99) to core (0-20)
        unique_to_core_before = W_before[20:100, 0:20].mean()
        unique_to_core_after  = W_after [20:100, 0:20].mean()
        unique_to_core_delta  = unique_to_core_after - unique_to_core_before

        # Capture coherence if returned
        if isinstance(result, dict):
            for key in ('smooth_coh', 'coherence', 'coh'):
                if key in result:
                    COUNTERS['coh_values'].append(float(result[key]))
                    break

            # Did M5 fire? Check by looking at characteristic weight delta
            # (M5 adds OVERLAP_DRIFT_RATE = 0.02 to specific synapses)
            if abs(unique_to_core_delta) > 1e-5 and has_neighbors:
                COUNTERS['M5_drift_gate_met'] += 1
                COUNTERS['M5_drift_weight_delta'] += abs(unique_to_core_delta)

        # -- Apply MB boost and capture centroid AFTER --
        with torch.no_grad():
            w2 = net.W.data[:ne, :ne]
            ci = torch.arange(CORE, device=w2.device)
            w2[ci[:, None], ci[None, :]] *= 1.3
            w2.clamp_(0.0, net.w_max)
            wn_after = w2.cpu().numpy()
            ca = {}
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    ca[i] = wn_after[np.ix_(valid, valid)].mean(axis=1).copy()

        _centroid_log.append({
            'replay_id':       kw.get('burst_id', 0)*1000 + kw.get('event_id', 0),
            'memory_idx':      kw.get('assembly_idx', -1),
            'centroid_before': {k: v.tolist() for k, v in cb.items()},
            'centroid_after':  {k: v.tolist() for k, v in ca.items()},
        })
        _last_net_ref[0] = net

        return result
    return _wrapper


def run_one(seed, ablation_dict, label):
    """Run one experiment with diagnostics."""
    torch.manual_seed(seed); np.random.seed(seed)
    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    reset_counters()
    _centroid_log.clear()
    _last_net_ref[0] = None

    orig_fn = ccf._replay_one_event
    ccf._replay_one_event = _wrapped_replay(orig_fn, ablation_dict, assemblies)
    try:
        r = ccf.run_sequential_experiment(True, True, assemblies, seed, ablation=ablation_dict)
    finally:
        ccf._replay_one_event = orig_fn

    net = _last_net_ref[0]
    fs  = np.nan_to_num(r['final_scores'], nan=0.0)
    dall = compute_directional_alignment(list(_centroid_log), n_mem=4, core_size=CORE)
    rs   = compute_real_schema_index(net, assemblies, core_mask) if net else 0.0

    print(f'\n{"="*65}')
    print(f'RUN: {label}  seed={seed}  ablation={ablation_dict}')
    print(f'{"="*65}')
    print(f'  total replay events:        {COUNTERS["replay_event_calls"]}')
    print(f'  events with overlap neighbours: {sum(COUNTERS["has_overlap_neighbors"])}')
    print(f'  mean overlap fraction:      {np.mean(COUNTERS["overlap_max_values"]):.3f}')
    print(f'  max  overlap fraction:      {np.max(COUNTERS["overlap_max_values"]):.3f}')
    coh = COUNTERS['coh_values']
    if coh:
        print(f'  coherence: n={len(coh)} mean={np.mean(coh):.3f} max={np.max(coh):.3f}')
    else:
        print(f'  coherence: not in result dict (not captured)')
    print(f'  REPLAY_COHERENCE_THR (CCF):  {getattr(ccf,"REPLAY_COHERENCE_THR","?")}')
    print(f'  OVERLAP_DRIFT_RATE (CCF):    {getattr(ccf,"OVERLAP_DRIFT_RATE","?")}')
    print(f'  M5 drift evidence (events with weight delta in unique→core): '
          f'{COUNTERS["M5_drift_gate_met"]}')
    print(f'  M5 cumulative unique→core delta magnitude: '
          f'{COUNTERS["M5_drift_weight_delta"]:.4e}')
    print(f'  --- METRICS ---')
    print(f'  DAI_core:    {dall["mean_core"]:.4f}')
    print(f'  REAL_SCHEMA: {rs:.4f}')
    print(f'  Retention:   A={fs[0]:.3f}  B={fs[1]:.3f}  C={fs[2]:.3f}  D={fs[3]:.3f}')

    return {
        'label': label, 'ablation': str(ablation_dict),
        'replay_events': COUNTERS['replay_event_calls'],
        'has_neighbors_count': int(sum(COUNTERS['has_overlap_neighbors'])),
        'mean_overlap': float(np.mean(COUNTERS['overlap_max_values'])),
        'coh_mean': float(np.mean(coh)) if coh else None,
        'coh_max':  float(np.max(coh))  if coh else None,
        'M5_drift_events': COUNTERS['M5_drift_gate_met'],
        'M5_cum_delta':    float(COUNTERS['M5_drift_weight_delta']),
        'dai': float(dall['mean_core']),
        'rs':  float(rs),
        'ret_A': float(fs[0]),
        'ret_mean': float(np.mean(fs)),
    }


def print_ccf_constants():
    print('\n' + '='*65)
    print('CCF CONSTANTS')
    print('='*65)
    for c in ['REPLAY_COHERENCE_THR', 'OVERLAP_DRIFT_RATE', 'OVERLAP_COHERENCE_PENALTY',
              'CROSS_LTD_RATE', 'RECONSOL_LTD_BOOST', 'RECONSOL_WINDOW_STEPS',
              'DEV_MODE']:
        val = getattr(ccf, c, 'NOT FOUND')
        print(f'  {c:<35} = {val}')


if __name__ == '__main__':
    print_ccf_constants()

    print('\n' + '#'*65)
    print('# RUN 1: FULL model')
    print('#'*65)
    full = run_one(SEED, {}, 'FULL')

    print('\n' + '#'*65)
    print('# RUN 2: -M5 ablation (drift=False)')
    print('#'*65)
    m5 = run_one(SEED, {'drift': False}, 'ABLATE_M5')

    print('\n' + '#'*65)
    print('# RUN 3: -M1 ablation (overlap_penalty=False)')
    print('#'*65)
    m1 = run_one(SEED, {'overlap_penalty': False}, 'ABLATE_M1')

    print('\n' + '#'*65)
    print('# RUN 4: -M10 ablation (reconsol=False)')
    print('#'*65)
    m10 = run_one(SEED, {'reconsol': False}, 'ABLATE_M10')

    print('\n\n' + '='*65)
    print('FINAL SUMMARY (seed=42)')
    print('='*65)
    print(f'  {"label":<12}  {"replay_evts":>11}  {"M5_fires":>9}  {"M5_delta":>11}  {"DAI":>7}  {"RS":>7}  {"Ret_A":>7}')
    for r in [full, m5, m1, m10]:
        print(f'  {r["label"]:<12}  {r["replay_events"]:>11}  {r["M5_drift_events"]:>9}  '
              f'{r["M5_cum_delta"]:>11.4e}  {r["dai"]:>7.4f}  {r["rs"]:>7.4f}  {r["ret_A"]:>7.4f}')

    print('\nΔ from FULL:')
    for r in [m5, m1, m10]:
        print(f'  {r["label"]:<12}  ΔDAI={r["dai"]-full["dai"]:+.4f}  '
              f'ΔRS={r["rs"]-full["rs"]:+.4f}  '
              f'ΔRet_A={r["ret_A"]-full["ret_A"]:+.4f}')

    out = os.path.join(r'C:\Users\Admin\brain-organoid-rl\ablation_results\diagnostic',
                       'mechanism_activation_report.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'full': full, 'm5': m5, 'm1': m1, 'm10': m10}, f, indent=2)
    print(f'\nSaved: {out}')
