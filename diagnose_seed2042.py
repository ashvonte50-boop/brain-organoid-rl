"""
SEED=2042 IMMUNITY DIAGNOSTIC
==============================
Single-question experiment: why does seed=2042 show high RS/Ret at COH_THR=0.00
while seeds 42 and 1042 collapse?

Instruments:
  1. Number of replay events
  2. Coherence value per event (mean + peak)
  3. Ignition pass/fail count
  4. Steps where cv > COH_THR per event (proxy for mechanism fire count)
  5. Weight-change signatures: M2 (LTD) and M5 (drift) in overlap block
  6. STDP weight increases in target block (proxy for STDP fires)
  7. Final DAI_core, REAL_SCHEMA, Retention

Runs seed=42 (known to collapse) vs seed=2042 (immune) at COH_THR=0.00.
"""
import os, sys, warnings, time
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import torch

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

from ablation_pipeline import _make_wrapper, _CENTROID_LOG, _last_net, CORE_SIZE
from _distortion_paper import compute_directional_alignment, compute_real_schema_index

COH_THR_OVERRIDE = 0.00
CORE = SCHEMA_CORE_SIZE   # 20


class DiagnosticCollector:
    def __init__(self):
        self.events = []

    def record(self, ev_dict, W_before, W_after, ne):
        mean_coh   = ev_dict.get('mean_coherence', 0.0)
        peak_coh   = ev_dict.get('peak_coherence', 0.0)
        n_coh_steps = ev_dict.get('n_steps_coherent', 0)
        ignition   = ev_dict.get('ignition_pass', False)

        # Weight-change blocks
        # M5 drift signature: unique→core weight INCREASE
        # unique neurons: indices CORE..CORE+80 (approx); core: 0..CORE
        uc_before = W_before[CORE:CORE+80, 0:CORE].mean() if ne > CORE+80 else 0.0
        uc_after  = W_after [CORE:CORE+80, 0:CORE].mean() if ne > CORE+80 else 0.0
        m5_delta  = float(uc_after - uc_before)

        # M2 LTD signature: off-diagonal block weight DECREASE (unique of A → core)
        # Proxy: sum of all weight decreases in off-diagonal (cross-assembly) block
        diff = W_after[:ne, :ne] - W_before[:ne, :ne]
        m2_ltd_sum = float(diff[diff < 0].sum())   # total LTD applied

        # STDP signature: weight INCREASE in diagonal (within-assembly) blocks
        diag_block = W_after[:CORE, :CORE] - W_before[:CORE, :CORE]
        stdp_ltp_sum = float(diag_block[diag_block > 0].sum())

        self.events.append({
            'mean_coh':    mean_coh,
            'peak_coh':    peak_coh,
            'n_coh_steps': n_coh_steps,
            'ignition':    ignition,
            'm5_delta':    m5_delta,
            'm2_ltd':      m2_ltd_sum,
            'stdp_ltp':    stdp_ltp_sum,
        })

    def summary(self):
        if not self.events:
            return {'n_events': 0}
        n   = len(self.events)
        mc  = np.array([e['mean_coh']    for e in self.events])
        pc  = np.array([e['peak_coh']    for e in self.events])
        ns  = np.array([e['n_coh_steps'] for e in self.events])
        ig  = np.array([e['ignition']    for e in self.events])
        m5  = np.array([e['m5_delta']    for e in self.events])
        m2  = np.array([e['m2_ltd']      for e in self.events])
        ltp = np.array([e['stdp_ltp']    for e in self.events])
        return {
            'n_events':           n,
            'n_ignition_pass':    int(ig.sum()),
            'n_ignition_fail':    int((~ig).sum()),
            'mean_coh_mean':      float(mc.mean()),
            'mean_coh_max':       float(mc.max()),
            'peak_coh_mean':      float(pc.mean()),
            'peak_coh_max':       float(pc.max()),
            'n_coh_steps_mean':   float(ns.mean()),
            'n_coh_steps_total':  int(ns.sum()),
            'm5_delta_mean':      float(m5.mean()),
            'm5_delta_total':     float(m5.sum()),
            'm2_ltd_mean':        float(m2.mean()),
            'm2_ltd_total':       float(m2.sum()),
            'stdp_ltp_mean':      float(ltp.mean()),
            'stdp_ltp_total':     float(ltp.sum()),
        }


def make_instrumented_wrapper(assemblies, collector, boost_scale=1.3):
    orig_fn = ccf._replay_one_event
    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)

    def _wrapper(net, assembly, tags=None, **kw):
        _last_net[0] = net
        ne = net.n_exc

        W_before = net.W.data[:ne, :ne].cpu().numpy().copy()

        result = orig_fn(net, assembly, tags=tags, **p, **kw)

        W_after = net.W.data[:ne, :ne].cpu().numpy().copy()

        if isinstance(result, dict):
            collector.record(result, W_before, W_after, ne)

        with torch.no_grad():
            w = net.W.data[:ne, :ne]
            if boost_scale != 1.0:
                ci = np.array([int(x) for x in range(CORE_SIZE) if x < ne])
                if len(ci):
                    ci_t = torch.as_tensor(ci, device=w.device)
                    w[ci_t[:, None], ci_t[None, :]] *= boost_scale
                    w.clamp_(0.0, net.w_max)
            ca = {}
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    ca[i] = net.W.data[:ne, :ne][np.ix_(valid, valid)].mean(axis=1).cpu().numpy()
            cb = {}
            for i, asm in enumerate(assemblies):
                valid = [int(x) for x in asm if 0 <= int(x) < ne]
                if valid:
                    cb[i] = torch.tensor(W_before)[np.ix_(valid, valid)].mean(axis=1).numpy()
            _CENTROID_LOG.append({
                'replay_id':       kw.get('burst_id', 0) * 1000 + kw.get('event_id', 0),
                'memory_idx':      kw.get('assembly_idx', -1),
                'centroid_before': {k: v.tolist() for k, v in cb.items()},
                'centroid_after':  {k: v.tolist() for k, v in ca.items()},
            })
        return result
    return orig_fn, _wrapper


def run_diagnostic(seed, label, coh_thr=COH_THR_OVERRIDE):
    print(f'\n{"="*65}', flush=True)
    print(f'DIAGNOSTIC: {label}  seed={seed}  COH_THR={coh_thr}', flush=True)
    print(f'{"="*65}', flush=True)

    ccf.REPLAY_COHERENCE_THR = coh_thr
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)

    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    # Print assembly overlap stats before running
    for i in range(4):
        for j in range(i+1, 4):
            ai = set(assemblies[i].tolist())
            aj = set(assemblies[j].tolist())
            sh = ai & aj
            print(f'  overlap({i},{j}): {len(sh)} / {len(ai)} = {len(sh)/len(ai):.3f}', flush=True)

    collector = DiagnosticCollector()
    _CENTROID_LOG.clear()

    orig_fn, wrapper = make_instrumented_wrapper(assemblies, collector, boost_scale=1.3)
    ccf._replay_one_event = wrapper

    t0 = time.time()
    try:
        r = ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf._replay_one_event = orig_fn

    elapsed = time.time() - t0
    net = _last_net[0]

    fs  = np.nan_to_num(r.get('final_scores', np.zeros(4)), nan=0.0)
    rs  = compute_real_schema_index(net, assemblies, core_mask) if net else 0.0
    dai = compute_directional_alignment(list(_CENTROID_LOG), n_mem=4, core_size=CORE)
    rep_stats = r.get('rep_stats', {})

    s = collector.summary()

    print(f'\n  --- REPLAY EVENTS ---', flush=True)
    print(f'  Total replay events:      {s["n_events"]}', flush=True)
    print(f'  Ignition PASS:            {s["n_ignition_pass"]}', flush=True)
    print(f'  Ignition FAIL:            {s["n_ignition_fail"]}', flush=True)
    print(f'\n  --- COHERENCE PER EVENT ---', flush=True)
    print(f'  mean(mean_coh):           {s["mean_coh_mean"]:.4f}', flush=True)
    print(f'  max(mean_coh):            {s["mean_coh_max"]:.4f}', flush=True)
    print(f'  mean(peak_coh):           {s["peak_coh_mean"]:.4f}', flush=True)
    print(f'  max(peak_coh):            {s["peak_coh_max"]:.4f}', flush=True)
    print(f'  Steps where cv>COH_THR:   {s["n_coh_steps_total"]} total  '
          f'(mean {s["n_coh_steps_mean"]:.1f}/event)', flush=True)
    print(f'\n  --- MECHANISM FIRE EVIDENCE ---', flush=True)
    print(f'  M5 drift (unique->core delta):', flush=True)
    print(f'    total: {s["m5_delta_total"]:+.6f}  mean/event: {s["m5_delta_mean"]:+.6f}', flush=True)
    print(f'  M2 LTD (off-diag weight decrease):', flush=True)
    print(f'    total: {s["m2_ltd_total"]:.4f}  mean/event: {s["m2_ltd_mean"]:.4f}', flush=True)
    print(f'  STDP LTP (diag weight increase):', flush=True)
    print(f'    total: {s["stdp_ltp_total"]:.4f}  mean/event: {s["stdp_ltp_mean"]:.4f}', flush=True)
    print(f'\n  --- FINAL METRICS ---', flush=True)
    print(f'  DAI_core:   {dai.get("mean_core", 0):.4f}', flush=True)
    print(f'  REAL_SCHEMA:{rs:.4f}', flush=True)
    print(f'  Retention:  A={fs[0]:.3f}  B={fs[1]:.3f}  C={fs[2]:.3f}  D={fs[3]:.3f}', flush=True)
    print(f'  ret_mean:   {fs.mean():.4f}', flush=True)
    print(f'  Elapsed:    {elapsed:.0f}s', flush=True)

    return {
        'seed': seed, 'label': label, 'coh_thr': coh_thr,
        'n_events': s['n_events'],
        'n_ignition_pass': s['n_ignition_pass'],
        'n_ignition_fail': s['n_ignition_fail'],
        'mean_coh': s['mean_coh_mean'],
        'peak_coh_max': s['peak_coh_max'],
        'n_coh_steps_total': s['n_coh_steps_total'],
        'm5_delta_total': s['m5_delta_total'],
        'm2_ltd_total': s['m2_ltd_total'],
        'stdp_ltp_total': s['stdp_ltp_total'],
        'dai': float(dai.get('mean_core', 0)),
        'rs': float(rs),
        'ret_mean': float(fs.mean()),
        'ret_A': float(fs[0]),
    }


def print_comparison(r42, r2042):
    print(f'\n\n{"="*65}', flush=True)
    print('COMPARISON: seed=42 vs seed=2042 at COH_THR=0.00', flush=True)
    print(f'{"="*65}', flush=True)
    print(f'  {"Metric":<30}  {"seed=42":>10}  {"seed=2042":>10}  {"diff":>10}', flush=True)
    print('  ' + '-'*64, flush=True)
    fields = [
        ('n_events',          'Replay events'),
        ('n_ignition_pass',   'Ignition PASS'),
        ('n_ignition_fail',   'Ignition FAIL'),
        ('mean_coh',          'Mean coherence/event'),
        ('peak_coh_max',      'Max peak coherence'),
        ('n_coh_steps_total', 'Total coh steps (cv>THR)'),
        ('m5_delta_total',    'M5 drift total'),
        ('m2_ltd_total',      'M2 LTD total'),
        ('stdp_ltp_total',    'STDP LTP total'),
        ('dai',               'DAI_core'),
        ('rs',                'REAL_SCHEMA'),
        ('ret_mean',          'Retention mean'),
    ]
    for key, label in fields:
        v42   = r42.get(key, float('nan'))
        v2042 = r2042.get(key, float('nan'))
        diff  = v2042 - v42 if isinstance(v42, float) else float('nan')
        if isinstance(v42, int):
            print(f'  {label:<30}  {v42:>10d}  {v2042:>10d}', flush=True)
        else:
            print(f'  {label:<30}  {v42:>10.4f}  {v2042:>10.4f}  {diff:>+10.4f}', flush=True)

    print(f'\n{"="*65}', flush=True)
    print('CONCLUSION', flush=True)
    print(f'{"="*65}', flush=True)

    # Determine which hypothesis is supported
    coh_diff    = r2042['mean_coh'] - r42['mean_coh']
    ig_pass_42  = r42['n_ignition_pass']
    ig_pass_2k  = r2042['n_ignition_pass']
    m5_42       = abs(r42['m5_delta_total'])
    m5_2k       = abs(r2042['m5_delta_total'])
    m2_42       = abs(r42['m2_ltd_total'])
    m2_2k       = abs(r2042['m2_ltd_total'])
    coh_steps_42 = r42['n_coh_steps_total']
    coh_steps_2k = r2042['n_coh_steps_total']

    print(flush=True)
    if coh_steps_2k < 5 and coh_steps_42 > 20:
        print('HYPOTHESIS B: Mechanisms NOT firing in seed=2042.', flush=True)
        print(f'  seed=42   coherent steps: {coh_steps_42}', flush=True)
        print(f'  seed=2042 coherent steps: {coh_steps_2k}', flush=True)
        print('  Coherence never exceeds COH_THR=0.00 for seed=2042 -- no mechanism activation.', flush=True)
        print('  => Dynamical exception: seed=2042 replay is incoherent, gate never opens.', flush=True)
    elif m5_2k < 1e-4 and m2_2k < 1e-4 and m5_42 > 1e-4:
        print('HYPOTHESIS B (weight-based): Mechanisms not active in seed=2042.', flush=True)
        print(f'  seed=42   M5={m5_42:.4e}  M2={m2_42:.4e}', flush=True)
        print(f'  seed=2042 M5={m5_2k:.4e}  M2={m2_2k:.4e}', flush=True)
        print('  No weight changes from M2/M5 in seed=2042.', flush=True)
    elif ig_pass_2k == 0 and ig_pass_42 > 0:
        print('HYPOTHESIS B (ignition): Replay fails ignition in seed=2042.', flush=True)
        print(f'  seed=42   ignition PASS: {ig_pass_42}', flush=True)
        print(f'  seed=2042 ignition PASS: {ig_pass_2k}', flush=True)
        print('  Ignition always fails => spont phase cut to 15 steps => no STDP damage.', flush=True)
    elif abs(m5_2k) > abs(m5_42) * 0.5 and abs(m2_2k) > abs(m2_42) * 0.5:
        print('HYPOTHESIS A: Mechanisms ARE firing in seed=2042 (genuine dynamical exception).', flush=True)
        print(f'  seed=42   M5={m5_42:.4e}  M2={m2_42:.4e}  DAI={r42["dai"]:.4f}  RS={r42["rs"]:.4f}', flush=True)
        print(f'  seed=2042 M5={m5_2k:.4e}  M2={m2_2k:.4e}  DAI={r2042["dai"]:.4f}  RS={r2042["rs"]:.4f}', flush=True)
        print('  Despite mechanism firing, seed=2042 maintains RS/Ret.', flush=True)
        print('  => Seed initializes in a more robust basin; mechanism damage is insufficient to collapse.', flush=True)
    else:
        print('INCONCLUSIVE. Manual review of numbers above required.', flush=True)
        print(f'  seed=42   M5={m5_42:.4e}  M2={m2_42:.4e}  coh_steps={coh_steps_42}', flush=True)
        print(f'  seed=2042 M5={m5_2k:.4e}  M2={m2_2k:.4e}  coh_steps={coh_steps_2k}', flush=True)


if __name__ == '__main__':
    print('SEED=2042 IMMUNITY DIAGNOSTIC', flush=True)
    print(f'COH_THR override: {COH_THR_OVERRIDE}', flush=True)
    print(f'Runs: 2 x ~8 min = ~16 min total', flush=True)
    print(f'CCF constants: REPLAY_COHERENCE_THR={ccf.REPLAY_COHERENCE_THR}  '
          f'STDP_GATE_ENABLED={getattr(ccf,"STDP_GATE_ENABLED","?")}  '
          f'STDP_GATE_BIAS={getattr(ccf,"STDP_GATE_BIAS","?")}  '
          f'DEV_MODE={ccf.DEV_MODE}', flush=True)

    r42   = run_diagnostic(42,   'seed=42 (collapses)',  coh_thr=COH_THR_OVERRIDE)
    r2042 = run_diagnostic(2042, 'seed=2042 (immune)',   coh_thr=COH_THR_OVERRIDE)

    print_comparison(r42, r2042)
    print(f'\nDone.', flush=True)
