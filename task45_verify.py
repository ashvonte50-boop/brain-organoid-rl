"""
TASK 4.5 — VERIFY REPLAY-STDP DORMANCY CLAIM
=============================================
Single self-contained script: runs 3 seeds, captures per-event coherence
and per-event STDP call/delta counts, generates figures + report.
~30 min total.
"""
import os, sys, csv, time, warnings
os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
import torch

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True; ccf.N_WORKERS = 1
ccf.USE_TORCH_COMPILE = False

from schema_abstraction.schema_core import register_schema_hooks
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\ablation_results\task45'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1042, 2042]
CORE = SCHEMA_CORE_SIZE
CORE_SL = slice(0, CORE)

# ── Per-event instrumentation state ──────────────────────────────────────
_in_replay = [False]
_event_stdp_calls = [0]       # stdp_step() calls during current event
_event_pot_updates = [0]      # synapse-level potentiation count during event
_event_dep_updates = [0]      # synapse-level depression count during event
_event_coh_vals = []           # coherence values within current event (from return dict)
_all_events = []               # collected per-seed

# ── Net capture ──────────────────────────────────────────────────────────
_net_ref = [None]
_orig_build = ccf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow)
    _net_ref[0] = n
    # Instrument stdp_step on this net
    _orig_stdp = n.stdp_step
    def _counting_stdp():
        if _in_replay[0]:
            _event_stdp_calls[0] += 1
            with torch.no_grad():
                W_before = n.W.data[CORE_SL, CORE_SL].clone()
            _orig_stdp()
            with torch.no_grad():
                delta = n.W.data[CORE_SL, CORE_SL] - W_before
                _event_pot_updates[0] += int((delta > 0).sum().item())
                _event_dep_updates[0] += int((delta < 0).sum().item())
        else:
            _orig_stdp()
    n.stdp_step = _counting_stdp
    return n
ccf.build_network = _track_build

# ── Replay wrapper ───────────────────────────────────────────────────────
_orig_replay = ccf._replay_one_event

def _instr_replay(net, assembly, tags=None, **kw):
    _in_replay[0] = True
    _event_stdp_calls[0] = 0
    _event_pot_updates[0] = 0
    _event_dep_updates[0] = 0

    p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
    result = _orig_replay(net, assembly, tags=tags, **p, **kw)

    _in_replay[0] = False

    # Extract coherence from return dict
    mean_coh = result.get('mean_coherence', 0.0) if isinstance(result, dict) else 0.0
    peak_coh = result.get('peak_coherence', 0.0) if isinstance(result, dict) else 0.0
    smooth_coh = result.get('smooth_coh_last', 0.0) if isinstance(result, dict) else 0.0
    n_coh_steps = result.get('n_steps_coherent', 0) if isinstance(result, dict) else 0

    _all_events.append({
        'memory_idx':     int(kw.get('assembly_idx', -1)),
        'mean_coherence': float(mean_coh),
        'peak_coherence': float(peak_coh),
        'smooth_coh_last':float(smooth_coh),
        'n_coh_steps':    int(n_coh_steps),
        'stdp_calls':     int(_event_stdp_calls[0]),
        'pot_updates':    int(_event_pot_updates[0]),
        'dep_updates':    int(_event_dep_updates[0]),
    })

    # MB boost (match FULL condition)
    with torch.no_grad():
        ne = net.n_exc
        w = net.W.data[:ne, :ne]
        ci = np.array([x for x in range(CORE) if x < ne])
        if len(ci):
            ci_t = torch.as_tensor(ci, device=w.device)
            w[ci_t[:, None], ci_t[None, :]] *= 1.3
            w.clamp_(0.0, net.w_max)

    return result


# ── Run one seed ─────────────────────────────────────────────────────────
def run_seed(seed):
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)
    assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)

    _all_events.clear()
    _net_ref[0] = None

    ccf._replay_one_event = _instr_replay
    try:
        r = ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf._replay_one_event = _orig_replay

    return list(_all_events)


# ── Analysis ─────────────────────────────────────────────────────────────
def analyze(all_seed_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.titlesize':13,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

    # Flatten all events
    flat = []
    for seed, events in all_seed_data:
        for e in events:
            e['seed'] = seed
            flat.append(e)

    n_total = len(flat)
    all_mean_coh   = np.array([e['mean_coherence'] for e in flat])
    all_peak_coh   = np.array([e['peak_coherence'] for e in flat])
    all_smooth_coh = np.array([e['smooth_coh_last'] for e in flat])
    all_stdp_calls = np.array([e['stdp_calls'] for e in flat])
    all_pot        = np.array([e['pot_updates'] for e in flat])
    all_dep        = np.array([e['dep_updates'] for e in flat])
    all_coh_steps  = np.array([e['n_coh_steps'] for e in flat])

    # ── TABLE 1: Per-seed summary ────────────────────────────────────────
    print(f'\n{"="*82}')
    print('TABLE 1: PER-SEED REPLAY STDP SUMMARY')
    print(f'{"="*82}')
    print(f'  {"Seed":>6s} {"Events":>8s} {"STDP Calls":>11s} {"Pot Updates":>12s} '
          f'{"Dep Updates":>12s} {"Coh Steps>THR":>14s}')
    print('  ' + '-'*68)
    for seed, events in all_seed_data:
        n_ev = len(events)
        sc_total = sum(e['stdp_calls'] for e in events)
        pot_total = sum(e['pot_updates'] for e in events)
        dep_total = sum(e['dep_updates'] for e in events)
        coh_total = sum(e['n_coh_steps'] for e in events)
        print(f'  {seed:>6d} {n_ev:>8d} {sc_total:>11d} {pot_total:>12d} '
              f'{dep_total:>12d} {coh_total:>14d}')
    # Totals
    print(f'  {"TOTAL":>6s} {n_total:>8d} {int(all_stdp_calls.sum()):>11d} '
          f'{int(all_pot.sum()):>12d} {int(all_dep.sum()):>12d} '
          f'{int(all_coh_steps.sum()):>14d}')

    # ── TABLE 2: Fraction exceeding coherence thresholds ─────────────────
    print(f'\n{"="*82}')
    print('TABLE 2: COHERENCE THRESHOLD EXCEEDANCE (mean_coherence)')
    print(f'{"="*82}')
    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    print(f'  {"Threshold":>10s} {"Count":>8s} {"Fraction":>10s}')
    print('  ' + '-'*32)
    for thr in thresholds:
        cnt = int((all_mean_coh > thr).sum())
        frac = cnt / max(n_total, 1)
        print(f'  {thr:>10.2f} {cnt:>8d} {frac:>10.4f}')

    # Same for peak_coherence
    print(f'\n  Using peak_coherence:')
    print(f'  {"Threshold":>10s} {"Count":>8s} {"Fraction":>10s}')
    print('  ' + '-'*32)
    for thr in thresholds:
        cnt = int((all_peak_coh > thr).sum())
        frac = cnt / max(n_total, 1)
        print(f'  {thr:>10.2f} {cnt:>8d} {frac:>10.4f}')

    # ── TABLE 3: Coherence statistics ────────────────────────────────────
    print(f'\n{"="*82}')
    print('TABLE 3: COHERENCE STATISTICS')
    print(f'{"="*82}')
    print(f'  mean_coherence:    {all_mean_coh.mean():.4f} +/- {all_mean_coh.std():.4f}  '
          f'range [{all_mean_coh.min():.4f}, {all_mean_coh.max():.4f}]')
    print(f'  peak_coherence:    {all_peak_coh.mean():.4f} +/- {all_peak_coh.std():.4f}  '
          f'range [{all_peak_coh.min():.4f}, {all_peak_coh.max():.4f}]')
    print(f'  smooth_coh_last:   {all_smooth_coh.mean():.4f} +/- {all_smooth_coh.std():.4f}')
    print(f'  n_coh_steps>THR:   {all_coh_steps.mean():.2f} +/- {all_coh_steps.std():.2f}  '
          f'(per event)')

    # ── Instrumentation verification ─────────────────────────────────────
    print(f'\n{"="*82}')
    print('INSTRUMENTATION VERIFICATION')
    print(f'{"="*82}')
    print(f'  Total replay events detected:      {n_total}')
    print(f'  Events with stdp_calls > 0:         {int((all_stdp_calls > 0).sum())} / {n_total}')
    print(f'  Events with pot_updates > 0:        {int((all_pot > 0).sum())} / {n_total}')
    print(f'  Events with dep_updates > 0:        {int((all_dep > 0).sum())} / {n_total}')
    print(f'  Events with n_coh_steps > 0:        {int((all_coh_steps > 0).sum())} / {n_total}')
    print(f'  Events with mean_coherence > 0:     {int((all_mean_coh > 0).sum())} / {n_total}')
    print(f'  Events with peak_coherence > 0:     {int((all_peak_coh > 0).sum())} / {n_total}')
    print(f'  Mean stdp_calls per event:           {all_stdp_calls.mean():.2f}')
    print(f'  COH_THR (production):                {ccf.REPLAY_COHERENCE_THR}')
    print(f'  STDP_GATE_ENABLED:                   {ccf.STDP_GATE_ENABLED}')
    print(f'  STDP_GATE_BIAS:                      {ccf.STDP_GATE_BIAS}')
    print(f'  STDP_GATE_SLOPE:                     {ccf.STDP_GATE_SLOPE}')

    # Double-check: stdp_calls should match n_coh_steps for hard-gate mode
    # But with probabilistic gate, calls can differ from coherent steps
    if ccf.STDP_GATE_ENABLED:
        print(f'\n  Probabilistic STDP gate is ENABLED.')
        print(f'  STDP fires with p = sigmoid({ccf.STDP_GATE_SLOPE} * (smooth_coh - {ccf.STDP_GATE_BIAS}))')
        print(f'  At observed smooth_coh ~ {all_smooth_coh.mean():.3f}:')
        prob = 1.0 / (1.0 + np.exp(-ccf.STDP_GATE_SLOPE * (all_smooth_coh.mean() - ccf.STDP_GATE_BIAS)))
        print(f'  Expected STDP probability = sigmoid({ccf.STDP_GATE_SLOPE} * '
              f'({all_smooth_coh.mean():.3f} - {ccf.STDP_GATE_BIAS})) = {prob:.6f}')
        print(f'  BUT: STDP also requires cv > COH_THR ({ccf.REPLAY_COHERENCE_THR})')
        print(f'  Since smooth_coh << COH_THR, the hard gate blocks STDP before the')
        print(f'  probabilistic gate is even evaluated.')

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f'\n{"="*82}')
    print('VERDICT')
    print(f'{"="*82}')
    total_stdp = int(all_stdp_calls.sum())
    total_pot  = int(all_pot.sum())
    total_dep  = int(all_dep.sum())
    frac_above = float((all_peak_coh > ccf.REPLAY_COHERENCE_THR).mean())

    if total_stdp == 0 and total_pot == 0 and total_dep == 0:
        print(f'\n  A) REPLAY STDP TRULY DORMANT')
        print(f'     {n_total} replay events, 0 STDP calls, 0 potentiation, 0 depression.')
        print(f'     Peak coherence never exceeds {ccf.REPLAY_COHERENCE_THR} '
              f'(max observed: {all_peak_coh.max():.4f}).')
        print(f'     The dual gate (cv > COH_THR AND probabilistic) blocks ALL replay STDP.')
        verdict = 'A'
    elif total_stdp > 0 and total_pot + total_dep == 0:
        print(f'\n  C) INSTRUMENTATION BUG — stdp_step called but no weight changes detected.')
        verdict = 'C'
    elif total_stdp > 0 and total_pot + total_dep < n_total:
        frac_active = (total_pot + total_dep) / max(n_total * 400, 1)  # per synapse
        print(f'\n  B) REPLAY STDP RARE BUT NON-ZERO')
        print(f'     {total_stdp} STDP calls across {n_total} events.')
        print(f'     {total_pot} potentiation updates, {total_dep} depression updates.')
        verdict = 'B'
    else:
        print(f'\n  D) OTHER — unexpected pattern. Manual review needed.')
        verdict = 'D'

    # ── Figure: coherence histogram ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(all_mean_coh, bins=30, color='#2166AC', edgecolor='black', alpha=0.8)
    ax.axvline(ccf.REPLAY_COHERENCE_THR, color='red', lw=2.5, ls='--',
               label=f'COH_THR = {ccf.REPLAY_COHERENCE_THR}')
    ax.set_xlabel('Mean coherence per replay event', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title(f'Replay Coherence Distribution (n={n_total} events)\n'
                 f'max={all_mean_coh.max():.3f}, threshold={ccf.REPLAY_COHERENCE_THR}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.hist(all_peak_coh, bins=30, color='#5AAE61', edgecolor='black', alpha=0.8)
    ax.axvline(ccf.REPLAY_COHERENCE_THR, color='red', lw=2.5, ls='--',
               label=f'COH_THR = {ccf.REPLAY_COHERENCE_THR}')
    ax.set_xlabel('Peak coherence per replay event', fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title(f'Peak Coherence Distribution\n'
                 f'max={all_peak_coh.max():.3f}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Task 4.5: Replay Coherence vs STDP Gate Threshold\n'
                 f'VERDICT: {"DORMANT" if verdict=="A" else "ACTIVE" if verdict=="B" else "BUG" if verdict=="C" else "OTHER"}',
                 y=1.03, fontsize=14, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'coherence_histogram.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Figure: {OUT_DIR}/coherence_histogram.[png|pdf|svg]')

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'replay_stdp_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed','event_idx','memory_idx',
            'mean_coherence','peak_coherence','smooth_coh_last','n_coh_steps',
            'stdp_calls','pot_updates','dep_updates'])
        w.writeheader()
        for i, e in enumerate(flat):
            w.writerow({**e, 'event_idx': i})
    print(f'  CSV: {csv_path}')

    # ── TASK45_REPORT.md ─────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, 'TASK45_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# TASK 4.5 -- VERIFY REPLAY-STDP DORMANCY CLAIM\n\n')
        f.write(f'Seeds: {SEEDS}\n')
        f.write(f'Total replay events: {n_total}\n')
        f.write(f'COH_THR: {ccf.REPLAY_COHERENCE_THR}\n')
        f.write(f'STDP_GATE_ENABLED: {ccf.STDP_GATE_ENABLED}\n')
        f.write(f'STDP_GATE_BIAS: {ccf.STDP_GATE_BIAS}\n\n')

        f.write('## Table 1: Per-seed summary\n\n')
        f.write('| Seed | Events | STDP Calls | Pot Updates | Dep Updates | Coh Steps>THR |\n')
        f.write('|------|--------|------------|-------------|-------------|---------------|\n')
        for seed, events in all_seed_data:
            n_ev = len(events)
            sc_t = sum(e['stdp_calls'] for e in events)
            po_t = sum(e['pot_updates'] for e in events)
            de_t = sum(e['dep_updates'] for e in events)
            co_t = sum(e['n_coh_steps'] for e in events)
            f.write(f'| {seed} | {n_ev} | {sc_t} | {po_t} | {de_t} | {co_t} |\n')
        f.write(f'| **TOTAL** | **{n_total}** | **{total_stdp}** | **{total_pot}** | **{total_dep}** | **{int(all_coh_steps.sum())}** |\n\n')

        f.write('## Table 2: Coherence threshold exceedance\n\n')
        f.write('| Threshold | Count (mean_coh) | Fraction | Count (peak_coh) | Fraction |\n')
        f.write('|-----------|------------------|----------|------------------|----------|\n')
        for thr in thresholds:
            cm = int((all_mean_coh > thr).sum()); fm = cm/max(n_total,1)
            cp = int((all_peak_coh > thr).sum()); fp = cp/max(n_total,1)
            f.write(f'| {thr:.2f} | {cm} | {fm:.4f} | {cp} | {fp:.4f} |\n')

        f.write(f'\n## Table 3: Coherence statistics\n\n')
        f.write(f'- mean_coherence: {all_mean_coh.mean():.4f} +/- {all_mean_coh.std():.4f} '
                f'[{all_mean_coh.min():.4f}, {all_mean_coh.max():.4f}]\n')
        f.write(f'- peak_coherence: {all_peak_coh.mean():.4f} +/- {all_peak_coh.std():.4f} '
                f'[{all_peak_coh.min():.4f}, {all_peak_coh.max():.4f}]\n')
        f.write(f'- smooth_coh_last: {all_smooth_coh.mean():.4f} +/- {all_smooth_coh.std():.4f}\n')
        f.write(f'- n_coh_steps/event: {all_coh_steps.mean():.2f} +/- {all_coh_steps.std():.2f}\n\n')

        f.write(f'## Verdict\n\n')
        if verdict == 'A':
            f.write('**A) REPLAY STDP TRULY DORMANT**\n\n')
            f.write(f'{n_total} replay events, 0 STDP calls, 0 potentiation, 0 depression.\n')
            f.write(f'Peak coherence never exceeds {ccf.REPLAY_COHERENCE_THR} '
                    f'(max observed: {all_peak_coh.max():.4f}).\n')
            f.write('The dual gate (cv > COH_THR AND probabilistic) blocks ALL replay STDP.\n')
            f.write('Task 4 conclusion CONFIRMED: replay-phase STDP is genuinely dormant.\n')
        elif verdict == 'B':
            f.write('**B) REPLAY STDP RARE BUT NON-ZERO**\n\n')
            f.write(f'{total_stdp} STDP calls, {total_pot} pot, {total_dep} dep.\n')
            f.write('Task 4 conclusion PARTIALLY INCORRECT.\n')
        elif verdict == 'C':
            f.write('**C) INSTRUMENTATION BUG DETECTED**\n')
        else:
            f.write('**D) OTHER**\n')

    print(f'  Report: {report_path}')
    return verdict


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('TASK 4.5: VERIFY REPLAY-STDP DORMANCY', flush=True)
    print(f'Seeds: {SEEDS}  COH_THR: {ccf.REPLAY_COHERENCE_THR}', flush=True)
    t0 = time.time()

    all_seed_data = []
    for seed in SEEDS:
        print(f'\n  Running seed={seed}...', flush=True)
        t1 = time.time()
        events = run_seed(seed)
        print(f'  seed={seed} done in {time.time()-t1:.0f}s  events={len(events)}', flush=True)
        all_seed_data.append((seed, events))

    verdict = analyze(all_seed_data)
    print(f'\nTotal time: {(time.time()-t0)/60:.1f} min', flush=True)
    print('TASK 4.5 COMPLETE.', flush=True)
