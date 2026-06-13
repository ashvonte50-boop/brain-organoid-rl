#!/usr/bin/env python3
"""
MAJOR-1: The Scheduling-Artifact Test
=======================================
Tests whether the primacy gradient is a pure replay-scheduling artifact.

Design: 15 seeds × 3 conditions:
  NATURAL       — default RGCC (interference_aware replay scheduling)
  EQUALIZED     — forced equal replay counts per memory (round-robin)
  EQUALIZED_POS — equal replay counts + position-matched encoding
                  (all memories get same number of presentations)

The key manipulation: in EQUALIZED, we monkeypatch the replay loop to force
exact round-robin allocation (each memory gets N_events/N_mem replays per rest).
If the primacy gradient vanishes → temporal priority is a scheduling artifact.
If the gradient persists → there is a consolidation-order effect beyond replay counts.

Part 1a: Analytical harmonic-series derivation (written to file)
Part 1b: Simulation

Output: major1_results/major1_decoupling.csv
        major1_results/major1_scheduling_test.{png,pdf,svg}
        major1_results/major1_verdict.txt
        major1_results/major1_harmonic_derivation.txt
"""
import os, sys, time, warnings
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

os.environ['DEV_MODE'] = '1'
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')

import compare_catastrophic_forgetting as ccf
ccf.DEV_MODE = True
ccf.N_WORKERS = 1

from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _CENTROID_LOG, _last_net

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS = [42 + i*1000 for i in range(15)]
N_MEM = 4
NE = 750
BASE_PRES = ccf._N_PRESENTATIONS  # 7 in DEV

CONDITIONS = ['NATURAL', 'EQUALIZED', 'EQUALIZED_POS']

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\major1_results'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, 'major1_decoupling.csv')

print("=" * 70, flush=True)
print("MAJOR-1: THE SCHEDULING-ARTIFACT TEST", flush=True)
print("=" * 70, flush=True)
print(f"Seeds: {len(SEEDS)}, Conditions: {CONDITIONS}", flush=True)
print(f"Total runs: {len(SEEDS) * len(CONDITIONS)}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1a: Harmonic Series Derivation
# ═══════════════════════════════════════════════════════════════════════════════
harmonic_text = r"""
==============================================================================
MAJOR-1 Part 1a: ANALYTICAL HARMONIC-SERIES DERIVATION
==============================================================================

SETUP:
  - N memories encoded sequentially: M0, M1, ..., M_{N-1}
  - After encoding memory k, there is a rest period with R total replay events
  - Memory k is first eligible for replay starting at rest period k
  - Total rest periods: N-1 (between consecutive memories)

REPLAY ALLOCATION UNDER UNIFORM SCHEDULING:
  At rest period r (after encoding M_r), memories M_0..M_r are eligible.
  Under uniform scheduling, each eligible memory gets R/(r+1) replay events.

  Expected total replay for memory k across all rest periods:
    E[replay(k)] = Sum_{r=k}^{N-2} R/(r+1)
                 = R * Sum_{r=k}^{N-2} 1/(r+1)
                 = R * (H(N-1) - H(k))

  where H(n) = Sum_{j=1}^{n} 1/j is the n-th harmonic number.

FOR N=4 MEMORIES, R=15 REPLAY EVENTS PER REST:
  H(1) = 1.000
  H(2) = 1.500
  H(3) = 1.833

  E[replay(M0)] = R * (H(3) - H(0)) = 15 * 1.833 = 27.5
  E[replay(M1)] = R * (H(3) - H(1)) = 15 * 0.833 = 12.5
  E[replay(M2)] = R * (H(3) - H(2)) = 15 * 0.333 = 5.0
  E[replay(M3)] = 0 (last memory, no rest follows)

  RATIO M0:M3 = 27.5:0 = infinity (M3 gets zero replay!)

  More conservatively, under interference_aware mode (which weights by
  retention deficit), the allocation is biased toward struggling memories,
  but earlier memories still get exposed to more rest periods.

ACTUAL REST STRUCTURE:
  Rest 0 (after M0): M0 eligible  → all R events go to M0
  Rest 1 (after M1): M0, M1      → ~R/2 each (or biased)
  Rest 2 (after M2): M0, M1, M2  → ~R/3 each (or biased)
  No rest after M3 (last memory).

TOTAL EXPOSURE:
  M0: 3 rest periods (rests 0, 1, 2)
  M1: 2 rest periods (rests 1, 2)
  M2: 1 rest period  (rest 2)
  M3: 0 rest periods (never replayed)

PREDICTION:
  Under EQUALIZED replay (forced round-robin at each rest), the per-rest
  allocation is equalized, but M0 still benefits from more rest periods.
  The scheduling artifact should be REDUCED but may not be fully eliminated
  because M0 undergoes consolidation EARLIER (temporal position effect).

  If the gradient vanishes completely → pure scheduling artifact
  If the gradient persists significantly → consolidation-order effect exists
"""

deriv_path = os.path.join(OUT_DIR, 'major1_harmonic_derivation.txt')
with open(deriv_path, 'w', encoding='utf-8') as f:
    f.write(harmonic_text)
print(f"[1a] Harmonic derivation saved: {deriv_path}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1b: Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def is_done(seed, cond):
    if not os.path.exists(RESULTS_FILE):
        return False
    d = pd.read_csv(RESULTS_FILE)
    return ((d.seed == seed) & (d.condition == cond)).any()

def save_row(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)


def run_major1(seed, condition, assemblies, assemblies_np, core_set):
    """
    Run one trial with the specified replay condition.

    NATURAL: standard RGCC
    EQUALIZED: monkeypatch replay to force round-robin
    EQUALIZED_POS: round-robin + all memories get same presentations
    """
    _net_ref = [None]
    _replay_log = []
    _train_idx = [0]
    W_encode = {}

    _orig_build = ccf.build_network
    _orig_replay = ccf._replay_one_event
    _orig_train = ccf.train_one_memory

    # Track network
    def _track_build(use_slow=False):
        n = _orig_build(use_slow=use_slow)
        _net_ref[0] = n
        return n
    ccf.build_network = _track_build

    # For EQUALIZED_POS: all memories get same presentations
    def _train_hook(net, assembly, **kw):
        _net_ref[0] = net
        j = _train_idx[0]
        if condition == 'EQUALIZED_POS':
            kw = dict(kw)
            kw['n_presentations'] = BASE_PRES  # all same
        r = _orig_train(net, assembly, **kw)
        asm = assemblies[j]
        uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < NE]
        with torch.no_grad():
            Wf = net.W.detach().cpu().numpy()
        W_encode[j] = float(Wf[np.ix_(uniq, uniq)].mean()) if len(uniq) >= 2 else float('nan')
        _train_idx[0] += 1
        return r
    ccf.train_one_memory = _train_hook

    # For EQUALIZED/EQUALIZED_POS: force round-robin replay
    _round_robin_counter = [0]
    _n_learned = [0]

    def _equalized_replay(net, assembly, tags=None, **kw):
        """Round-robin: cycle through learned assemblies in order."""
        _net_ref[0] = net
        _last_net[0] = net
        # Determine how many assemblies are currently learned
        # The assembly passed is one of the learned assemblies
        # We override which assembly gets replayed
        n = _n_learned[0]
        if n <= 0:
            n = 1
        idx = _round_robin_counter[0] % n
        _round_robin_counter[0] += 1

        p = dict(cue_size=4, seed_strength=0.3, seed_dur=2, spont_steps=5, noise=8.0)
        result = _orig_replay(net, assemblies_np[idx], tags=tags, **p, **kw)
        _replay_log.append(idx)
        return result

    if condition in ('EQUALIZED', 'EQUALIZED_POS'):
        ccf._replay_one_event = _equalized_replay

    # We also need to track n_learned for round-robin — monkeypatch the rest function
    _orig_rest = ccf.inter_memory_rest_with_replay

    def _tracked_rest(net, learned_assemblies, **kw):
        _n_learned[0] = len(learned_assemblies)
        _round_robin_counter[0] = 0  # reset per rest for clean allocation
        return _orig_rest(net, learned_assemblies, **kw)

    if condition in ('EQUALIZED', 'EQUALIZED_POS'):
        ccf.inter_memory_rest_with_replay = _tracked_rest

    _CENTROID_LOG.clear()
    _last_net[0] = None
    _replay_log.clear()
    _train_idx[0] = 0
    _round_robin_counter[0] = 0
    _n_learned[0] = 0

    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)

    try:
        ccf.run_sequential_experiment(True, True, assemblies, seed, ablation={})
    finally:
        ccf.build_network = _orig_build
        ccf._replay_one_event = _orig_replay
        ccf.train_one_memory = _orig_train
        ccf.inter_memory_rest_with_replay = _orig_rest

    net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
    assert net is not None, f'Net not captured seed={seed}'

    # Probe all memories
    ret = []
    for asm in assemblies:
        try:
            ret.append(float(ccf.probe_memory(net, asm)['isyn_score']))
        except Exception:
            ret.append(0.0)
    ret = np.nan_to_num(ret, nan=0.0)

    # W_slow
    with torch.no_grad():
        WS = net.W_slow.cpu().numpy()

    from collections import Counter
    mc = Counter(_replay_log)
    replay_counts = [mc.get(i, 0) for i in range(N_MEM)]

    wslow = {}
    for mi in range(N_MEM):
        asm = assemblies[mi]
        uniq = [int(x) for x in asm if int(x) not in core_set and int(x) < NE]
        wslow[mi] = float(WS[np.ix_(uniq, uniq)].mean()) if len(uniq) >= 2 else float('nan')

    return {
        'retention': ret,
        'replay': replay_counts,
        'wslow': wslow,
        'W_encode': W_encode,
    }


# ── Main loop ────────────────────────────────────────────────────────────────
total = len(CONDITIONS) * len(SEEDS)
run_n = 0
t_global = time.time()

for cond in CONDITIONS:
    for sd in SEEDS:
        run_n += 1
        if is_done(sd, cond):
            print(f'[MAJOR-1] Skip {cond} seed={sd} — done', flush=True)
            continue

        t0 = time.time()
        print(f'\n[MAJOR-1] Run {run_n}/{total}: {cond} seed={sd}', flush=True)

        ccf.torch.manual_seed(sd)
        ccf.np.random.seed(sd)
        assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
        assemblies_np = [np.array(a) for a in assemblies]
        core_set = set(int(x) for x in core_mask.tolist())

        try:
            res = run_major1(sd, cond, assemblies, assemblies_np, core_set)

            row = {
                'seed': sd,
                'condition': cond,
            }
            for mi in range(N_MEM):
                row[f'M{mi}_retention'] = float(res['retention'][mi])
                row[f'M{mi}_replay'] = res['replay'][mi]
                row[f'M{mi}_wslow'] = res['wslow'][mi]
                row[f'M{mi}_W_encode'] = res['W_encode'].get(mi, float('nan'))

            save_row(row)

            elapsed = time.time() - t0
            remaining = (total - run_n) * elapsed
            print(f'[MAJOR-1] Done {elapsed:.0f}s | '
                  f'M0={res["retention"][0]:.3f} M1={res["retention"][1]:.3f} '
                  f'M2={res["retention"][2]:.3f} M3={res["retention"][3]:.3f} | '
                  f'replay={res["replay"]} | '
                  f'ETA ~{remaining/3600:.1f}h',
                  flush=True)
        except Exception as e:
            print(f'[MAJOR-1] ERROR seed={sd} cond={cond}: {e}', flush=True)
            import traceback
            traceback.print_exc()

elapsed_total = time.time() - t_global
print(f'\n[MAJOR-1] All runs complete in {elapsed_total/3600:.2f}h', flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Figures
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[MAJOR-1] Generating figures...', flush=True)

df = pd.read_csv(RESULTS_FILE)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Retention by position × condition
ax = axes[0]
colors = {'NATURAL': '#1f77b4', 'EQUALIZED': '#ff7f0e', 'EQUALIZED_POS': '#2ca02c'}
x_pos = np.arange(N_MEM)
width = 0.25

for ci, cond in enumerate(CONDITIONS):
    sub = df[df.condition == cond]
    means = [sub[f'M{mi}_retention'].mean() for mi in range(N_MEM)]
    sems = [sub[f'M{mi}_retention'].sem() for mi in range(N_MEM)]
    ax.bar(x_pos + ci * width, means, width, yerr=sems, capsize=3,
           color=colors[cond], label=cond, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(['M0', 'M1', 'M2', 'M3'])
ax.set_ylabel('isyn_score (retention)')
ax.set_title('A. Retention by memory position\nand replay scheduling condition')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Primacy gradient (M0 - M3) per condition
ax = axes[1]
gradients = {}
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    grads = sub['M0_retention'].values - sub['M3_retention'].values
    gradients[cond] = grads

cond_labels = CONDITIONS
grad_means = [np.mean(gradients[c]) for c in cond_labels]
grad_sems = [stats.sem(gradients[c]) for c in cond_labels]

bars = ax.bar(range(3), grad_means, yerr=grad_sems, capsize=5,
              color=[colors[c] for c in cond_labels], edgecolor='black', linewidth=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(cond_labels, fontsize=8, rotation=15)
ax.set_ylabel('Primacy gradient (M0 − M3)')
ax.set_title('B. Scheduling artifact test\n(0 = no gradient)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Stats: NATURAL vs EQUALIZED
if len(gradients['NATURAL']) > 2 and len(gradients['EQUALIZED']) > 2:
    t_ne, p_ne = stats.ttest_ind(gradients['NATURAL'], gradients['EQUALIZED'])
    ax.text(0.5, 0.95, f'NATURAL vs EQUALIZED:\nt={t_ne:.2f}, p={p_ne:.4f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel C: Replay counts (verification that equalization worked)
ax = axes[2]
for ci, cond in enumerate(CONDITIONS):
    sub = df[df.condition == cond]
    means = [sub[f'M{mi}_replay'].mean() for mi in range(N_MEM)]
    ax.bar(x_pos + ci * width, means, width, color=colors[cond],
           label=cond, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(['M0', 'M1', 'M2', 'M3'])
ax.set_ylabel('Total replay count')
ax.set_title('C. Replay allocation\n(verification of equalization)')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('MAJOR-1: Scheduling Artifact Test — Is Primacy a Replay-Scheduling Artifact?',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for fmt in ['png', 'pdf', 'svg']:
    path = os.path.join(OUT_DIR, f'major1_scheduling_test.{fmt}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {path}', flush=True)
plt.close()

# ── Verdict ──────────────────────────────────────────────────────────────────
verdict_lines = []
verdict_lines.append("=" * 70)
verdict_lines.append("MAJOR-1: SCHEDULING ARTIFACT TEST — VERDICT")
verdict_lines.append("=" * 70)

for cond in CONDITIONS:
    sub = df[df.condition == cond]
    verdict_lines.append(f"\n{cond} (n={len(sub)}):")
    for mi in range(N_MEM):
        m = sub[f'M{mi}_retention'].mean()
        s = sub[f'M{mi}_retention'].sem()
        verdict_lines.append(f"  M{mi}: {m:.4f} ± {s:.4f}")
    grad = sub['M0_retention'].values - sub['M3_retention'].values
    verdict_lines.append(f"  Gradient (M0-M3): {np.mean(grad):.4f} ± {stats.sem(grad):.4f}")

# Key comparison
nat_grad = gradients['NATURAL']
eq_grad = gradients['EQUALIZED']
if len(nat_grad) > 2 and len(eq_grad) > 2:
    t_val, p_val = stats.ttest_ind(nat_grad, eq_grad)
    reduction = 1 - np.mean(eq_grad) / max(np.mean(nat_grad), 1e-10)

    verdict_lines.append(f"\n{'='*70}")
    verdict_lines.append(f"NATURAL gradient:    {np.mean(nat_grad):.4f} ± {stats.sem(nat_grad):.4f}")
    verdict_lines.append(f"EQUALIZED gradient:  {np.mean(eq_grad):.4f} ± {stats.sem(eq_grad):.4f}")
    verdict_lines.append(f"Gradient reduction:  {100*reduction:.1f}%")
    verdict_lines.append(f"t-test: t={t_val:.3f}, p={p_val:.6f}")

    if p_val < 0.05 and reduction > 0.5:
        verdict_lines.append(f"\nVERDICT: OUTCOME A — Equalization SIGNIFICANTLY reduces primacy gradient")
        verdict_lines.append(f"  The primacy gradient is primarily a SCHEDULING ARTIFACT.")
        verdict_lines.append(f"  Temporal priority arises from differential replay exposure,")
        verdict_lines.append(f"  not from an intrinsic consolidation-order advantage.")
    elif p_val >= 0.05:
        verdict_lines.append(f"\nVERDICT: OUTCOME B — Equalization does NOT significantly reduce gradient")
        verdict_lines.append(f"  There is a consolidation-order effect BEYOND replay scheduling.")
        verdict_lines.append(f"  The primacy gradient has a component that is not explained by")
        verdict_lines.append(f"  differential replay counts alone.")
    else:
        verdict_lines.append(f"\nVERDICT: PARTIAL — Gradient reduced but not eliminated")
        verdict_lines.append(f"  Scheduling explains ~{100*reduction:.0f}% of the primacy gradient,")
        verdict_lines.append(f"  but a residual order effect remains.")

    # One-sample t-test: is EQUALIZED gradient significantly > 0?
    t_eq0, p_eq0 = stats.ttest_1samp(eq_grad, 0)
    verdict_lines.append(f"\nEQUALIZED gradient vs 0: t={t_eq0:.3f}, p={p_eq0:.6f}")
    if p_eq0 < 0.05:
        verdict_lines.append(f"  EQUALIZED gradient is still significantly > 0 → residual order effect")
    else:
        verdict_lines.append(f"  EQUALIZED gradient is NOT significant → pure scheduling artifact")

verdict_text = '\n'.join(verdict_lines)
verdict_path = os.path.join(OUT_DIR, 'major1_verdict.txt')
with open(verdict_path, 'w', encoding='utf-8') as f:
    f.write(verdict_text)
print(verdict_text.encode('ascii', 'replace').decode('ascii'), flush=True)

print(f"\n[MAJOR-1] COMPLETE — all outputs in {OUT_DIR}", flush=True)
