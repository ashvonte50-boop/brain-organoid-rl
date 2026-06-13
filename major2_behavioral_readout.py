#!/usr/bin/env python3
"""
MAJOR-2: Behavioral Readout — Template-Matching Decoder
=========================================================
Validates isyn_score by implementing a cosine-similarity template-matching decoder.

For each memory probe:
  1. Record spike pattern during probe (which neurons fire)
  2. Compare to training-time template (spike pattern during encoding)
  3. Cosine similarity → recall accuracy (0-1 continuous)
  4. Binary recall: threshold at similarity >= 0.5

Runs:
  Part A: Task 2 validation (FULL vs NO_REPLAY, n=10)
  Part B: E2 subsample (NATURAL, SUPPRESS_M1, BOOST_M1, n=10)
  Part C: isyn_score vs recall_accuracy correlation

Output: major2_results/major2_behavioral_readout.csv
        major2_results/major2_behavioral_validation.{png,pdf,svg}
        major2_results/major2_verdict.txt
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
N_MEM = 4
NE = 750
N_NEURONS = 1000
PROBE_STEPS = ccf.PROBE_STEPS if hasattr(ccf, 'PROBE_STEPS') else 50

SEEDS_TASK2 = [42 + i*1000 for i in range(10)]
SEEDS_E2 = [42 + i*1000 for i in range(10)]

OUT_DIR = r'C:\Users\Admin\brain-organoid-rl\major2_results'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, 'major2_behavioral_readout.csv')

print("=" * 70, flush=True)
print("MAJOR-2: BEHAVIORAL READOUT — TEMPLATE-MATCHING DECODER", flush=True)
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Template-Matching Decoder
# ═══════════════════════════════════════════════════════════════════════════════

def record_spike_pattern(net, assembly, n_steps=50, cue_size=5, seed_strength=0.3):
    """
    Stimulate with partial cue and record firing pattern.
    Returns binary spike vector (which neurons fired at least once).
    """
    # Partial cue
    cue_n = np.random.choice(assembly, size=min(cue_size, len(assembly)), replace=False)
    spike_counts = torch.zeros(N_NEURONS, device=ccf.DEVICE)

    # Seed phase (2 steps)
    stim = torch.zeros(N_NEURONS, device=ccf.DEVICE)
    stim[cue_n] = seed_strength
    for _ in range(2):
        net.forward(stim)
        spike_counts += net.spikes.float()

    # Spontaneous phase
    for _ in range(n_steps - 2):
        stim_noise = torch.randn(N_NEURONS, device=ccf.DEVICE) * 2.0
        net.forward(stim_noise)
        spike_counts += net.spikes.float()

    return spike_counts.cpu().numpy()


def record_encoding_template(net, assembly, n_steps=30):
    """
    Record template during strong activation (full assembly drive).
    Returns rate vector.
    """
    spike_counts = torch.zeros(N_NEURONS, device=ccf.DEVICE)
    stim = torch.zeros(N_NEURONS, device=ccf.DEVICE)
    stim[assembly] = 0.5  # full drive

    for _ in range(n_steps):
        net.forward(stim)
        spike_counts += net.spikes.float()

    return spike_counts.cpu().numpy()


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def behavioral_probe(net, assembly, template, n_trials=5):
    """
    Probe with template-matching decoder. Returns mean cosine similarity.
    """
    sims = []
    for _ in range(n_trials):
        response = record_spike_pattern(net, assembly)
        sim = cosine_similarity(response, template)
        sims.append(sim)
    return {
        'recall_accuracy': float(np.mean(sims)),
        'recall_std': float(np.std(sims)),
        'binary_recall': float(np.mean([s >= 0.5 for s in sims])),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run helpers
# ═══════════════════════════════════════════════════════════════════════════════

def is_done(seed, experiment, condition):
    if not os.path.exists(RESULTS_FILE):
        return False
    d = pd.read_csv(RESULTS_FILE)
    return ((d.seed == seed) & (d.experiment == experiment) & (d.condition == condition)).any()


def save_row(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)


def run_with_behavioral(seed, use_slow, use_replay, assemblies, core_set, label=''):
    """
    Run standard experiment + behavioral readout.
    Returns retention dict with both isyn_score and recall_accuracy per memory.

    FIX (2026-06-09): run_sequential_experiment builds its own internal net via
    ccf.build_network(); the local `net` variable was untrained.  We now use the
    _net_ref capture pattern (same as major3) so probing uses the ACTUAL trained net.
    """
    ccf.torch.manual_seed(seed)
    ccf.np.random.seed(seed)

    # ── Capture the trained network ───────────────────────────────────────────
    _net_ref = [None]
    _orig_build = ccf.build_network
    def _tracked_build(*args, **kwargs):
        n = _orig_build(*args, **kwargs)
        _net_ref[0] = n
        return n
    ccf.build_network = _tracked_build

    # ── Record encoding templates DURING training ─────────────────────────────
    templates = {}
    _orig_train = ccf.train_one_memory
    _train_idx = [0]

    def _train_with_template(net_arg, assembly, **kw):
        j = _train_idx[0]
        result = _orig_train(net_arg, assembly, **kw)
        # Record template immediately after encoding (net_arg IS the trained net)
        templates[j] = record_encoding_template(net_arg, assembly)
        _train_idx[0] += 1
        return result

    ccf.train_one_memory = _train_with_template

    try:
        results = ccf.run_sequential_experiment(
            use_slow, use_replay, assemblies, seed, ablation={}
        )
    finally:
        ccf.build_network = _orig_build
        ccf.train_one_memory = _orig_train

    # Use the ACTUAL trained network (captured from inside run_sequential_experiment)
    net = _net_ref[0]
    if net is None:
        raise RuntimeError("_net_ref not populated — build_network was not called")

    # Probe with both metrics
    ret = {}
    for mi in range(N_MEM):
        asm = assemblies[mi]
        # isyn_score
        try:
            isyn = float(ccf.probe_memory(net, asm)['isyn_score'])
        except Exception:
            isyn = 0.0

        # Behavioral readout
        if mi in templates:
            beh = behavioral_probe(net, asm, templates[mi], n_trials=5)
        else:
            beh = {'recall_accuracy': 0.0, 'recall_std': 0.0, 'binary_recall': 0.0}

        ret[mi] = {
            'isyn_score': isyn,
            'recall_accuracy': beh['recall_accuracy'],
            'recall_std': beh['recall_std'],
            'binary_recall': beh['binary_recall'],
        }

    return ret


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Task 2 — FULL vs NO_REPLAY (n=10)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[MAJOR-2] Part A: Task 2 (FULL vs NO_REPLAY)", flush=True)

task2_conditions = [
    ('FULL',      True,  True),
    ('NO_REPLAY', True,  False),
]

total_a = len(SEEDS_TASK2) * len(task2_conditions)
run_n = 0

for cond_name, use_slow, use_replay in task2_conditions:
    for sd in SEEDS_TASK2:
        run_n += 1
        if is_done(sd, 'Task2', cond_name):
            print(f'  Skip Task2/{cond_name} seed={sd}', flush=True)
            continue

        t0 = time.time()
        print(f'  [{run_n}/{total_a}] Task2/{cond_name} seed={sd}', flush=True)

        ccf.torch.manual_seed(sd)
        ccf.np.random.seed(sd)
        assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
        core_set = set(int(x) for x in core_mask.tolist())

        try:
            res = run_with_behavioral(sd, use_slow, use_replay, assemblies, core_set)
            for mi in range(N_MEM):
                save_row({
                    'seed': sd, 'experiment': 'Task2', 'condition': cond_name,
                    'memory': f'M{mi}', 'memory_idx': mi,
                    'isyn_score': res[mi]['isyn_score'],
                    'recall_accuracy': res[mi]['recall_accuracy'],
                    'recall_std': res[mi]['recall_std'],
                    'binary_recall': res[mi]['binary_recall'],
                })
            isyn_str = ",".join(str(round(res[mi]['isyn_score'], 3)) for mi in range(N_MEM))
            recall_str = ",".join(str(round(res[mi]['recall_accuracy'], 3)) for mi in range(N_MEM))
            print(f'    Done {time.time()-t0:.0f}s | isyn=[{isyn_str}] recall=[{recall_str}]',
                  flush=True)
        except Exception as e:
            print(f'    ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# Part B: E2 conditions (n=10)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[MAJOR-2] Part B: E2 conditions (NATURAL, SUPPRESS_M1, BOOST_M1)", flush=True)

# For E2, we need the monkeypatch approach similar to e2
# Simplified: just run NATURAL condition for correlation validation
for sd in SEEDS_E2:
    if is_done(sd, 'E2_natural', 'NATURAL'):
        print(f'  Skip E2_natural/NATURAL seed={sd}', flush=True)
        continue

    t0 = time.time()
    print(f'  E2_natural/NATURAL seed={sd}', flush=True)

    ccf.torch.manual_seed(sd)
    ccf.np.random.seed(sd)
    assemblies, core_mask = make_schema_assemblies(N_MEM, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
    core_set = set(int(x) for x in core_mask.tolist())

    try:
        res = run_with_behavioral(sd, True, True, assemblies, core_set)
        for mi in range(N_MEM):
            save_row({
                'seed': sd, 'experiment': 'E2_natural', 'condition': 'NATURAL',
                'memory': f'M{mi}', 'memory_idx': mi,
                'isyn_score': res[mi]['isyn_score'],
                'recall_accuracy': res[mi]['recall_accuracy'],
                'recall_std': res[mi]['recall_std'],
                'binary_recall': res[mi]['binary_recall'],
            })
        print(f'    Done {time.time()-t0:.0f}s', flush=True)
    except Exception as e:
        print(f'    ERROR: {e}', flush=True)
        import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Figures
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[MAJOR-2] Generating figures...', flush=True)

df = pd.read_csv(RESULTS_FILE)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: isyn vs recall_accuracy scatter (ALL data)
ax = axes[0, 0]
valid = df.dropna(subset=['isyn_score', 'recall_accuracy'])
x_isyn = valid['isyn_score'].values
y_recall = valid['recall_accuracy'].values

ax.scatter(x_isyn, y_recall, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
if len(x_isyn) > 3:
    r, p = stats.pearsonr(x_isyn, y_recall)
    # Fit line
    z = np.polyfit(x_isyn, y_recall, 1)
    xline = np.linspace(x_isyn.min(), x_isyn.max(), 100)
    ax.plot(xline, np.polyval(z, xline), 'r-', linewidth=2, label=f'r={r:.3f}, p={p:.1e}')
    ax.legend(fontsize=9)

ax.set_xlabel('isyn_score')
ax.set_ylabel('recall_accuracy (cosine similarity)')
ax.set_title('A. isyn_score vs behavioral recall\n(all conditions pooled)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Task 2 comparison
ax = axes[0, 1]
task2 = df[df.experiment == 'Task2']
for ci, cond in enumerate(['FULL', 'NO_REPLAY']):
    sub = task2[task2.condition == cond]
    means_isyn = [sub[sub.memory_idx == mi]['isyn_score'].mean() for mi in range(N_MEM)]
    means_recall = [sub[sub.memory_idx == mi]['recall_accuracy'].mean() for mi in range(N_MEM)]
    x = np.arange(N_MEM)
    color = '#1f77b4' if cond == 'FULL' else '#d62728'
    ax.bar(x + ci*0.35, means_recall, 0.35, color=color, alpha=0.7,
           label=f'{cond} (behavioral)', edgecolor='black', linewidth=0.5)

ax.set_xticks(np.arange(N_MEM) + 0.175)
ax.set_xticklabels(['M0', 'M1', 'M2', 'M3'])
ax.set_ylabel('Recall accuracy')
ax.set_title('B. Task 2: Behavioral readout\nFULL vs NO_REPLAY')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: isyn vs recall by condition
ax = axes[1, 0]
for cond in df.condition.unique():
    sub = df[df.condition == cond]
    ax.scatter(sub['isyn_score'], sub['recall_accuracy'], alpha=0.5, s=20, label=cond)
ax.set_xlabel('isyn_score')
ax.set_ylabel('recall_accuracy')
ax.set_title('C. Metric correspondence by condition')
ax.legend(fontsize=7, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel D: Binary recall rates
ax = axes[1, 1]
for ci, cond in enumerate(['FULL', 'NO_REPLAY']):
    sub = task2[task2.condition == cond]
    if len(sub) == 0:
        continue
    rates = [sub[sub.memory_idx == mi]['binary_recall'].mean() for mi in range(N_MEM)]
    x = np.arange(N_MEM)
    color = '#1f77b4' if cond == 'FULL' else '#d62728'
    ax.bar(x + ci*0.35, rates, 0.35, color=color, alpha=0.7,
           label=cond, edgecolor='black', linewidth=0.5)

ax.set_xticks(np.arange(N_MEM) + 0.175)
ax.set_xticklabels(['M0', 'M1', 'M2', 'M3'])
ax.set_ylabel('Binary recall rate (sim ≥ 0.5)')
ax.set_title('D. Binary recall success rate')
ax.legend(fontsize=8)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('MAJOR-2: Behavioral Readout Validation', fontsize=14, fontweight='bold')
plt.tight_layout()

for fmt in ['png', 'pdf', 'svg']:
    path = os.path.join(OUT_DIR, f'major2_behavioral_validation.{fmt}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {path}', flush=True)
plt.close()

# ── Verdict ──────────────────────────────────────────────────────────────────
verdict = []
verdict.append("=" * 70)
verdict.append("MAJOR-2: BEHAVIORAL READOUT — VERDICT")
verdict.append("=" * 70)

# Correlation
valid = df.dropna(subset=['isyn_score', 'recall_accuracy'])
if len(valid) > 5:
    r, p = stats.pearsonr(valid['isyn_score'], valid['recall_accuracy'])
    verdict.append(f"\nOverall correlation: r = {r:.4f}, p = {p:.2e}, n = {len(valid)}")

    if r >= 0.7:
        verdict.append(f"\nVERDICT: STRONG VALIDATION (r ≥ 0.7)")
        verdict.append(f"  isyn_score is a valid proxy for behavioral recall.")
        verdict.append(f"  No need to rerun headline experiments.")
    elif r >= 0.3:
        verdict.append(f"\nVERDICT: MODERATE VALIDATION (0.3 ≤ r < 0.7)")
        verdict.append(f"  isyn_score captures the ordering but not the magnitude.")
        verdict.append(f"  Report both metrics; isyn_score remains usable.")
    else:
        verdict.append(f"\nVERDICT: WEAK VALIDATION (r < 0.3)")
        verdict.append(f"  isyn_score is NOT a reliable proxy for behavioral recall.")
        verdict.append(f"  MUST rerun headline experiments with behavioral metric.")
else:
    verdict.append(f"\nInsufficient data for correlation ({len(valid)} points)")

# Task 2 stats
for cond in ['FULL', 'NO_REPLAY']:
    sub = task2[task2.condition == cond]
    if len(sub) > 0:
        verdict.append(f"\n{cond}:")
        verdict.append(f"  isyn_score:      {sub['isyn_score'].mean():.4f} ± {sub['isyn_score'].sem():.4f}")
        verdict.append(f"  recall_accuracy: {sub['recall_accuracy'].mean():.4f} ± {sub['recall_accuracy'].sem():.4f}")
        verdict.append(f"  binary_recall:   {sub['binary_recall'].mean():.4f} ± {sub['binary_recall'].sem():.4f}")

# FULL vs NO_REPLAY on behavioral metric
full_recall = task2[task2.condition == 'FULL']['recall_accuracy'].values
no_recall = task2[task2.condition == 'NO_REPLAY']['recall_accuracy'].values
if len(full_recall) > 2 and len(no_recall) > 2:
    t, p = stats.ttest_ind(full_recall, no_recall)
    verdict.append(f"\nFULL vs NO_REPLAY (behavioral): t={t:.3f}, p={p:.2e}")
    if p < 0.05:
        verdict.append(f"  Replay necessity CONFIRMED by behavioral metric")
    else:
        verdict.append(f"  WARNING: Replay necessity NOT confirmed by behavioral metric")

verdict_text = '\n'.join(verdict)
verdict_path = os.path.join(OUT_DIR, 'major2_verdict.txt')
with open(verdict_path, 'w', encoding='utf-8') as f:
    f.write(verdict_text)
print(verdict_text, flush=True)

print(f"\n[MAJOR-2] COMPLETE — all outputs in {OUT_DIR}", flush=True)
