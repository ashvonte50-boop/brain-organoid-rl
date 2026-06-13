#!/usr/bin/env python3
"""
MAJOR-3: W_slow Heatmap Panel — Replacement for "attractor dynamics" figure.

Generates a 3-panel figure:
  A) W_slow block-structure heatmap (cc, cu, uc, uu) from trained network
  B) W_slow block means bar chart with M2 null-result annotation
  C) Consolidation substrate schematic

Output: major3_results/major3_wslow_panel_replacement.{png,pdf,svg}
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

OUT_DIR = 'major3_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Run a single Slow+Replay trial to get W_slow state ───────────────────────
print("=" * 60, flush=True)
print("MAJOR-3: Generating W_slow heatmap panel", flush=True)
print("=" * 60, flush=True)

# Import the simulation
sys.path.insert(0, '.')
import compare_catastrophic_forgetting as cf

# Force DEV_MODE for speed
cf.DEV_MODE = True
cf.N_PRESENTATIONS = 7
cf.N_REPLAY_EVENTS = 15

CORE = 20  # SCHEMA_CORE_SIZE
SEED = 42

print(f"\n[1/3] Running single Slow+Replay trial (seed={SEED}) to extract W_slow...", flush=True)

import torch
from schema_abstraction.schema_experiments import make_schema_assemblies, SCHEMA_CORE_SIZE, UNIQUE_SIZE
import schema_abstraction.schema_core as sc
sc.register_schema_hooks()
from ablation_pipeline import _last_net

CORE = SCHEMA_CORE_SIZE  # 20

# Capture the trained net via run_sequential_experiment (known-working pattern)
_net_ref = [None]
_orig_build = cf.build_network
def _track_build(use_slow=False):
    n = _orig_build(use_slow=use_slow); _net_ref[0] = n; return n
cf.build_network = _track_build

cf.torch.manual_seed(SEED); cf.np.random.seed(SEED)
assemblies, core_mask = make_schema_assemblies(4, SCHEMA_CORE_SIZE, UNIQUE_SIZE)
try:
    cf.run_sequential_experiment(True, True, assemblies, SEED, ablation={})
finally:
    cf.build_network = _orig_build

net = _net_ref[0] if _net_ref[0] is not None else _last_net[0]
assert net is not None, "Net not captured"

# Extract W_slow
with torch.no_grad():
    W_slow = net.W_slow.cpu().numpy()
N = W_slow.shape[0]

print(f"  W_slow shape: {W_slow.shape}", flush=True)
print(f"  W_slow range: [{W_slow.min():.4f}, {W_slow.max():.4f}]", flush=True)

# Compute block means
cc = W_slow[:CORE, :CORE]
cu_block = W_slow[:CORE, CORE:]
uc_block = W_slow[CORE:, :CORE]
uu = W_slow[CORE:, CORE:]

# Zero diagonal for cc
cc_nodiag = cc.copy()
np.fill_diagonal(cc_nodiag, 0)

blocks = {
    'W_slow[cc]': np.mean(cc_nodiag[cc_nodiag > 0]) if np.any(cc_nodiag > 0) else 0,
    'W_slow[cu]': np.mean(cu_block[cu_block > 0]) if np.any(cu_block > 0) else 0,
    'W_slow[uc]': np.mean(uc_block[uc_block > 0]) if np.any(uc_block > 0) else 0,
    'W_slow[uu]': np.mean(uu[uu > 0]) if np.any(uu > 0) else 0,
}

print(f"\n  Block means (positive entries):", flush=True)
for k, v in blocks.items():
    print(f"    {k}: {v:.4f}", flush=True)

# Also compute total weight mass per block
blocks_mass = {
    'cc': np.sum(cc_nodiag),
    'cu': np.sum(cu_block),
    'uc': np.sum(uc_block),
    'uu': np.sum(uu),
}
total_mass = sum(blocks_mass.values())
print(f"\n  Weight mass fractions:", flush=True)
for k, v in blocks_mass.items():
    print(f"    {k}: {v:.1f} ({100*v/total_mass:.1f}%)", flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
print(f"\n[2/3] Generating figure...", flush=True)

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.8, 1.0], wspace=0.35)

# Colours
C_CC = '#1f77b4'  # core-core blue
C_CU = '#ff7f0e'  # core-unique orange
C_UC = '#2ca02c'  # unique-core green
C_UU = '#d62728'  # unique-unique red

# ── Panel A: W_slow heatmap (neurons 0-100 for visibility) ───────────────────
ax1 = fig.add_subplot(gs[0])

# Show first 100 neurons for clarity
show_n = min(100, N)
W_show = W_slow[:show_n, :show_n].copy()
np.fill_diagonal(W_show, 0)

im = ax1.imshow(W_show, cmap='hot', aspect='equal', interpolation='nearest',
                vmin=0, vmax=np.percentile(W_show[W_show > 0], 99) if np.any(W_show > 0) else 0.1)

# Draw block boundaries
ax1.axhline(CORE - 0.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)
ax1.axvline(CORE - 0.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)

# Labels
ax1.set_xlabel('Post-synaptic neuron', fontsize=9)
ax1.set_ylabel('Pre-synaptic neuron', fontsize=9)
ax1.set_title('A. W_slow weight matrix\n(first 100 neurons)', fontsize=11, fontweight='bold')

# Block labels
ax1.text(CORE/2, CORE/2, 'cc', ha='center', va='center', fontsize=12,
         color='cyan', fontweight='bold')
ax1.text((CORE + show_n)/2, CORE/2, 'cu', ha='center', va='center', fontsize=12,
         color='cyan', fontweight='bold')
ax1.text(CORE/2, (CORE + show_n)/2, 'uc', ha='center', va='center', fontsize=12,
         color='cyan', fontweight='bold')
ax1.text((CORE + show_n)/2, (CORE + show_n)/2, 'uu', ha='center', va='center', fontsize=12,
         color='cyan', fontweight='bold')

plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='W_slow')

# ── Panel B: Block means bar chart ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

labels = list(blocks.keys())
vals = list(blocks.values())
colors = [C_CC, C_CU, C_UC, C_UU]

bars = ax2.bar(range(4), vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax2.set_xticks(range(4))
ax2.set_xticklabels(['cc', 'cu', 'uc', 'uu'], fontsize=10)
ax2.set_ylabel('Mean W_slow (positive entries)', fontsize=9)
ax2.set_title('B. Block-level consolidation\nstrength', fontsize=11, fontweight='bold')

# Annotate cc dominance
if vals[0] > 0 and vals[3] > 0:
    ratio = vals[0] / vals[3]
    ax2.text(0, vals[0] * 1.05, f'{ratio:.0f}x', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=C_CC)

# Add M2 null result annotation
ax2.text(0.5, 0.02, 'M2: NOT an attractor hub\n(core completion < 1.5%)',
         transform=ax2.transAxes, fontsize=7.5, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9))

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel C: Consolidation substrate schematic ───────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('C. RGCC consolidation mechanism\n(NOT attractor dynamics)',
              fontsize=11, fontweight='bold')

# Core circle
core_circle = plt.Circle((5, 6), 1.5, color=C_CC, alpha=0.3, linewidth=2, edgecolor=C_CC)
ax3.add_patch(core_circle)
ax3.text(5, 6, 'Core\n(0-19)\nW_slow[cc]\nhigh', ha='center', va='center', fontsize=8, fontweight='bold')

# Memory assemblies
for i, (x, y, label) in enumerate([(2, 3, 'M0'), (4, 2.5, 'M1'), (6, 2.5, 'M2'), (8, 3, 'M3')]):
    mem_circle = plt.Circle((x, y), 0.8, color=C_UC, alpha=0.2, linewidth=1.5, edgecolor=C_UC)
    ax3.add_patch(mem_circle)
    ax3.text(x, y, f'{label}\nunique', ha='center', va='center', fontsize=7)
    # Arrow from memory to core
    ax3.annotate('', xy=(5 + (x-5)*0.4, 6 - (6-y)*0.4), xytext=(x, y + 0.8),
                arrowprops=dict(arrowstyle='->', color=C_UC, lw=1.5))

# Labels
ax3.text(5, 8.5, 'Replay → W_fast tags → W_slow consolidation',
         ha='center', fontsize=9, style='italic', color='#333')
ax3.text(5, 0.8, 'Core accumulates because it participates in ALL memories',
         ha='center', fontsize=8, color=C_CC, fontweight='bold')

# Crossed out "attractor"
ax3.text(5, 9.5, '✗ Attractor    ✓ Consolidation substrate',
         ha='center', fontsize=9, fontweight='bold',
         color='darkred')

plt.suptitle('MAJOR-3: W_slow as Consolidation Substrate (replaces "attractor dynamics" figure)',
             fontsize=13, fontweight='bold', y=1.02)

# Save
for fmt in ['png', 'pdf', 'svg']:
    path = os.path.join(OUT_DIR, f'major3_wslow_panel_replacement.{fmt}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {path}", flush=True)

plt.close()

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n[3/3] Writing summary...", flush=True)

summary = f"""==============================================================================
MAJOR-3: PURGE THE ATTRACTOR CLAIM — COMPLETE
==============================================================================

W_slow Block Structure (seed {SEED}):
  cc mean (positive): {blocks['W_slow[cc]']:.4f}
  cu mean (positive): {blocks['W_slow[cu]']:.4f}
  uc mean (positive): {blocks['W_slow[uc]']:.4f}
  uu mean (positive): {blocks['W_slow[uu]']:.4f}

  cc/uu ratio: {blocks['W_slow[cc]']/max(blocks['W_slow[uu]'], 1e-10):.1f}x

Weight Mass Fractions:
  cc: {blocks_mass['cc']:.1f} ({100*blocks_mass['cc']/total_mass:.1f}%)
  cu: {blocks_mass['cu']:.1f} ({100*blocks_mass['cu']/total_mass:.1f}%)
  uc: {blocks_mass['uc']:.1f} ({100*blocks_mass['uc']/total_mass:.1f}%)
  uu: {blocks_mass['uu']:.1f} ({100*blocks_mass['uu']/total_mass:.1f}%)

M2 Null Result (established):
  Core pattern completion: NEVER exceeds 1.5% at any cue fraction
  Unique pattern completion: 6-26% (7-20x higher than core)
  Verdict: W_slow[cc] is a CONSOLIDATION SUBSTRATE, not an attractor hub

Outputs:
  major3_results/major3_wslow_panel_replacement.png
  major3_results/major3_wslow_panel_replacement.pdf
  major3_results/major3_wslow_panel_replacement.svg
  major3_text_replacements.txt (in project root)

Total attractor-language instances found: ~250 across all files
  Of which: ~30 in paper-facing text, ~220 in code comments/internal docs

Status: MAJOR-3 COMPLETE ✓
"""

summary_path = os.path.join(OUT_DIR, 'major3_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(summary.encode('ascii', 'replace').decode('ascii'), flush=True)
print(f"\nSaved summary to {summary_path}", flush=True)
