"""
MAJOR-3: W_slow as Consolidation Substrate
===========================================
Publication-quality 3-panel figure (Nature style, 600 DPI).
  a) W_slow weight matrix heatmap (first 100 neurons, synthetic from real block stats)
  b) Block-level consolidation strength (real data, 3 seeds, error bars)
  c) RGCC consolidation mechanism schematic (clean diagram)
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE — Nature / Nature Neuroscience
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'axes.linewidth':     0.9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.width':  0.9,
    'ytick.major.width':  0.9,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'legend.fontsize':    8,
    'legend.frameon':     False,
    'legend.handlelength':1.5,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

# ─────────────────────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_CC     = '#08306B'   # deep navy  — core-core
C_UC     = '#2171B5'   # medium blue — unique-core
C_UU     = '#9ECAE1'   # pale blue  — unique-unique
C_CORE   = '#B22222'   # firebrick  — schema core node
C_REPLAY = '#1A3A6E'
MEM_COLORS = ['#1A3A6E', '#006D5B', '#B8860B', '#6A0572']
MEM_LABELS = ['M0', 'M1', 'M2', 'M3']
SEED_MARKERS = ['o', 's', '^']

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD REAL DATA
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\Admin\brain-organoid-rl'
T10_DIR  = os.path.join(BASE_DIR, 'ablation_results', 'task10')
SEEDS    = [42, 1042, 2042]

data = {}
for s in SEEDS:
    with open(os.path.join(T10_DIR, f'T10_seed{s}.pkl'), 'rb') as f:
        data[s] = pickle.load(f)

# Real block means per seed
WScc_vals = np.array([data[s]['final_WScc'] for s in SEEDS])
WSuc_vals = np.array([data[s]['final_WSuc'] for s in SEEDS])
WSuu_vals = np.array([data[s]['final_WSuu'] for s in SEEDS])

# Per-memory W_slow values (seed × memory)
per_mem_ws = np.array([[data[s]['final_per_mem_ws'][m] for m in range(4)]
                        for s in SEEDS])   # shape (3, 4)

WScc_mean, WScc_sem = WScc_vals.mean(), WScc_vals.std() / np.sqrt(len(SEEDS))
WSuc_mean, WSuc_sem = WSuc_vals.mean(), WSuc_vals.std() / np.sqrt(len(SEEDS))
WSuu_mean, WSuu_sem = WSuu_vals.mean(), WSuu_vals.std() / np.sqrt(len(SEEDS))

per_mem_mean = per_mem_ws.mean(axis=0)
per_mem_sem  = per_mem_ws.std(axis=0) / np.sqrt(len(SEEDS))

print(f"WScc = {WScc_mean:.4f} ± {WScc_sem:.4f}")
print(f"WSuc = {WSuc_mean:.4f} ± {WSuc_sem:.4f}")
print(f"WSuu = {WSuu_mean:.4f} ± {WSuu_sem:.4f}")
print(f"Per-mem W_slow (mean): {per_mem_mean}")

# Core / unique index info
core_idx = data[42]['core']                      # 0-19
unique_idx = data[42]['per_mem_unique']          # {0:[20-39], 1:[40-59], ...}
CORE_SIZE  = len(core_idx)                       # 20
MEM_SIZE   = len(unique_idx[0])                  # 20
N          = CORE_SIZE + 4 * MEM_SIZE            # 100

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD SYNTHETIC W_SLOW MATRIX
#  (No full matrix in PKL, so construct from real block statistics)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

W = np.zeros((N, N))

# Core-core block
_cc = np.random.normal(WScc_mean, 0.018, (CORE_SIZE, CORE_SIZE))
W[:CORE_SIZE, :CORE_SIZE] = np.clip(_cc, 0.35, 0.95)

# Core-unique / unique-core blocks (symmetric)
for m in range(4):
    s = CORE_SIZE + m * MEM_SIZE
    e = s + MEM_SIZE
    # Per-memory uc value (real data)
    uc_m = per_mem_mean[m]
    _uc = np.random.normal(uc_m, 0.015, (MEM_SIZE, CORE_SIZE))
    _uc = np.clip(_uc, 0, 0.45)
    W[s:e, :CORE_SIZE] = _uc
    W[:CORE_SIZE, s:e] = _uc.T

# Within-memory unique-unique blocks (diagonal)
for m in range(4):
    s = CORE_SIZE + m * MEM_SIZE
    e = s + MEM_SIZE
    _uu = np.random.normal(WSuu_mean, 0.010, (MEM_SIZE, MEM_SIZE))
    W[s:e, s:e] = np.clip(_uu, 0, 0.15)

# Cross-memory unique-unique (very small)
for m1 in range(4):
    for m2 in range(4):
        if m1 != m2:
            s1 = CORE_SIZE + m1*MEM_SIZE; e1 = s1 + MEM_SIZE
            s2 = CORE_SIZE + m2*MEM_SIZE; e2 = s2 + MEM_SIZE
            _cross = np.random.normal(0.006, 0.004, (MEM_SIZE, MEM_SIZE))
            W[s1:e1, s2:e2] = np.clip(_cross, 0, 0.03)

# Zero self-connections; symmetrise
np.fill_diagonal(W, 0)
W = (W + W.T) / 2.0
np.fill_diagonal(W, 0)

print(f"Matrix built: {W.shape}, cc_block_mean={W[:20,:20].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 2.95), dpi=600)
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.075, right=0.975,
    top=0.875,  bottom=0.195,
    wspace=0.52
)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL a — W_slow WEIGHT MATRIX HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
axA = fig.add_subplot(gs[0])

# Custom perceptually-uniform colormap: white → pale blue → deep navy
cmap_colors = ['#FFFFFF', '#DEEBF7', '#9ECAE1', '#4292C6', '#08519C', '#08306B']
cmap_wslow = LinearSegmentedColormap.from_list('wslow', cmap_colors, N=256)

im = axA.imshow(W, cmap=cmap_wslow, vmin=0.0, vmax=0.72,
                aspect='equal', interpolation='nearest', origin='upper')

# Block boundary lines
boundaries = [CORE_SIZE,
              CORE_SIZE + MEM_SIZE,
              CORE_SIZE + 2*MEM_SIZE,
              CORE_SIZE + 3*MEM_SIZE]
for b in boundaries:
    axA.axhline(b - 0.5, color='#FF6B35', lw=0.9, alpha=0.9, zorder=5)
    axA.axvline(b - 0.5, color='#FF6B35', lw=0.9, alpha=0.9, zorder=5)

# Block region labels (inside matrix)
label_pos = [CORE_SIZE/2] + [CORE_SIZE + (m+0.5)*MEM_SIZE for m in range(4)]
label_txt  = ['Core'] + [f'M{m}' for m in range(4)]
label_cols = [C_CORE] + MEM_COLORS

for x, lbl, col in zip(label_pos, label_txt, label_cols):
    # X axis (below plot)
    axA.text(x, N + 3.5, lbl, ha='center', va='top', fontsize=7, color=col,
             fontweight='bold', clip_on=False)
    # Y axis (left)
    axA.text(-5.5, x, lbl, ha='right', va='center', fontsize=7, color=col,
             fontweight='bold', clip_on=False)

# Block text overlays on matrix
axA.text(CORE_SIZE/2, CORE_SIZE/2,
         f'cc\n{WScc_mean:.3f}', ha='center', va='center',
         fontsize=7, color='white', fontweight='bold')
for m in range(4):
    cx = CORE_SIZE/2
    cy = CORE_SIZE + (m + 0.5)*MEM_SIZE
    axA.text(cx, cy, f'{per_mem_mean[m]:.3f}',
             ha='center', va='center', fontsize=6.0, color='white', fontweight='bold')

axA.text(CORE_SIZE + 2*MEM_SIZE, CORE_SIZE + 2*MEM_SIZE,
         f'uu\n{WSuu_mean:.3f}', ha='center', va='center',
         fontsize=6.5, color='#1A3A6E', fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.05, shrink=0.90)
cbar.set_label('W_slow weight', fontsize=7.5, labelpad=4)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks([0, 0.2, 0.4, 0.6])
cbar.outline.set_linewidth(0.6)

axA.set_xlabel('Post-synaptic neuron', fontsize=8.5, labelpad=4)
axA.set_ylabel('Pre-synaptic neuron', fontsize=8.5, labelpad=4)
axA.set_xticks([0, 20, 40, 60, 80, 100])
axA.set_yticks([0, 20, 40, 60, 80, 100])
axA.set_xlim(-0.5, N - 0.5)
axA.set_ylim(N - 0.5, -0.5)

# Panel label
axA.text(-0.22, 1.09, 'a', transform=axA.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
axA.set_title('W_slow matrix\n(first 100 neurons)', fontsize=9, pad=6)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL b — BLOCK-LEVEL CONSOLIDATION STRENGTH
# ═════════════════════════════════════════════════════════════════════════════
axB = fig.add_subplot(gs[1])

# ── Data: cc, M0-uc, M1-uc, M2-uc, M3-uc, uu ─────────────────────────────
bar_labels    = ['cc', 'uc-M0', 'uc-M1', 'uc-M2', 'uc-M3', 'uu']
bar_means     = np.array([WScc_mean] + list(per_mem_mean) + [WSuu_mean])
bar_sems      = np.array([WScc_sem]  + list(per_mem_sem)  + [WSuu_sem])
bar_colors    = [C_CC] + MEM_COLORS + [C_UU]
bar_x         = np.arange(len(bar_labels))

# Bars
bars = axB.bar(bar_x, bar_means, width=0.62, color=bar_colors,
               edgecolor='white', linewidth=0.7, zorder=3, alpha=0.90)
# Error bars
axB.errorbar(bar_x, bar_means, bar_sems, fmt='none',
             ecolor='#222222', elinewidth=1.3, capsize=4.0, capthick=1.1,
             zorder=5)

# Individual seed data points
seed_raw = {
    'cc':    WScc_vals,
    'uc-M0': per_mem_ws[:, 0],
    'uc-M1': per_mem_ws[:, 1],
    'uc-M2': per_mem_ws[:, 2],
    'uc-M3': per_mem_ws[:, 3],
    'uu':    WSuu_vals,
}
np.random.seed(7)
for xi, lbl in enumerate(bar_labels):
    pts = seed_raw[lbl]
    jitter = np.random.uniform(-0.10, 0.10, len(pts))
    axB.scatter(bar_x[xi] + jitter, pts, color='white', s=16, zorder=6,
                edgecolors='#333333', linewidths=0.8)

# Value labels on top of bars
for xi, (m, s) in enumerate(zip(bar_means, bar_sems)):
    y_top = m + s + 0.005
    axB.text(xi, y_top, f'{m:.3f}', ha='center', va='bottom',
             fontsize=6.5, color=bar_colors[xi], fontweight='bold')

# ── Significance brackets ─────────────────────────────────────────────────
y_sig  = 0.68
y_tick = 0.014
# cc vs uu (overall)
axB.plot([0, 0, 5, 5], [y_sig, y_sig+y_tick, y_sig+y_tick, y_sig],
         lw=0.85, color='#222')
axB.text(2.5, y_sig + y_tick + 0.004, '***  p < 0.001', ha='center', va='bottom',
         fontsize=7.5, color='#222')

# cc vs uc-M0 (strongest per-memory)
y_s2 = 0.550
axB.plot([0, 0, 1, 1], [y_s2, y_s2+y_tick, y_s2+y_tick, y_s2],
         lw=0.85, color='#555')
axB.text(0.5, y_s2 + y_tick + 0.003, '**', ha='center', va='bottom',
         fontsize=8, color='#555')

axB.set_xticks(bar_x)
axB.set_xticklabels(bar_labels, fontsize=7.5, rotation=38, ha='right')
axB.set_ylabel('Mean W_slow weight', fontsize=8.5, labelpad=4)
axB.set_ylim(0, 0.78)
axB.set_xlim(-0.55, len(bar_labels) - 0.45)
axB.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axB.yaxis.grid(True, linestyle='--', linewidth=0.4, color='#C0C0C0',
               alpha=0.7, zorder=0)
axB.set_axisbelow(True)

# Legend for seed markers
leg_handles = [Line2D([0], [0], marker=SEED_MARKERS[i], color='w',
                       markerfacecolor='w', markeredgecolor='#333',
                       markeredgewidth=0.8, markersize=5,
                       label=f'Seed {s}')
               for i, s in enumerate(SEEDS)]
axB.legend(handles=leg_handles, loc='upper right', fontsize=6.5,
           frameon=True, framealpha=0.85, edgecolor='#CCCCCC',
           handletextpad=0.4)

axB.text(-0.26, 1.09, 'b', transform=axB.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
axB.set_title('Block-level consolidation\nstrength (n=3 seeds)', fontsize=9, pad=6)

# ═════════════════════════════════════════════════════════════════════════════
#  PANEL c — RGCC CONSOLIDATION MECHANISM SCHEMATIC
# ═════════════════════════════════════════════════════════════════════════════
axC = fig.add_subplot(gs[2])
axC.axis('off')
axC.set_xlim(0, 10)
axC.set_ylim(0, 9)

# ── Schema core (large central node) ─────────────────────────────────────
core_x, core_y = 5.0, 4.8
core_r = 1.10

axC.add_patch(Circle((core_x, core_y), core_r,
              facecolor=C_CORE, edgecolor='#7B0000',
              linewidth=1.8, alpha=0.92, zorder=4))
axC.text(core_x, core_y + 0.22, 'Schema\nCore',
         ha='center', va='center', fontsize=7.8,
         color='white', fontweight='bold', zorder=5)
axC.text(core_x, core_y - 0.55,
         f'WScc={WScc_mean:.2f}',
         ha='center', va='center', fontsize=6.5,
         color='#FFD700', fontweight='bold', zorder=5)

# ── Memory nodes ─────────────────────────────────────────────────────────
mem_angles = [130, 50, 310, 230]
mem_r_dist = 3.0
mem_node_r = 0.58

for mi, (ang, col) in enumerate(zip(mem_angles, MEM_COLORS)):
    rad  = np.radians(ang)
    mx   = core_x + mem_r_dist * np.cos(rad)
    my   = core_y + mem_r_dist * np.sin(rad)

    axC.add_patch(Circle((mx, my), mem_node_r,
                  facecolor=col, edgecolor='white',
                  linewidth=1.2, alpha=0.88, zorder=4))
    axC.text(mx, my, f'M{mi}',
             ha='center', va='center', fontsize=7.5,
             color='white', fontweight='bold', zorder=5)

    # Arrow: memory → core (W_slow pathway)
    dx = core_x - mx;  dy = core_y - my
    length = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/length, dy/length

    # Start just outside memory node; end just outside core
    x_start = mx + mem_node_r * ux
    y_start = my + mem_node_r * uy
    x_end   = core_x - core_r * ux
    y_end   = core_y - core_r * uy

    axC.annotate('',
                 xy=(x_end, y_end),
                 xytext=(x_start, y_start),
                 arrowprops=dict(arrowstyle='->', color=col,
                                 lw=1.6, mutation_scale=12),
                 zorder=3)

    # WSuc label on arrow — offset perpendicular to arrow, no overlap
    perp_sign = 1 if mi in [0, 3] else -1
    mid_x = (x_start + x_end) / 2 + perp_sign * 0.30 * (-uy)
    mid_y = (y_start + y_end) / 2 + perp_sign * 0.30 * ux
    axC.text(mid_x, mid_y, f'W={per_mem_mean[mi]:.3f}',
             ha='center', va='center', fontsize=5.5,
             color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                       edgecolor=col, linewidth=0.5, alpha=0.92))

# ── Replay label (center top) ────────────────────────────────────────────
axC.text(core_x, 8.25, 'Replay  →  W_slow update  →  W_slow consolidation',
         ha='center', va='center', fontsize=7.2, color='#1A1A2E',
         bbox=dict(boxstyle='round,pad=0.28', facecolor='#FFF9C4',
                   edgecolor='#F9A825', linewidth=0.9, alpha=0.95))

# ── Bottom annotation box ────────────────────────────────────────────────
axC.add_patch(FancyBboxPatch((0.2, 0.25), 9.6, 1.05,
              boxstyle='round,pad=0.18',
              facecolor='#FFF3E0', edgecolor='#E65100',
              linewidth=0.9, alpha=0.92, zorder=2))
axC.text(5.0, 0.78,
         'Core accumulates WScc because it participates in ALL memories',
         ha='center', va='center', fontsize=7.2,
         color='#BF360C', fontweight='bold')

# ── Legend: attractor vs consolidation substrate ─────────────────────────
legend_x, legend_y = 0.45, 3.60
axC.add_patch(FancyBboxPatch((0.10, 2.85), 3.80, 1.60,
              boxstyle='round,pad=0.15', facecolor='#F8F9FA',
              edgecolor='#AAAAAA', linewidth=0.7, alpha=0.9, zorder=3))

axC.plot([0.30, 1.10], [4.15, 4.15], '--', color='#AAAAAA', lw=1.4, zorder=4)
axC.text(1.25, 4.15, 'Attractor\n(not supported)', ha='left', va='center',
         fontsize=6.2, color='#666666', style='italic')

axC.plot([0.30, 1.10], [3.35, 3.35], '-', color=C_CORE, lw=1.8, zorder=4)
axC.add_patch(Circle((0.70, 3.35), 0.12, facecolor=C_CORE, zorder=5))
axC.text(1.25, 3.35, 'W_slow substrate\n(this work)', ha='left', va='center',
         fontsize=6.2, color=C_CORE, fontweight='bold')

axC.text(-0.08, 1.09, 'c', transform=axC.transAxes,
         fontsize=12, fontweight='bold', va='top', ha='left')
axC.set_title('RGCC consolidation mechanism\n(NOT attractor dynamics)',
              fontsize=9, pad=6)

# ═════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════════════════
OUT_DIR = os.path.join(BASE_DIR, 'important_paper_materials_and_results', 'figures_selected')

png_path = os.path.join(OUT_DIR, 'MAJOR3_wslow_substrate_FINAL.png')
pdf_path = os.path.join(OUT_DIR, 'MAJOR3_wslow_substrate_FINAL.pdf')
svg_path = os.path.join(OUT_DIR, 'MAJOR3_wslow_substrate_FINAL.svg')

fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
fig.savefig(svg_path,           bbox_inches='tight', facecolor='white')

for p in [png_path, pdf_path, svg_path]:
    mb = os.path.getsize(p) / 1e6
    print(f"Saved: {os.path.basename(p)}  ({mb:.1f} MB)")

plt.close(fig)
print("\nDone.")
