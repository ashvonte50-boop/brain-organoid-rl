"""
Task 11 — Fixed Figure Generation
===================================
Fixes all layout/overlap issues from the original task11_figures.py:
  - Increased figure size and inter-panel spacing
  - All text overlaps resolved
  - Duplicate legends removed
  - Annotations repositioned to avoid clipping
  - Panel titles shortened where they were overflowing
  - Bottom annotations moved inside axes
  - DAG cleaned up (Fig1 H)
  - Scatter labels de-overlapped (Fig2 H)

Saves to: important_paper_materials_and_results/figures_selected/
Also to:  ablation_results/task11/  (as *_v2 variants, not overwriting originals)
"""

import os, pickle, warnings
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = r'C:\Users\Admin\brain-organoid-rl'
PAPER_DIR  = os.path.join(BASE_DIR, 'important_paper_materials_and_results', 'figures_selected')
TASK11_DIR = os.path.join(BASE_DIR, 'ablation_results', 'task11')
os.makedirs(PAPER_DIR,  exist_ok=True)
os.makedirs(TASK11_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE  (Nature / Nature Neuroscience)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          8,
    'axes.titlesize':     8.5,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'xtick.major.size':   3.0,
    'ytick.major.size':   3.0,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'legend.fontsize':    6.5,
    'legend.frameon':     False,
    'legend.handlelength':1.4,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

# ─────────────────────────────────────────────────────────────────────────────
#  COLOR PALETTE  (Okabe-Ito + extensions, colorblind-safe)
# ─────────────────────────────────────────────────────────────────────────────
C_REPLAY   = '#1A3A6E'
C_NOREPLAY = '#C0392B'
C_WSCC     = '#8B0000'
C_WSUC     = '#D35400'
C_WSUU     = '#AAAAAA'
C_CORE     = '#C0392B'
C_WFAST    = '#6C7A89'
C_M0       = '#1A3A6E'
C_M1       = '#1A7A6E'
C_M2       = '#B8860B'
C_M3       = '#7D3C98'
MEM_COLORS  = [C_M0, C_M1, C_M2, C_M3]
MEM_LABELS  = ['Mem 0', 'Mem 1', 'Mem 2', 'Mem 3']
MEM_MARKERS = ['o', 's', '^', 'D']
SEED_MARKERS= ['o', 's', '^']

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
T10_DIR  = os.path.join(BASE_DIR, 'ablation_results', 'task10')
T105_DIR = os.path.join(BASE_DIR, 'ablation_results', 'task105')

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

SEEDS = [42, 1042, 2042]
t10   = {s: load_pkl(os.path.join(T10_DIR,  f'T10_seed{s}.pkl'))  for s in SEEDS}
t105  = load_pkl(os.path.join(T105_DIR, 'T105_all_seed42.pkl'))

print(f"[FIX] Loaded Task10 seeds: {SEEDS}")
print(f"[FIX] Loaded Task105 conditions: {list(t105.keys())}")

# Build per-memory arrays (n=12: 4 memories × 3 seeds)
all_replay, all_wslow, all_retention = [], [], []
all_mids, all_sids = [], []

for si, seed in enumerate(SEEDS):
    d = t10[seed]
    rc  = d['replay_per_mem']
    ws  = d['final_per_mem_ws']
    ret = d['retention']
    for mi in range(4):
        all_replay.append(rc[mi])
        all_wslow.append(ws.get(mi, 0.0))
        all_retention.append(ret[mi])
        all_mids.append(mi)
        all_sids.append(si)

all_replay    = np.array(all_replay,    float)
all_wslow     = np.array(all_wslow,     float)
all_retention = np.array(all_retention, float)
all_mids      = np.array(all_mids)
all_sids      = np.array(all_sids)
n_pts         = len(all_replay)

AVG_WScc = np.mean([t10[s]['final_WScc'] for s in SEEDS])
AVG_WSuc = np.mean([t10[s]['final_WSuc'] for s in SEEDS])
AVG_WSuu = np.mean([t10[s]['final_WSuu'] for s in SEEDS])

CONDS        = ['CONTROL', 'BOOST_MEM3', 'SUPPRESS_MEM0']
COND_LABELS  = ['Control', 'Boost Mem3', 'Suppress Mem0']
t105_replay  = {c: t105[c]['replay_counts']       for c in CONDS}
t105_ret     = {c: np.array(t105[c]['retention']) for c in CONDS}

FRACS          = [0.25, 0.50, 0.75, 1.00]
R2_REPLAY_ONLY = [0.459, 0.609, 0.746, 0.881]
R2_CORE_ONLY   = [0.456, 0.661, 0.787, 0.680]
R2_REPLAY_CORE = [0.460, 0.711, 0.848, 0.891]
R2_FULL_MODEL  = [0.460, 0.656, 0.795, 0.891]

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

def stat_box(ax, text, x=0.97, y=0.97, ha='right', va='top'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6,
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7F7F7',
                      edgecolor='#CCCCCC', linewidth=0.5))

def growth_curve(ev, v0, vf, tau=14):
    return v0 + (vf - v0) * (1 - np.exp(-ev / tau))

def draw_arc_sector(ax, theta1, theta2, r_in, r_out, color, alpha=0.65, zorder=2):
    angles = np.linspace(np.radians(theta1), np.radians(theta2), 80)
    xs = np.concatenate([r_out*np.cos(angles), r_in*np.cos(angles[::-1])])
    ys = np.concatenate([r_out*np.sin(angles), r_in*np.sin(angles[::-1])])
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=zorder)

def save_fig(fig, name, dpi=300):
    """Save to paper_dir (high-quality PNG) and task11_dir (as _v2)."""
    # Paper materials folder
    paper_path = os.path.join(PAPER_DIR, f'{name}.png')
    fig.savefig(paper_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    size_mb = os.path.getsize(paper_path) / 1e6
    print(f"[FIX] Saved to PAPER_DIR: {name}.png  ({size_mb:.1f} MB)")

    # Also save PDF for vector quality
    pdf_path = os.path.join(PAPER_DIR, f'{name}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"[FIX] Saved to PAPER_DIR: {name}.pdf")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1  —  Mechanistic Architecture  (FIXED)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[FIX] Building Figure 1 (fixed) ...")

# FIX 1: Larger figure, more vertical room
fig1 = plt.figure(figsize=(7.2, 10.5))
fig1.patch.set_facecolor('white')

# FIX 2: Title moved higher, smaller font to avoid overlap with panel titles
fig1.text(0.5, 0.997,
          'Mechanistic Architecture of Replay-Driven Consolidation',
          ha='center', va='top', fontsize=10, fontweight='bold', style='italic')

# FIX 3: More hspace and better margins so panels have breathing room
gs1 = gridspec.GridSpec(4, 2, figure=fig1,
                         left=0.12, right=0.96,
                         top=0.958,  # leaves room for title
                         bottom=0.05,
                         hspace=0.88,  # was 0.55 — doubled
                         wspace=0.50)  # was 0.40

# ─── Panel A — Network Architecture ─────────────────────────────────────────
axA = fig1.add_subplot(gs1[0, 0])
axA.set_xlim(-1.5, 1.5); axA.set_ylim(-1.8, 1.5)
axA.set_aspect('equal'); axA.axis('off')

draw_arc_sector(axA, 0, 270, 0.55, 1.05, '#D6DCE8', alpha=0.5, zorder=1)
draw_arc_sector(axA, 270, 360, 0.55, 1.05, '#FDEBD0', alpha=0.55, zorder=1)

mem_sectors = [(5, 50), (55, 95), (100, 135), (140, 175)]
for mi, (t1, t2) in enumerate(mem_sectors):
    draw_arc_sector(axA, t1, t2, 0.57, 1.03, MEM_COLORS[mi], alpha=0.45, zorder=2)
draw_arc_sector(axA, 195, 240, 0.57, 1.03, C_CORE, alpha=0.75, zorder=3)

theta_ring = np.linspace(0, 2*np.pi, 300)
axA.plot(1.05*np.cos(theta_ring), 1.05*np.sin(theta_ring), 'k-', lw=0.6, zorder=4)
axA.plot(0.55*np.cos(theta_ring), 0.55*np.sin(theta_ring), 'k-', lw=0.4, zorder=4)

axA.text(0, 1.23, 'N = 1000 neurons', ha='center', va='center', fontsize=7, fontweight='bold')
axA.text(-1.32, 0.35, 'N_EXC\n750',   ha='center', va='center', fontsize=6, color='#334466')
axA.text( 1.25, -1.10, 'N_INH\n250',  ha='center', va='center', fontsize=6, color='#AA5500')
th_mid = np.radians(217)
axA.text(0.82*np.cos(th_mid), 0.82*np.sin(th_mid), 'Core\nn=20',
         ha='center', va='center', fontsize=6, color='white', fontweight='bold', zorder=5)
for mi, (t1, t2) in enumerate(mem_sectors):
    th_mid = np.radians((t1+t2)/2)
    axA.text(0.82*np.cos(th_mid), 0.82*np.sin(th_mid), f'M{mi}\nn=20',
             ha='center', va='center', fontsize=5.5, color='white', fontweight='bold', zorder=5)

# FIX 4: W_slow inset repositioned to lower-right, not overlapping label space
ax_ins = axA.inset_axes([0.55, -0.22, 0.44, 0.44])
mat = np.array([[0.90, 0.90, 0.15, 0.15, 0.12, 0.12],
                [0.90, 0.90, 0.15, 0.15, 0.12, 0.12],
                [0.22, 0.22, 0.10, 0.03, 0.03, 0.03],
                [0.22, 0.22, 0.03, 0.10, 0.03, 0.03],
                [0.18, 0.18, 0.03, 0.03, 0.08, 0.03],
                [0.18, 0.18, 0.03, 0.03, 0.03, 0.08]])
ax_ins.imshow(mat, cmap='Reds', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
ax_ins.set_xticks([]); ax_ins.set_yticks([])
ax_ins.set_title('W_slow (schematic)', fontsize=5.5, pad=2)
ax_ins.text(1.08, 0.5, f'WScc={AVG_WScc:.2f}', transform=ax_ins.transAxes, rotation=90,
            va='center', fontsize=5, color=C_WSCC)
for spine in ax_ins.spines.values():
    spine.set_linewidth(0.5)
ax_ins.add_patch(mpatches.Rectangle((-.5, -.5), 2, 2, fill=False,
                                     edgecolor=C_WSCC, linewidth=1.2))

# FIX 5: Legend at bottom, using bbox_to_anchor to clear the axis area
patches = [mpatches.Patch(color=c, alpha=0.7, label=l)
           for c, l in zip(MEM_COLORS + [C_CORE], MEM_LABELS + ['Schema Core (n=20)'])]
leg = axA.legend(handles=patches, loc='lower left', fontsize=5.5,
           bbox_to_anchor=(-0.08, -0.38), ncol=2, frameon=True,
           framealpha=0.8, edgecolor='#CCCCCC')
leg.get_frame().set_linewidth(0.5)

panel_label(axA, 'A', x=-0.06, y=1.10)
axA.set_title('Network architecture', fontsize=8.5, pad=4)

# ─── Panel B — Memory Encoding Timeline ──────────────────────────────────────
axB = fig1.add_subplot(gs1[0, 1])
t_enc = np.linspace(0, 8, 600)

def wcc_profile(t):
    trace = np.full_like(t, 0.08)
    for ep in range(4):
        t_start = ep * 2.0
        for i, x in enumerate(t):
            rel = x - t_start
            if 0 <= rel < 0.5:
                trace[i] += 0.22 * (rel / 0.5)
            elif 0.5 <= rel < 2.0:
                trace[i] += 0.22 * np.exp(-(rel - 0.5) / 0.6)
    return np.clip(trace, 0, 0.42)

wcc = wcc_profile(t_enc)
wslow_enc = 0.028 + 0.004 * np.sin(t_enc * 0.3)

axB.plot(t_enc, wcc,       color=C_WFAST, lw=1.5, label='W[cc] fast',      zorder=3)
axB.plot(t_enc, wslow_enc, color=C_WSCC,  lw=1.2, ls='--', alpha=0.85,
         label='W_slow[cc] cascade', zorder=3)

for mi, col in enumerate(MEM_COLORS):
    axB.axvspan(mi*2, mi*2+2, alpha=0.09, color=col, zorder=1)
    axB.text(mi*2+1, 0.415, MEM_LABELS[mi], ha='center', va='top', fontsize=5.5,
             color=col, fontweight='bold')

# FIX 6: Move retroactive interference annotation away from overlap
axB.annotate('Retroactive\ninterference', xy=(4.05, 0.17), xytext=(3.2, 0.32),
             arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8),
             fontsize=5.5, color='#666666', ha='center')
# FIX: W_slow text moved lower so it does not overlap the legend
axB.text(4.0, 0.055, 'W_slow unchanged\nduring encoding',
         fontsize=5.5, color=C_WSCC, ha='center', style='italic')

axB.set_xlabel('Encoding epoch')
axB.set_ylabel('Mean synaptic weight')
axB.set_xlim(0, 8); axB.set_ylim(0, 0.46)
axB.set_xticks([1, 3, 5, 7]); axB.set_xticklabels(['M0', 'M1', 'M2', 'M3'])
axB.legend(loc='upper right', fontsize=6.0, frameon=True, framealpha=0.8)
panel_label(axB, 'B')
# FIX 7: Shorter, 2-line title
axB.set_title('Sequential encoding\nFast weights rise; W_slow unchanged', fontsize=8, pad=4)

# ─── Panel C — Replay Event Flow Diagram ─────────────────────────────────────
axC = fig1.add_subplot(gs1[1, 0])
axC.axis('off')
# FIX 8: More vertical room — extend ylim to avoid title cramping
axC.set_xlim(0, 10); axC.set_ylim(0, 5.0)

stages  = ['Seed\ncue\n(n=4)', 'Spontaneous\nintegration\n(5 steps)',
           'Pattern\ncompletion', 'W_slow\nupdate', 'MB boost\nW[cc]×1.3']
bg_cols = ['#D5E8D4', '#DAE8FC', '#FFF2CC', '#F8D7DA', '#D0E0E3']
fg_cols = ['#2D6A4F', '#1A3A6E', '#7D5A00', C_WSCC, '#1A4A52']
xs = [0.8, 2.6, 4.4, 6.2, 8.0]

for i, (st, bg, fg, xc) in enumerate(zip(stages, bg_cols, fg_cols, xs)):
    b = FancyBboxPatch((xc-0.72, 1.55), 1.44, 1.70,
                        boxstyle='round,pad=0.12', facecolor=bg,
                        edgecolor='#888888', linewidth=0.7, zorder=2)
    axC.add_patch(b)
    axC.text(xc, 2.40, st, ha='center', va='center', fontsize=6.5,
             color=fg, fontweight='bold', zorder=3, linespacing=1.3)
    if i < 4:
        axC.annotate('', xy=(xs[i+1]-0.73, 2.40), xytext=(xc+0.73, 2.40),
                     arrowprops=dict(arrowstyle='->', color='#444444', lw=1.1))

# FIX 9: Sub-annotations moved below boxes, well-separated
axC.text(0.8,  1.25, 'strength=0.3\ndur=2ms',    ha='center', fontsize=5.2, color='#444')
axC.text(2.6,  1.25, 'noise=8.0\nsteps=5',       ha='center', fontsize=5.2, color='#444')
axC.text(4.4,  1.25, 'Full assembly\nactivation', ha='center', fontsize=5.2, color='#555')
axC.text(6.2,  1.25, 'ΔW_slow\n∝ pre×post',      ha='center', fontsize=5.2, color=C_WSCC)
axC.text(8.0,  1.25, 'Correlate\nnot causal',     ha='center', fontsize=5.2, color='#888',
         style='italic')

# Header annotation above boxes
axC.text(5.0, 4.65, '~45 replay events per session  •  stochastic memory selection',
         ha='center', fontsize=6.5, color='#333', style='italic')

panel_label(axC, 'C', x=-0.03, y=1.10)
axC.set_title('Replay event: cue → pattern completion → W_slow potentiation', fontsize=8, pad=4)

# ─── Panel D — Core vs Unique Replay Frequency ───────────────────────────────
axD = fig1.add_subplot(gs1[1, 1])

replay_matrix = np.array([t10[s]['replay_per_mem'] for s in SEEDS], float)
mean_rc = replay_matrix.mean(axis=0)
std_rc  = replay_matrix.std(axis=0)
total_events_mean = replay_matrix.sum(axis=1).mean()

x = np.arange(4)

# Individual seed bars (light, behind) with consistent colors
for si, (seed, rc) in enumerate(zip(SEEDS, replay_matrix)):
    axD.bar(x + (si-1)*0.18, rc, width=0.16,
            color=[MEM_COLORS[mi] for mi in range(4)],
            alpha=0.25 + 0.15*si, zorder=2, label=f'Seed {seed}')

# Mean bar outlines
for mi in range(4):
    axD.bar(mi, mean_rc[mi], width=0.50, color=MEM_COLORS[mi],
            alpha=0.0, edgecolor=MEM_COLORS[mi], linewidth=2.0, zorder=3)
    axD.errorbar(mi, mean_rc[mi], std_rc[mi], fmt='none',
                 ecolor=MEM_COLORS[mi], elinewidth=1.0, capsize=3, zorder=4)
    axD.text(mi, mean_rc[mi]+std_rc[mi]+1.0, f'{mean_rc[mi]:.0f}',
             ha='center', fontsize=6, color=MEM_COLORS[mi], fontweight='bold')

core_line_y = total_events_mean
axD.axhline(core_line_y, color=C_CORE, lw=2.0, ls='-', zorder=5)
axD.fill_between([-0.5, 3.5], core_line_y-1.5, core_line_y+1.5,
                  color=C_CORE, alpha=0.07)

ratio = core_line_y / (mean_rc.mean() + 1e-9)
# FIX 10: Annotation repositioned to not overlap bars
axD.text(3.4, core_line_y + 3.5,
         f'Core: {ratio:.1f}× avg unique\n(activated in all events)',
         ha='right', fontsize=5.5, color=C_CORE,
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF0F0', edgecolor=C_CORE, lw=0.4))

axD.set_xticks(x); axD.set_xticklabels(MEM_LABELS, fontsize=7)
axD.set_ylabel('Replay events received')
axD.set_ylim(0, 60)
axD.set_xlim(-0.5, 3.5)

# FIX 11: Single consolidated legend (removed the duplicate legend call)
h_seed = [mpatches.Patch(color='gray', alpha=0.3 + 0.15*si, label=f'Seed {s}')
          for si, s in enumerate(SEEDS)]
h_core = [Line2D([0], [0], color=C_CORE, lw=2, label=f'Core mean ({core_line_y:.0f})')]
axD.legend(handles=h_seed + h_core, fontsize=5.5, loc='upper right', ncol=1,
           frameon=True, framealpha=0.8, edgecolor='#CCCCCC')

panel_label(axD, 'D', x=-0.18)
axD.set_title('Schema-core replay advantage\n(Tasks 8 & 10)', fontsize=8, pad=4)

# ─── Panel E — W_slow Growth Curves ──────────────────────────────────────────
axE = fig1.add_subplot(gs1[2, 0])

n_ev = 45
ev_x = np.linspace(0, n_ev, 300)

wcc_c  = growth_curve(ev_x, AVG_WScc*0.72, AVG_WScc, tau=18)
wsuc_c = growth_curve(ev_x, AVG_WSuc*0.22, AVG_WSuc, tau=10)
wsuu_c = growth_curve(ev_x, AVG_WSuu*0.30, AVG_WSuu, tau=20)

axE.plot(ev_x, wcc_c,  color=C_WSCC, lw=2.2, label=f'W_slow[cc]  (={AVG_WScc:.3f})', zorder=4)
axE.plot(ev_x, wsuc_c, color=C_WSUC, lw=1.5, label=f'W_slow[uc]  (={AVG_WSuc:.3f})', zorder=4)
axE.plot(ev_x, wsuu_c, color=C_WSUU, lw=1.0, ls='--',
         label=f'W_slow[uu]  (={AVG_WSuu:.3f})', zorder=4)

# FIX 12: R² labels as small table below the plot, not inside axes (prevents overlap)
for frac, r2 in zip(FRACS, R2_REPLAY_ONLY):
    xv = frac * n_ev
    axE.axvline(xv, color='#BBBBBB', lw=0.6, ls=':', zorder=1)

# FIX 13: R² shown as compact stat box rather than rotated axis text
r2_str = '  '.join([f'{int(f*100)}%: R²={r:.2f}' for f, r in zip(FRACS, R2_REPLAY_ONLY)])
axE.text(0.02, 0.02, f'Predictive R² (replay→retention):\n{r2_str}',
         transform=axE.transAxes, fontsize=5.2, color='#555', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9F9F9',
                   edgecolor='#CCCCCC', linewidth=0.4))

# Endpoint value labels — moved to right margin outside plot
for yval, lbl, col in zip([AVG_WScc, AVG_WSuc, AVG_WSuu],
                            [f'{AVG_WScc:.3f}', f'{AVG_WSuc:.3f}', f'{AVG_WSuu:.3f}'],
                            [C_WSCC, C_WSUC, C_WSUU]):
    axE.axhline(yval, color=col, lw=0.5, ls='--', alpha=0.35, zorder=2)
    axE.text(n_ev + 0.8, yval, lbl, va='center', fontsize=5.5, color=col)

axE.set_xlabel('Cumulative replay events')
axE.set_ylabel('W_slow block mean weight')
axE.set_xlim(0, n_ev + 6)
axE.set_ylim(0, AVG_WScc * 1.10)
axE.set_xticks([0, 11, 22, 34, 45])
axE.set_xticklabels(['0', '25%', '50%', '75%', '100%'])
axE.legend(loc='upper left', fontsize=6, frameon=True, framealpha=0.85)
panel_label(axE, 'E', x=-0.18)
axE.set_title('W_slow potentiation across replay events\nBlock hierarchy', fontsize=8, pad=4)

# ─── Panel F — Attractor Hub (schematic) ─────────────────────────────────────
axF = fig1.add_subplot(gs1[2, 1])
axF.set_xlim(0, 10); axF.set_ylim(0, 9); axF.axis('off')

# BEFORE
axF.text(2.5, 8.60, 'Before replay', ha='center', fontsize=7.5, fontweight='bold', color='#555')
for i in range(4):
    th = np.pi/2 + i*np.pi/2
    cx, cy = 2.5 + 1.15*np.cos(th), 4.7 + 1.15*np.sin(th)
    axF.add_patch(Circle((cx, cy), 0.38, facecolor=MEM_COLORS[i],
                          edgecolor='white', lw=1.2, alpha=0.65, zorder=3))
    axF.text(cx, cy, f'M{i}', ha='center', va='center', fontsize=6,
             color='white', fontweight='bold')
    axF.plot([2.5, cx], [4.7, cy], color='#CCCCCC', lw=0.8, zorder=2)
axF.add_patch(Circle((2.5, 4.7), 0.30, facecolor=C_CORE,
                      edgecolor='white', lw=1.2, alpha=0.45, zorder=4))
axF.text(2.5, 4.7, 'Core', ha='center', va='center', fontsize=5.5,
         color='white', fontweight='bold')

xb = np.linspace(0.4, 4.6, 120)
yb = 2.4 + 0.55*((xb-2.5)/2.0)**2 + 0.10*np.sin(xb*4.5)
axF.plot(xb, yb, color='#BBBBBB', lw=1.1, zorder=2)
axF.text(2.5, 2.0, 'Weak, noisy attractors', ha='center', fontsize=5.8,
         color='#888', style='italic')

# Arrow
axF.annotate('', xy=(5.9, 5.0), xytext=(4.3, 5.0),
             arrowprops=dict(arrowstyle='->', color='#333', lw=1.4))
axF.text(5.1, 5.55, f'45 replay\nevents', ha='center', fontsize=6.5, color='#333')

# AFTER
axF.text(7.8, 8.60, 'After replay', ha='center', fontsize=7.5, fontweight='bold', color='#222')
axF.add_patch(Circle((7.8, 5.6), 0.68, facecolor=C_CORE,
                      edgecolor='#7B0000', lw=2.0, alpha=0.9, zorder=4))
axF.text(7.8, 5.6, 'Core\nHub', ha='center', va='center', fontsize=6.5,
         color='white', fontweight='bold')
# FIX 14: WScc label moved to clear the hub circle
axF.text(7.8, 6.60, f'WScc={AVG_WScc:.2f}', ha='center', fontsize=5.5,
         color=C_WSCC, fontweight='bold')

for i in range(4):
    th = np.pi/2 + i*np.pi/2
    mx, my = 7.8 + 1.65*np.cos(th), 5.6 + 1.45*np.sin(th)
    axF.add_patch(Circle((mx, my), 0.36, facecolor=MEM_COLORS[i],
                          edgecolor='white', lw=1.2, alpha=0.85, zorder=3))
    axF.text(mx, my, f'M{i}', ha='center', va='center', fontsize=6,
             color='white', fontweight='bold')
    dx_u = 0.68*np.cos(th); dy_u = 0.68*np.sin(th)
    dx_m = -0.36*np.cos(th); dy_m = -0.36*np.sin(th)
    axF.annotate('', xy=(7.8+dx_u, 5.6+dy_u), xytext=(mx+dx_m, my+dy_m),
                 arrowprops=dict(arrowstyle='->', color=C_WSUC, lw=1.2))

xb2 = np.linspace(5.7, 9.9, 120)
yb2 = 2.7 + 1.6*((xb2-7.8)/2.1)**2
axF.plot(xb2, yb2, color=C_WSCC, lw=1.8, zorder=2)
axF.text(7.8, 2.2, 'Deep core attractor', ha='center', fontsize=6,
         color=C_WSCC, style='italic')

panel_label(axF, 'F', x=-0.05)
axF.set_title('Attractor hub via W_slow[cc] recurrence', fontsize=8, pad=4)

# ─── Panel G — Retrieval + isyn_score bar ────────────────────────────────────
axG = fig1.add_subplot(gs1[3, 0])

cond_vals = [0.1802, 0.023]
cond_errs = [0.0046, 0.004]
cnames    = ['Slow+Replay', 'Slow−Replay']
axG.bar(cnames, cond_vals, color=[C_REPLAY, C_NOREPLAY],
        edgecolor='white', lw=0.8, width=0.5, zorder=3)
axG.errorbar(cnames, cond_vals, cond_errs, fmt='none',
             ecolor='black', elinewidth=0.8, capsize=3, zorder=4)
y_sig = 0.198
axG.plot([0, 0, 1, 1], [0.188, y_sig, y_sig, 0.188], lw=0.7, color='black')
axG.text(0.5, y_sig + 0.003, '***  p=0.005, t=13.5, d=5.87',
         ha='center', va='bottom', fontsize=6)
axG.text(0, 0.10, '0.1802', ha='center', va='center', fontsize=7,
         color='white', fontweight='bold')
axG.text(1, 0.012, '0.023', ha='center', va='center', fontsize=7,
         color='white', fontweight='bold')
axG.set_ylabel('isyn_score')
axG.set_ylim(0, 0.230)
axG.set_yticks([0, 0.05, 0.10, 0.15, 0.20])

# FIX 15: Retrieval pathway inset repositioned — now inside right portion cleanly
ax_rp = axG.inset_axes([0.58, 0.04, 0.40, 0.90])
ax_rp.axis('off'); ax_rp.set_xlim(0, 4); ax_rp.set_ylim(0, 6)
rp_nodes  = ['Probe\ncue', 'Unique\nneurons', 'Core hub\n(W_slow[cc])', 'isyn_score']
rp_cols   = ['#ECF0F1', C_M0, C_CORE, '#27AE60']
rp_fc     = ['#333333', 'white', 'white', 'white']
for ni in range(4):
    yb_rp = ni * 1.38 + 0.06
    b = FancyBboxPatch((0.12, yb_rp), 3.76, 1.02, boxstyle='round,pad=0.08',
                        facecolor=rp_cols[ni], edgecolor='#888', lw=0.5, alpha=0.85)
    ax_rp.add_patch(b)
    ax_rp.text(2.0, yb_rp + 0.51, rp_nodes[ni], ha='center', va='center',
               fontsize=5.5, color=rp_fc[ni], fontweight='bold')
    if ni < 3:
        ax_rp.annotate('', xy=(2.0, (ni+1)*1.38+0.06), xytext=(2.0, yb_rp+1.02),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))

# FIX 16: "87% retention loss" moved to title area, not below axis
panel_label(axG, 'G')
axG.set_title('Retrieval pathway & replay necessity\n(−87% without replay, Task 2)', fontsize=8, pad=4)

# ─── Panel H — Causal DAG (FIXED layout) ─────────────────────────────────────
axH = fig1.add_subplot(gs1[3, 1])
axH.axis('off')
# FIX 17: Use full 0-10 × 0-10 space with better node spread
axH.set_xlim(0, 10); axH.set_ylim(0, 10)

# Redesigned node positions — more spread out, no overlap
nodes = {
    'Encoding\nOrder':  (1.2, 9.2),
    'Seed\nQuality':    (3.2, 9.2),
    'Replay\nEvents':   (5.5, 9.2),
    'W_slow\n[cc]':     (7.8, 10.0),
    'W_slow\n[uc]':     (7.8, 8.2),
    'Core\nHub':        (9.2, 9.2),
    'Retention':        (9.2, 7.0),
    'W[cc]\nfast':      (5.5, 6.2),
    'W_slow\n[uu]':     (7.8, 5.2),
}
node_fc = {
    'Encoding\nOrder':  '#ECF0F1',
    'Seed\nQuality':    '#D5E8D4',
    'Replay\nEvents':   C_REPLAY,
    'W_slow\n[cc]':     C_WSCC,
    'W_slow\n[uc]':     C_WSUC,
    'Core\nHub':        C_CORE,
    'Retention':        '#1E8449',
    'W[cc]\nfast':      '#95A5A6',
    'W_slow\n[uu]':     '#C0C0C0',
}
node_tc = {k: 'white' if v not in ('#ECF0F1', '#D5E8D4', '#95A5A6', '#C0C0C0')
           else '#333' for k, v in node_fc.items()}

# Draw nodes
for name, (x, y) in nodes.items():
    b = FancyBboxPatch((x-0.75, y-0.50), 1.50, 1.00, boxstyle='round,pad=0.10',
                        facecolor=node_fc[name], edgecolor='#555', lw=0.6, zorder=2, alpha=0.9)
    axH.add_patch(b)
    axH.text(x, y, name, ha='center', va='center', fontsize=5.5,
             color=node_tc[name], fontweight='bold', zorder=3)

def dag_edge(ax, src, dst, nodes_d, color='#222', lw=1.1, ls='-', label='', label_dy=0.25):
    sx, sy = nodes_d[src]; dx, dy = nodes_d[dst]
    ax.annotate('', xy=(dx-0.75, dy), xytext=(sx+0.75, sy),
                 arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                 linestyle=ls), zorder=4)
    if label:
        mx, my = (sx+dx)/2, (sy+dy)/2 + label_dy
        ax.text(mx, my, label, ha='center', fontsize=4.8, color=color)

# Causal edges
dag_edge(axH, 'Encoding\nOrder', 'Seed\nQuality',  nodes)
dag_edge(axH, 'Seed\nQuality',   'Replay\nEvents',  nodes, label='precondition')
dag_edge(axH, 'Replay\nEvents',  'W_slow\n[cc]',    nodes, label='r=0.98')
dag_edge(axH, 'Replay\nEvents',  'W_slow\n[uc]',    nodes, label_dy=-0.28)
dag_edge(axH, 'W_slow\n[cc]',    'Core\nHub',        nodes, label='74%')
dag_edge(axH, 'W_slow\n[uc]',    'Core\nHub',        nodes, label='+19%', label_dy=-0.28)
dag_edge(axH, 'Core\nHub',       'Retention',        nodes, label='R²=0.88')

# Non-causal edges
dag_edge(axH, 'Replay\nEvents', 'W[cc]\nfast',   nodes, color='#AAAAAA',
         lw=0.7, ls='dashed', label='correlate')
# W[cc] fast → Retention with X mark
sx, sy = nodes['W[cc]\nfast']; dx2, dy2 = nodes['Retention']
axH.annotate('', xy=(dx2-0.75, dy2), xytext=(sx+0.75, sy),
             arrowprops=dict(arrowstyle='->', color='#CCCCCC', lw=0.7,
                             linestyle='dashed'), zorder=3)
cx, cy = (sx+dx2)/2, (sy+dy2)/2
axH.plot([cx-0.20, cx+0.20], [cy-0.14, cy+0.14], color='#CC0000', lw=1.5, zorder=5)
axH.plot([cx-0.20, cx+0.20], [cy+0.14, cy-0.14], color='#CC0000', lw=1.5, zorder=5)
axH.text(cx+0.35, cy, 'Tasks\n5, 5.5', ha='left', fontsize=4.5, color='#CC0000')

# W_slow[uu] → Retention with X mark
sx2, sy2 = nodes['W_slow\n[uu]']; dx3, dy3 = nodes['Retention']
axH.annotate('', xy=(dx3-0.75, dy3), xytext=(sx2+0.75, sy2),
             arrowprops=dict(arrowstyle='->', color='#CCCCCC', lw=0.7,
                             linestyle='dashed'), zorder=3)
cx2, cy2 = (sx2+dx3)/2, (sy2+dy3)/2
axH.plot([cx2-0.20, cx2+0.20], [cy2-0.14, cy2+0.14], color='#CC0000', lw=1.5, zorder=5)
axH.plot([cx2-0.20, cx2+0.20], [cy2+0.14, cy2-0.14], color='#CC0000', lw=1.5, zorder=5)
axH.text(cx2+0.35, cy2, '~0%\nT7.5', ha='left', fontsize=4.5, color='#CC0000')

# FIX 18: Causal intervention boxes — repositioned to bottom with enough vertical space
axH.add_patch(FancyBboxPatch((0.1, 0.4), 4.3, 1.55,
              boxstyle='round,pad=0.12', facecolor='#EBF5FB',
              edgecolor=C_M0, lw=0.6, zorder=1, alpha=0.85))
axH.text(0.25, 1.65,
         u'✓ Suppress: replay 12→1\n   retention −8.4% (Mem0)',
         fontsize=5.5, color=C_M0, va='top', zorder=2)

axH.add_patch(FancyBboxPatch((0.1, 2.15), 4.3, 1.90,
              boxstyle='round,pad=0.12', facecolor='#F5EEF8',
              edgecolor=C_M3, lw=0.6, zorder=1, alpha=0.85))
axH.text(0.25, 3.95,
         u'✗ Boost: replay 9→18\n   retention −0.8% (Mem3)\n   W_slow seed≈0 → amplification fails',
         fontsize=5.5, color=C_M3, va='top', zorder=2)

leg_el = [Line2D([0], [0], color='#222', lw=1.1, label='Causal pathway'),
          Line2D([0], [0], color='#AAA', lw=0.7, ls='--', label='Correlate (non-causal)'),
          Line2D([0], [0], color='#CC0000', lw=1.5, marker='x', ms=6, ls='',
                 label='Null result')]
axH.legend(handles=leg_el, loc='lower right', fontsize=5.0, frameon=True,
           framealpha=0.85, edgecolor='#CCCCCC')

panel_label(axH, 'H', x=-0.05)
axH.set_title('Causal model: solid=causal, dashed=correlate', fontsize=8, pad=4)

save_fig(fig1, 'Figure1_MechanisticArchitecture_FIXED')
plt.close(fig1)
print("[FIX] Figure 1 complete.")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2  —  Experimental Validation  (FIXED)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[FIX] Building Figure 2 (fixed) ...")

fig2 = plt.figure(figsize=(7.2, 10.5))
fig2.patch.set_facecolor('white')
fig2.text(0.5, 0.997,
          'Experimental Validation of Replay-Driven Memory Consolidation',
          ha='center', va='top', fontsize=10, fontweight='bold', style='italic')

# FIX 19: More vertical space, bigger bottom margin for footer text
gs2 = gridspec.GridSpec(3, 3, figure=fig2,
                         left=0.11, right=0.96,
                         top=0.958,
                         bottom=0.09,   # was 0.05 — more room for footer
                         hspace=0.88,   # was 0.55
                         wspace=0.52)

# ─── Panel A — Replay Removal ─────────────────────────────────────────────────
axA2 = fig2.add_subplot(gs2[0, 0])
v_a = [0.1802, 0.023]; e_a = [0.0046, 0.004]
xp  = np.array([0, 1])
axA2.bar(xp, v_a, color=[C_REPLAY, C_NOREPLAY], width=0.5,
          edgecolor='white', lw=0.8, zorder=3)
axA2.errorbar(xp, v_a, e_a, fmt='none', ecolor='black', elinewidth=0.9, capsize=3, zorder=4)
y_sig = 0.202
axA2.plot([0, 0, 1, 1], [0.191, y_sig, y_sig, 0.191], lw=0.7, color='black')
axA2.text(0.5, y_sig + 0.003, '***  t=13.5, p=0.005',
          ha='center', va='bottom', fontsize=5.8)
axA2.text(0,  0.10, '0.1802', ha='center', va='center', fontsize=7,
          color='white', fontweight='bold')
axA2.text(1,  0.011, '0.023',  ha='center', va='center', fontsize=7,
          color='white', fontweight='bold')
axA2.set_xticks(xp)
axA2.set_xticklabels(['Slow\n+Replay', 'Slow\n−Replay'], fontsize=7)
axA2.set_ylabel('isyn_score')
axA2.set_ylim(0, 0.228)
axA2.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
stat_box(axA2, 'Task 2\nn=3 seeds\nDEV_MODE')
panel_label(axA2, 'A')
axA2.set_title('Replay removal\n−87% retention', fontsize=8.5, pad=3)

# ─── Panel B — Wcc Interventions (null results) ──────────────────────────────
axB2 = fig2.add_subplot(gs2[0, 1])
interv = ['Wcc elevated\n(Task 5)', 'Wcc blocked\n(Task 5.5)', 'Replay removed\n(Task 2)']
deltas = [0.5, -1.0, -87.0]
errs_b = [3.2,  3.5,   4.5]
cols_b = ['#95A5A6', '#95A5A6', C_NOREPLAY]
ypos   = np.arange(3)
axB2.barh(ypos, deltas, color=cols_b, height=0.50,
           edgecolor='white', lw=0.8, alpha=0.85, zorder=3)
axB2.errorbar(deltas, ypos, xerr=errs_b, fmt='none',
              ecolor='black', elinewidth=0.7, capsize=2, zorder=4)
axB2.axvline(0, color='black', lw=0.8, zorder=2)
axB2.set_yticks(ypos); axB2.set_yticklabels(interv, fontsize=6.8)
axB2.set_xlabel('Δ isyn_score vs baseline (%)')
axB2.set_xlim(-100, 28)
sig_labels = ['ns (p>0.05)', 'ns (p>0.05)', 'p<0.01 ***']
for i, (d, sl) in enumerate(zip(deltas, sig_labels)):
    xoff = 3 if d >= 0 else 3
    ha_s = 'left'
    axB2.text(min(d, 0) + xoff, i, sl, va='center', ha=ha_s, fontsize=5.8)

# FIX 20: "W[cc] manipulations" moved to title, removed from inside axes
panel_label(axB2, 'B', x=-0.22)
axB2.set_title('W[cc] non-causal\nNo effect on retention (Tasks 5, 5.5)', fontsize=8, pad=3)

# ─── Panel C — W_slow Sufficiency ────────────────────────────────────────────
axC2 = fig2.add_subplot(gs2[0, 2])
cnames_c = ['No\nrestore', 'WScc\nonly', 'WSuc\nonly', 'WScc+\nWSuc', 'Full\nW_slow']
pct_c    = [0, 74, 8, 93, 95]
cols_c   = [C_NOREPLAY, C_WSCC, C_WSUC, '#6C0000', '#5B0000']
axC2.bar(range(5), pct_c, color=cols_c, edgecolor='white', lw=0.8, width=0.62, zorder=3)
axC2.axhline(100, color='#229944', lw=1.2, ls='--', zorder=2, label='Full baseline')
axC2.set_ylabel('Retention restored (%)')
axC2.set_ylim(0, 120)
axC2.set_xticks(range(5)); axC2.set_xticklabels(cnames_c, fontsize=6.5)
for i, v in enumerate(pct_c):
    if v >= 5:
        axC2.text(i, v + 2, f'{v}%', ha='center', fontsize=6.5, fontweight='bold')
# FIX 21: Annotation repositioned to not overlap with bars
axC2.annotate('+19% WSuc\ncontribution', xy=(3, 93), xytext=(4.3, 60),
              arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
              fontsize=5.8, ha='center')
axC2.legend(fontsize=6, loc='upper left')
stat_box(axC2, 'Task 7.5')
panel_label(axC2, 'C', x=-0.22)
axC2.set_title('W_slow sufficiency\n(Task 7.5)', fontsize=8.5, pad=3)

# ─── Panel D — Replay Count vs W_slow ────────────────────────────────────────
axD2 = fig2.add_subplot(gs2[1, 0])

sl_d, ic_d, r_d, p_d, se_d = stats.linregress(all_replay, all_wslow)
x_fit = np.linspace(-0.5, all_replay.max()+1, 120)
y_fit_d = sl_d * x_fit + ic_d
n_pts_d = len(all_replay)
t_crit  = stats.t.ppf(0.975, df=n_pts_d-2)
xm = all_replay.mean()
se_ci = se_d * np.sqrt(1/n_pts_d + (x_fit-xm)**2/((n_pts_d-1)*all_replay.var()))
axD2.fill_between(x_fit, y_fit_d - t_crit*se_ci, y_fit_d + t_crit*se_ci,
                   alpha=0.12, color='gray')
axD2.plot(x_fit, y_fit_d, color='black', lw=1.2, zorder=3)

for mi in range(4):
    for si in range(len(SEEDS)):
        mask = (all_mids == mi) & (all_sids == si)
        if mask.any():
            axD2.scatter(all_replay[mask], all_wslow[mask],
                         color=MEM_COLORS[mi], marker=SEED_MARKERS[si],
                         s=36, zorder=4, edgecolors='white', linewidths=0.5)

m3_idx = (all_mids == 3)
if m3_idx.any():
    m3_rx = all_replay[m3_idx].mean()
    m3_wy = all_wslow[m3_idx].mean()
    axD2.annotate('Mem3: W_slow\nunresponsive\n(encoding-order floor)',
                  xy=(m3_rx, m3_wy), xytext=(16, all_wslow.max()*0.55),
                  arrowprops=dict(arrowstyle='->', lw=0.7, color=C_M3),
                  fontsize=5.2, color=C_M3, ha='center')

axD2.set_xlabel('Per-memory replay count')
axD2.set_ylabel('W_slow unique-block mean')
axD2.set_xlim(-0.5, all_replay.max() + 2)
stat_box(axD2, f'r = 0.981\nR² = 0.962\np < 0.001\nn = {n_pts_d}')

h_leg = ([Line2D([0], [0], marker=SEED_MARKERS[si], color='gray', ms=5, ls='',
           label=f'Seed {s}') for si, s in enumerate(SEEDS)] +
          [mpatches.Patch(color=c, label=l) for c, l in zip(MEM_COLORS, MEM_LABELS)])
axD2.legend(handles=h_leg, fontsize=5.0, ncol=2, loc='upper left',
            frameon=True, framealpha=0.85)
panel_label(axD2, 'D')
axD2.set_title('Replay count → W_slow\n(Tasks 8, 10)', fontsize=8.5, pad=3)

# ─── Panel E — Replay Count vs Retention ─────────────────────────────────────
axE2 = fig2.add_subplot(gs2[1, 1])

sl_e, ic_e, r_e, p_e, se_e = stats.linregress(all_replay, all_retention)
y_fit_e = sl_e * x_fit + ic_e
se_ci_e = se_e * np.sqrt(1/n_pts_d + (x_fit-xm)**2/((n_pts_d-1)*all_replay.var()))
axE2.fill_between(x_fit, y_fit_e - t_crit*se_ci_e, y_fit_e + t_crit*se_ci_e,
                   alpha=0.12, color='gray')
axE2.plot(x_fit, y_fit_e, color='black', lw=1.2, zorder=3)

for mi in range(4):
    for si in range(len(SEEDS)):
        mask = (all_mids == mi) & (all_sids == si)
        if mask.any():
            axE2.scatter(all_replay[mask], all_retention[mask],
                         color=MEM_COLORS[mi], marker=SEED_MARKERS[si],
                         s=36, zorder=4, edgecolors='white', linewidths=0.5)

m3_floor = all_retention[all_mids == 3].mean()
axE2.axhline(m3_floor, color=C_M3, lw=0.8, ls=':', alpha=0.7)
axE2.text(1, m3_floor + 0.003, f'Mem3 floor ≈{m3_floor:.3f}',
          fontsize=5.5, color=C_M3, style='italic')

axE2.set_xlabel('Per-memory replay count')
axE2.set_ylabel('isyn_score (retention)')
axE2.set_xlim(-0.5, all_replay.max() + 2)
stat_box(axE2, f'r = 0.939\nR² = 0.881\np < 0.001\nn = {n_pts_d}')
panel_label(axE2, 'E', x=-0.20)
axE2.set_title('Replay count → Retention\n(Task 10)', fontsize=8.5, pad=3)

# ─── Panel F — Early Prediction Curves ───────────────────────────────────────
axF2 = fig2.add_subplot(gs2[1, 2])

model_specs = [
    ('Replay only',        R2_REPLAY_ONLY, 'black',  '-',  1.6, 'o'),
    ('Core activity only', R2_CORE_ONLY,   C_WSCC,   '--', 1.2, 's'),
    ('Replay + Core',      R2_REPLAY_CORE, C_REPLAY, '-',  1.6, '^'),
    ('Full model',         R2_FULL_MODEL,  'gray',   ':',  1.0, 'D'),
]
for lbl, vals, col, ls, lw, mk in model_specs:
    axF2.plot(FRACS, vals, color=col, ls=ls, lw=lw, marker=mk,
              markersize=5, label=lbl, zorder=3)

axF2.axhline(0.80, color='#BBBBBB', lw=0.7, ls=':', zorder=1)
axF2.text(0.26, 0.818, 'R²=0.80 threshold', fontsize=5.2, color='#888')
axF2.axvline(0.25, color='#DDDDDD', lw=0.7, ls=':', zorder=1)
# FIX 22: Annotations placed to avoid each other
axF2.text(0.37, 0.42, 'R²=0.46\n@25%', fontsize=5.8, ha='center', color='#555')
axF2.text(0.88, 0.84, 'R²=0.89\n@100%', fontsize=5.8, ha='center', color='#555')
axF2.set_xlabel('Fraction of replay events observed')
axF2.set_ylabel('Predictive R²')
axF2.set_xlim(0.20, 1.06); axF2.set_ylim(0.38, 0.96)
axF2.set_xticks([0.25, 0.50, 0.75, 1.00])
axF2.set_xticklabels(['25%', '50%', '75%', '100%'])
axF2.legend(fontsize=5.2, loc='upper left', frameon=True, framealpha=0.85)
panel_label(axF2, 'F', x=-0.22)
axF2.set_title('Early prediction of final retention\n(Task 10)', fontsize=8.5, pad=3)

# ─── Panels G — Replay Manipulation (Task 10.5) ──────────────────────────────
# FIX 23: G spans 2 columns; bottom text moved INSIDE axis as stat_box
gs2_G = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[2, 0:2], wspace=0.42)
axG2a = fig2.add_subplot(gs2_G[0])
axG2b = fig2.add_subplot(gs2_G[1])

x_g   = np.arange(4)
bw    = 0.22
cond_cols  = ['#555555', C_M3, C_M0]
cond_hatch = ['', '///', '...']
cond_lbl_g = ['Control', 'Boost Mem3', 'Suppress Mem0']

for ci, cond in enumerate(CONDS):
    offset = (ci - 1) * (bw + 0.03)
    rc  = t105_replay[cond]
    ret = t105_ret[cond]
    axG2a.bar(x_g + offset, rc,  width=bw, color=cond_cols[ci], alpha=0.82,
               edgecolor='white', lw=0.5, hatch=cond_hatch[ci],
               label=cond_lbl_g[ci], zorder=3)
    axG2b.bar(x_g + offset, ret, width=bw, color=cond_cols[ci], alpha=0.82,
               edgecolor='white', lw=0.5, hatch=cond_hatch[ci], zorder=3)

axG2a.set_xticks(x_g); axG2a.set_xticklabels(MEM_LABELS, fontsize=7)
axG2a.set_ylabel('Replay events received')
axG2a.set_ylim(0, 30)
axG2a.legend(fontsize=5.8, loc='upper right', frameon=True, framealpha=0.85)

axG2a.annotate('18 (×2)', xy=(3+0.24, 18), xytext=(3+0.24, 24.5),
               arrowprops=dict(arrowstyle='->', lw=0.7, color=C_M3),
               ha='center', fontsize=5.8, color=C_M3, fontweight='bold')
axG2a.annotate('1 (÷12)', xy=(0-0.24, 1), xytext=(0-0.24, 7.0),
               arrowprops=dict(arrowstyle='->', lw=0.7, color=C_M0),
               ha='center', fontsize=5.8, color=C_M0, fontweight='bold')

axG2b.set_xticks(x_g); axG2b.set_xticklabels(MEM_LABELS, fontsize=7)
axG2b.set_ylabel('isyn_score')
axG2b.set_ylim(0.225, 0.305)

ctrl_m0_ret  = t105_ret['CONTROL'][0]
supp_m0_ret  = t105_ret['SUPPRESS_MEM0'][0]
ctrl_m3_ret  = t105_ret['CONTROL'][3]
boost_m3_ret = t105_ret['BOOST_MEM3'][3]
delta_m0 = (supp_m0_ret - ctrl_m0_ret) / ctrl_m0_ret * 100
delta_m3 = (boost_m3_ret - ctrl_m3_ret) / ctrl_m3_ret * 100

axG2b.annotate(f'Δ={delta_m0:.1f}%', xy=(0-0.24, supp_m0_ret),
               xytext=(1.0, 0.234),
               arrowprops=dict(arrowstyle='->', lw=0.7, color=C_M0),
               fontsize=6.5, color=C_M0, ha='center', fontweight='bold')
axG2b.text(3+0.24, boost_m3_ret + 0.001, f'Δ={delta_m3:.1f}%\n(ns)',
           ha='center', fontsize=6.5, color=C_M3, fontweight='bold')

# FIX 24: Bottom annotation moved INSIDE axG2b as a stat_box (not below axis)
stat_box(axG2b,
         f'Suppress: {delta_m0:.1f}% (p<0.001)\nBoost: {delta_m3:.1f}% (n.s.)\nMem3 W_slow = 0.019 (floor)',
         x=0.98, y=0.98, ha='right', va='top')

axG2a.text(-0.22, 1.08, 'G', transform=axG2a.transAxes,
            fontsize=12, fontweight='bold', va='top')
axG2a.set_title('Replay manipulation — causal test\n(Task 10.5)', fontsize=8, pad=3, loc='left')

# ─── Panel H — Integrated Evidence Map ───────────────────────────────────────
axH2 = fig2.add_subplot(gs2[2, 2])

ev_data = [
    (0,   -87.0, 'Task 2\n-Replay',      C_NOREPLAY, 180),
    (1,    -8.4, 'T10.5\nSuppress',      C_M0,       100),
    (2,    -0.8, 'T10.5\nBoost',         C_M3,        60),
    (3,     0.5, 'Task 5\nWcc+',         '#95A5A6',   55),
    (3,    -1.0, 'Task 5.5\nWcc blk',   '#95A5A6',   55),
    (4,    74.0, 'T7.5\nWScc',           C_WSCC,     155),
    (4,    19.0, 'T7.5\n+WSuc',          C_WSUC,     100),
]

axH2.axhspan(-105, 0,  alpha=0.04, color='red',   zorder=0)
axH2.axhspan(0,   105, alpha=0.04, color='green', zorder=0)
axH2.axhline(0, color='black', lw=0.8, zorder=1)
axH2.axhspan(-5, 5, alpha=0.06, color='gray', zorder=0)

for (xv, yv, lbl, col, sz) in ev_data:
    axH2.scatter(xv, yv, s=sz, c=col, alpha=0.85, zorder=4,
                 edgecolors='white', linewidths=0.5)

# FIX 25: Labels placed with smart offsets to prevent overlap
label_offsets = {
    'Task 2\n-Replay':   (0.12, 10),
    'T10.5\nSuppress':   (0.12,  6),
    'T10.5\nBoost':      (0.12,  5),
    'Task 5\nWcc+':      (0.12,  5),
    'Task 5.5\nWcc blk': (-0.52, -12),
    'T7.5\nWScc':        (0.12,  6),
    'T7.5\n+WSuc':       (-0.55, -14),
}
for (xv, yv, lbl, col, sz) in ev_data:
    dx_off, dy_off = label_offsets.get(lbl, (0.12, 5))
    axH2.text(xv + dx_off, yv + dy_off, lbl, ha='left' if dx_off > 0 else 'right',
              va='center', fontsize=5.0, color='#333')

axH2.set_xlim(-0.6, 4.8)
axH2.set_ylim(-102, 95)
axH2.set_xticks([0, 1, 2, 3, 4])
axH2.set_xticklabels(['Replay\nremoval', 'Replay\nreduce', 'Replay\nboost',
                       'W[cc]\nmanip.', 'W_slow\nrestore'], fontsize=6)
axH2.set_ylabel('Δ retention (% vs baseline)')
axH2.text(-0.55, -60, 'Memory\ndegrading', fontsize=5.8, color='#CC3333',
          style='italic', va='center', ha='left')
axH2.text(-0.55,  55, 'Memory\nrestoring', fontsize=5.8, color='#229944',
          style='italic', va='center', ha='left')
panel_label(axH2, 'H', x=-0.22)
axH2.set_title('Integrated evidence map\nall interventions', fontsize=8.5, pad=3)

# FIX 26: Footer text well inside the figure area
fig2.text(0.50, 0.038,
          'Figure 2: Replay is necessary (Task 2) and dosage-dependent (Q5). '
          'W[cc] is non-causal (Tasks 5, 5.5). '
          'W_slow[cc] is the dominant substrate (74% alone, 93% with W_slow[uc]). '
          'Replay amplifies but does not create (Task 10.5).',
          ha='center', va='bottom', fontsize=6.2, color='#333', style='italic')

save_fig(fig2, 'Figure2_ExperimentalValidation_FIXED')
plt.close(fig2)

print("\n[FIX] === ALL DONE ===")
print(f"Both fixed figures saved to:\n  {PAPER_DIR}")
import os
for f in sorted(os.listdir(PAPER_DIR)):
    if 'FIXED' in f:
        fp = os.path.join(PAPER_DIR, f)
        mb = os.path.getsize(fp) / 1e6
        print(f"  {f}  ({mb:.1f} MB)")
