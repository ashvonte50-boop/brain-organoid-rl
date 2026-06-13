"""
FINAL MECHANISTIC VALIDATION — PUBLICATION-GRADE FIGURES
Replay Distortion / Directional Schema Abstraction
===================================================

Generates 10 publication-quality figures from pilot_ablations.pkl.
All figures: 300 dpi PNG + PDF + SVG vector export.

Usage:
  python ablation_figures_pub.py
"""
import os, sys, pickle, warnings
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
FIGDIR = os.path.join(BASE, 'pub_figures')
os.makedirs(FIGDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'pilot_ablations.pkl'), 'rb') as f:
    RAW = pickle.load(f)

MODE = 'natural'
N_SEEDS = 3

# ── Publication matplotlib style ───────────────────────────────────────────────
plt.rcParams.update({
    # Font
    'font.family':           'DejaVu Sans',
    'font.size':             11,
    'axes.titlesize':        13,
    'axes.titleweight':      'bold',
    'axes.labelsize':        12,
    'axes.labelweight':      'bold',
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'legend.fontsize':       10,
    'legend.title_fontsize': 10,
    # Lines & spines
    'axes.linewidth':        1.4,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'xtick.major.width':     1.4,
    'ytick.major.width':     1.4,
    'xtick.major.size':      5,
    'ytick.major.size':      5,
    'lines.linewidth':       2.0,
    # Error bars
    'errorbar.capsize':      5,
    # Grid
    'axes.grid':             False,
    # Save
    'savefig.dpi':           300,
    'savefig.bbox':          'tight',
    'savefig.facecolor':     'white',
    'figure.facecolor':      'white',
    'figure.dpi':            150,
})

# ── Colour scheme (Nature Neuroscience palette) ────────────────────────────────
C = {
    'full':     '#2166AC',   # deep blue  — full model
    'M5':       '#D73027',   # red        — most important
    'M1':       '#E8601C',   # orange-red
    'M10':      '#F4A736',   # amber
    'M2':       '#7BBCDA',   # light blue — negligible
    'M7':       '#A9C5A0',   # sage green — negligible
    'ablated':  '#E84855',
    'ns':       '#AAAAAA',
    'sig':      '#222222',
    'ci':       '#BBDDF0',
    'grid':     '#EEEEEE',
    'zero':     '#555555',
}

COND_COLORS = {
    'FULL':       C['full'],
    'ABLATE_M5':  C['M5'],
    'ABLATE_M1':  C['M1'],
    'ABLATE_M10': C['M10'],
    'ABLATE_M2':  C['M2'],
    'ABLATE_M7':  C['M7'],
}
COND_LABELS_SHORT = {
    'FULL':       'Full\nModel',
    'ABLATE_M5':  '−M5\nDrift',
    'ABLATE_M1':  '−M1\nOv.Coh',
    'ABLATE_M10': '−M10\nReconsol',
    'ABLATE_M2':  '−M2\nXLTD',
    'ABLATE_M7':  '−M7\nHetTag',
}
COND_LABELS_LONG = {
    'FULL':       'Full Model (all mechanisms)',
    'ABLATE_M5':  '−M5: Directional Drift',
    'ABLATE_M1':  '−M1: Overlap-Sensitive Coherence',
    'ABLATE_M10': '−M10: Reconsolidation Window',
    'ABLATE_M2':  '−M2: Cross-Assembly LTD',
    'ABLATE_M7':  '−M7: Heterosynaptic LTD Tag',
}
COND_ORDER = ['FULL','ABLATE_M5','ABLATE_M1','ABLATE_M10','ABLATE_M2','ABLATE_M7']

METRIC_META = {
    'dai_core':       {'label': 'DAIₑₒₑₑ (Directional Alignment Index)',
                       'short': 'DAI_core', 'ylim': None, 'fmt': '+.3f'},
    'real_schema':    {'label': 'REAL_SCHEMA (Weight Schema Index)',
                       'short': 'REAL_SCHEMA', 'ylim': None, 'fmt': '+.3f'},
    'distortion':     {'label': 'Distortion Index (centroid displacement)',
                       'short': 'Distortion', 'ylim': (0, None), 'fmt': '.4f'},
    'retention_A':    {'label': 'Retention — Memory A (Iₛʸⁿ score)',
                       'short': 'Ret_A', 'ylim': None, 'fmt': '.3f'},
    'retention_mean': {'label': 'Mean Retention (A–D)',
                       'short': 'Ret_mean', 'ylim': None, 'fmt': '.3f'},
}

# ── Statistics helpers ─────────────────────────────────────────────────────────

def get_vals(cname, metric):
    return np.array([s[MODE][metric] for s in RAW.get(cname, [])
                     if MODE in s and metric in s[MODE]])

def agg(cname, metric):
    v = get_vals(cname, metric)
    if len(v) == 0:
        return dict(mean=np.nan, sem=0, ci95=(np.nan,np.nan), vals=v, n=0)
    m   = float(np.mean(v))
    sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    ci  = (m - 1.96*sem, m + 1.96*sem)
    return dict(mean=m, sem=sem, ci95=ci, vals=v, n=len(v))

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2: return np.nan
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                 / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))

def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def welch_t(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan, 1.0
    t, p = ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

def effect_ci(d, n1, n2):
    """95 % CI on Cohen's d (large-sample approx)."""
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2-2)) + 1e-12)
    return d - 1.96*se, d + 1.96*se

# ── Save helper ────────────────────────────────────────────────────────────────

def savefig(fig, name):
    for ext in ('pdf', 'svg'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), format=ext,
                    bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGDIR, f'{name}.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {name}.[pdf|svg|png]', flush=True)

def footnote(fig, text, y=0.01):
    fig.text(0.5, y, text, ha='center', va='bottom',
             fontsize=8.5, color='#555555', style='italic',
             transform=fig.transFigure)

# ── Shared bar engine ──────────────────────────────────────────────────────────

def bar_panel(ax, metric, ylabel=None, title=None, show_n=True,
              annotate_delta=False, highlight=None, ylim=None):
    full_a = agg('FULL', metric)
    full_v = full_a['vals']

    xs, means, sems, colors, labels, ns, sigs, ds = [], [], [], [], [], [], [], []
    for i, cname in enumerate(COND_ORDER):
        if cname not in RAW: continue
        a  = agg(cname, metric)
        v  = a['vals']
        t, p = welch_t(full_v, v) if cname != 'FULL' else (np.nan, 1.0)
        d  = cohens_d(full_v, v)  if cname != 'FULL' else 0.0
        c  = COND_COLORS.get(cname, '#888888')
        if cname == 'FULL':
            c = C['full']
        elif p >= 0.05:
            c = matplotlib.colors.to_rgba(c, alpha=0.55)
        xs.append(i); means.append(a['mean']); sems.append(a['sem'])
        colors.append(c); labels.append(COND_LABELS_SHORT.get(cname, cname))
        ns.append(a['n']); sigs.append(sig_label(p)); ds.append(d)

    x = np.arange(len(xs))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.8, zorder=3,
                  error_kw=dict(elinewidth=1.6, ecolor='#333333', capthick=1.6))

    # Full model reference dashes
    ax.axhline(full_a['mean'], color=C['full'], lw=1.4, ls='--', alpha=0.65, zorder=2)

    # Significance + delta annotations
    ymax = np.nanmax([m + s for m, s in zip(means, sems)])
    ymin = np.nanmin([m - s for m, s in zip(means, sems)])
    span = abs(ymax - ymin) if abs(ymax - ymin) > 1e-6 else 0.1
    for xi, (sig, mean, sem, d) in enumerate(zip(sigs, means, sems, ds)):
        if sig == 'n.s.' or xi == 0: continue
        ax.text(xi, mean + sem + 0.04*span, sig,
                ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=C['sig'])
        if annotate_delta and not np.isnan(d):
            ax.text(xi, mean - sem - 0.08*span, f'd={d:+.2f}',
                    ha='center', va='top', fontsize=8, color='#444444')

    if show_n:
        for xi, n in enumerate(ns):
            ax.text(xi, ymin - 0.15*span, f'n={n}',
                    ha='center', va='top', fontsize=8, color='#777777')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if ylim:
        ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.grid(axis='y', color=C['grid'], zorder=0, linewidth=0.8)
    return bars


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Single-ablation: DAI_core
# ══════════════════════════════════════════════════════════════════════════════
def fig01_dai():
    print('  Fig 1: DAI_core single-ablation bar chart', flush=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_panel(ax, 'dai_core',
              ylabel='Directional Alignment Index (DAI$_{core}$)',
              title='Effect of Single-Mechanism Ablations on Directional Schema Abstraction',
              annotate_delta=True)
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')

    legend_items = [
        mpatches.Patch(facecolor=C['full'],    label='Full model'),
        mpatches.Patch(facecolor=C['M5'],      label='Significant ablation (p<0.05)'),
        mpatches.Patch(facecolor=C['ns'],      label='Non-significant ablation'),
        plt.Line2D([0],[0], color=C['full'], lw=1.5, ls='--', label='Full model mean'),
    ]
    ax.legend(handles=legend_items, loc='lower right', framealpha=0.9,
              edgecolor='#cccccc', fontsize=9)
    footnote(fig, f'Welch t-test vs full model. Error bars = SEM. n={N_SEEDS} seeds per condition. '
             'Significance: * p<0.05, ** p<0.01, *** p<0.001. Cohen\'s d annotated below bars.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig01_dai_core')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Single-ablation: REAL_SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
def fig02_real_schema():
    print('  Fig 2: REAL_SCHEMA single-ablation bar chart', flush=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_panel(ax, 'real_schema',
              ylabel='REAL_SCHEMA (Weight Schema Index)',
              title='Effect of Single-Mechanism Ablations on Schema Weight Structure',
              annotate_delta=True)
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    footnote(fig, 'REAL_SCHEMA = (W$_{core-core}$ − W$_{unique→core}$) / '
             '(W$_{core-core}$ + W$_{unique→core}$). Positive = core-dominated schema. '
             f'n={N_SEEDS} seeds. Error bars = SEM.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig02_real_schema')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Single-ablation: Distortion
# ══════════════════════════════════════════════════════════════════════════════
def fig03_distortion():
    print('  Fig 3: Distortion single-ablation bar chart', flush=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_panel(ax, 'distortion',
              ylabel='Distortion Index (L2 centroid displacement per event)',
              title='Effect of Single-Mechanism Ablations on Replay Distortion',
              annotate_delta=True)
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    footnote(fig, 'Distortion = mean ||Δcentroid||$_2$ per replay event. '
             'Higher distortion = more schema-directed centroid movement. '
             f'n={N_SEEDS} seeds. Error bars = SEM.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig03_distortion')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Retention heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig04_retention_heatmap():
    print('  Fig 4: Retention heatmap (conditions × memories)', flush=True)
    mem_metrics = ['retention_A','retention_B','retention_C','retention_D']
    mem_labels  = ['Memory A','Memory B','Memory C','Memory D']

    rows, row_labels, row_ns = [], [], []
    for cname in COND_ORDER:
        row, n = [], 0
        for m in mem_metrics:
            a = agg(cname, m)
            row.append(a['mean'])
            n = max(n, a['n'])
        rows.append(row)
        row_labels.append(COND_LABELS_LONG.get(cname, cname))
        row_ns.append(n)

    mat  = np.array(rows)
    vmin = np.nanmin(mat) - 0.005
    vmax = np.nanmax(mat) + 0.005
    cmap = LinearSegmentedColormap.from_list(
        'ret', ['#313695','#74ADD1','#FEE090','#F46D43','#A50026'])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v  = mat[i, j]
            fc = 'white' if (v < vmin + 0.3*(vmax-vmin) or v > vmax - 0.3*(vmax-vmin)) else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=11, color=fc, fontweight='bold')

    ax.set_xticks(range(4)); ax.set_xticklabels(mem_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f'{l}  (n={n})' for l, n in zip(row_labels, row_ns)], fontsize=10)
    ax.set_xlabel('Memory', fontsize=12, fontweight='bold')
    ax.set_title('Retention Across Memories by Ablation Condition\n'
                 'Sequential learning A→B→C→D; final probe scores',
                 fontsize=12, fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Retention Score (I$_{syn}$ differential)', fontsize=10)
    # Highlight FULL row
    ax.add_patch(mpatches.FancyBboxPatch((-0.5,-0.5), 4, 1,
                 boxstyle='round,pad=0.05', linewidth=2.5,
                 edgecolor=C['full'], facecolor='none', zorder=5))
    footnote(fig, f'Values = mean retention score across n={N_SEEDS} seeds. '
             'Blue = low retention, red = high retention.')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    savefig(fig, 'fig04_retention_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Mechanism importance ranking
# ══════════════════════════════════════════════════════════════════════════════
def fig05_importance():
    print('  Fig 5: Mechanism importance ranking', flush=True)
    full_dai = get_vals('FULL', 'dai_core')
    full_rs  = get_vals('FULL', 'real_schema')
    full_ret = get_vals('FULL', 'retention_mean')

    mechs, d_dai_v, d_rs_v, d_ret_v, p_vals, cd_vals = [], [], [], [], [], []
    for cname in ['ABLATE_M5','ABLATE_M1','ABLATE_M10','ABLATE_M2','ABLATE_M7']:
        if cname not in RAW: continue
        dai_v = get_vals(cname,'dai_core')
        rs_v  = get_vals(cname,'real_schema')
        ret_v = get_vals(cname,'retention_mean')
        d_dai_v.append(float(np.mean(full_dai) - np.mean(dai_v)))
        d_rs_v.append(float(np.mean(full_rs)  - np.mean(rs_v)))
        d_ret_v.append(float(np.mean(full_ret) - np.mean(ret_v)))
        cd_vals.append(cohens_d(full_dai, dai_v))
        _, p = welch_t(full_dai, dai_v)
        p_vals.append(p)
        mechs.append(cname)

    # Sort by |ΔDAI|
    order = np.argsort(np.abs(d_dai_v))[::-1]
    mechs = [mechs[i] for i in order]
    d_dai_v = [d_dai_v[i] for i in order]
    d_rs_v  = [d_rs_v[i] for i in order]
    d_ret_v = [d_ret_v[i] for i in order]
    p_vals  = [p_vals[i] for i in order]
    cd_vals = [cd_vals[i] for i in order]

    x = np.arange(len(mechs))
    w = 0.26
    labels = [COND_LABELS_SHORT.get(c,'').replace('\n',' ') for c in mechs]
    bar_colors = [COND_COLORS.get(c, '#888888') for c in mechs]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x-w, d_dai_v, w, label='ΔDAI$_{core}$',
                color=[matplotlib.colors.to_rgba(bc, 0.9) for bc in bar_colors],
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x,   d_rs_v,  w, label='ΔREAL_SCHEMA',
                color=[matplotlib.colors.to_rgba(bc, 0.65) for bc in bar_colors],
                edgecolor='white', linewidth=0.5, hatch='//')
    b3 = ax.bar(x+w, d_ret_v, w, label='ΔRetention (mean)',
                color=[matplotlib.colors.to_rgba(bc, 0.4) for bc in bar_colors],
                edgecolor='white', linewidth=0.5, hatch='xx')
    ax.axhline(0, color='black', lw=1.2, ls='-', zorder=5)

    # Sig stars on ΔDAI bars
    for xi, (p, dv) in enumerate(zip(p_vals, d_dai_v)):
        sig = sig_label(p)
        if sig != 'n.s.':
            yoff = dv + 0.003 if dv >= 0 else dv - 0.008
            ax.text(xi-w, yoff, sig, ha='center',
                    va='bottom' if dv >= 0 else 'top',
                    fontsize=11, fontweight='bold', color=C['sig'])
        # Rank badge
        ax.text(xi, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
                f'#{xi+1}', ha='center', va='bottom',
                fontsize=9, color='#555555', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Ablate {l}' for l in labels], fontsize=10)
    ax.set_ylabel('Δ from Full Model (Full − Ablated)', fontsize=11, fontweight='bold')
    ax.set_title('Mechanism Importance Ranking\n'
                 'Higher bar = larger performance drop when mechanism removed = more critical',
                 fontsize=12, fontweight='bold', pad=8)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#cccccc', fontsize=9)
    ax.grid(axis='y', color=C['grid'], zorder=0, linewidth=0.8)

    # Impact classification bands
    ax.axhspan(0.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.05 else 0.1,
               alpha=0.06, color='#D73027', zorder=0, label='HIGH')
    ax.axhspan(0.01, 0.05, alpha=0.06, color='#F4A736', zorder=0)
    ax.axhspan(0,    0.01, alpha=0.06, color='#AAAAAA', zorder=0)

    footnote(fig, f'n={N_SEEDS} seeds per condition. Ranked by |ΔDAI$_{{core}}$|. '
             'Shading: HIGH >0.05, MEDIUM 0.01–0.05, LOW <0.01. '
             'Stars on ΔDAI bars indicate Welch t-test significance.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig05_importance_ranking')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Effect-size forest plot (DAI_core)
# ══════════════════════════════════════════════════════════════════════════════
def fig06_forest():
    print('  Fig 6: Effect-size forest plot', flush=True)
    full_v = get_vals('FULL', 'dai_core')
    rows = []
    for cname in ['ABLATE_M5','ABLATE_M1','ABLATE_M10','ABLATE_M2','ABLATE_M7']:
        if cname not in RAW: continue
        v  = get_vals(cname, 'dai_core')
        d  = cohens_d(full_v, v)
        t, p = welch_t(full_v, v)
        lo, hi = effect_ci(d, len(full_v), len(v))
        rows.append({'cname': cname, 'd': d, 'lo': lo, 'hi': hi,
                     'p': p, 'n': len(v), 'sig': sig_label(p)})
    rows.sort(key=lambda r: r['d'], reverse=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(rows)*0.9 + 2)))
    for yi, r in enumerate(rows):
        c  = COND_COLORS.get(r['cname'], C['ablated'])
        lw = 2.8 if r['p'] < 0.05 else 1.8
        ax.plot([r['lo'], r['hi']], [yi, yi], color=c, lw=lw, solid_capstyle='round', zorder=3)
        ax.plot([r['lo'], r['hi']], [yi, yi], color=c, lw=lw, zorder=3)
        ax.scatter(r['d'], yi, s=140, color=c, zorder=4,
                   edgecolors='white', linewidths=1.5)
        ax.text(max(r['hi'], 0.02) + 0.05, yi,
                f"d = {r['d']:+.3f}  [{r['lo']:+.2f}, {r['hi']:+.2f}]   "
                f"p = {r['p']:.3f}   {r['sig']}   n={r['n']}",
                va='center', fontsize=9.5, color='#222222')

    ax.axvline(0,    color='#333333', lw=1.3, ls='--', zorder=2)
    ax.axvline(0.2,  color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax.axvline(0.5,  color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax.axvline(0.8,  color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax.axvline(-0.2, color='#cccccc', lw=0.8, ls=':', zorder=1)
    ax.text(0.2,  len(rows)+0.1, 'small',  ha='center', fontsize=8, color='#999999')
    ax.text(0.5,  len(rows)+0.1, 'medium', ha='center', fontsize=8, color='#999999')
    ax.text(0.8,  len(rows)+0.1, 'large',  ha='center', fontsize=8, color='#999999')

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([COND_LABELS_LONG.get(r['cname'],'') for r in rows], fontsize=10)
    ax.set_xlabel("Cohen's d  (Full − Ablated, DAI$_{core}$)\n"
                  "Positive = removing mechanism reduces DAI = mechanism is important",
                  fontsize=11, fontweight='bold')
    ax.set_title("Effect-Size Forest Plot\n"
                 "95% CI on Cohen's d for each single-mechanism ablation",
                 fontsize=12, fontweight='bold', pad=8)
    sig_patch = mpatches.Patch(color=C['M5'],  label='p<0.05  (thick line)')
    ns_patch  = mpatches.Patch(color=C['ns'],  label='p≥0.05 (thin line)')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')
    ax.grid(axis='x', color=C['grid'], zorder=0)

    footnote(fig, "Effect size: Cohen's d with pooled SD. 95% CI by large-sample approx. "
             f"n={N_SEEDS} per condition. All n.s. at n=3 (low power); "
             "effect sizes are primary interpretive metric.")
    fig.tight_layout(rect=[0, 0.06, 0.72, 1])
    savefig(fig, 'fig06_forest_plot')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Per-seed strip chart (individual data points visible)
# ══════════════════════════════════════════════════════════════════════════════
def fig07_strip():
    print('  Fig 7: Per-seed strip chart (DAI + RS)', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, ylabel in zip(axes,
            ['dai_core', 'real_schema'],
            ['DAI$_{core}$', 'REAL_SCHEMA']):
        full_m = agg('FULL', metric)['mean']
        for xi, cname in enumerate(COND_ORDER):
            vals = get_vals(cname, metric)
            if len(vals) == 0: continue
            c = COND_COLORS.get(cname, '#888888')
            # Jitter
            jx = np.linspace(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), xi) + jx, vals,
                       color=c, s=90, zorder=4, edgecolors='white', linewidths=1.2,
                       alpha=0.85)
            m   = np.mean(vals)
            sem = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
            ax.plot([xi-0.22, xi+0.22], [m, m], color=c, lw=2.8, solid_capstyle='round', zorder=5)
            ax.plot([xi, xi], [m-sem, m+sem], color=c, lw=2.0, zorder=5)
        ax.axhline(full_m, color=C['full'], lw=1.4, ls='--', alpha=0.6, zorder=2)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS_SHORT.get(c,'') for c in COND_ORDER], fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'Per-Seed {ylabel} Values', fontsize=12, fontweight='bold')
        ax.grid(axis='y', color=C['grid'], linewidth=0.8, zorder=0)

    fig.suptitle('Individual Seed Data Points with Mean ± SEM\n'
                 'Each dot = one independent replication (seed)',
                 fontsize=13, fontweight='bold', y=1.01)
    footnote(fig, f'Horizontal bars = mean; vertical bars = SEM. n={N_SEEDS} seeds per condition. '
             'Dashed line = full model mean. Jitter added for visibility.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig07_strip_chart')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — 4-panel summary (all metrics, one figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig08_summary():
    print('  Fig 8: 4-panel summary', flush=True)
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

    panels = [
        ('dai_core',       'DAI$_{core}$\n(Directional Alignment Index)', gs[0,0]),
        ('real_schema',    'REAL_SCHEMA\n(Weight Schema Index)',           gs[0,1]),
        ('distortion',     'Distortion Index\n(centroid displacement)',    gs[1,0]),
        ('retention_mean', 'Mean Retention\n(memories A–D)',          gs[1,1]),
    ]
    for metric, ylabel, pos in panels:
        ax = fig.add_subplot(pos)
        bar_panel(ax, metric, ylabel=ylabel, show_n=(metric == 'dai_core'), annotate_delta=False)
        ax.set_xlabel('')

    fig.suptitle('Mechanistic Ablation Study — Summary of All Outcome Metrics\n'
                 'Single-mechanism ablations: Full model vs each mechanism removed',
                 fontsize=14, fontweight='bold', y=1.01)

    legend_items = [
        mpatches.Patch(facecolor=C['full'],  label='Full model'),
        mpatches.Patch(facecolor=C['M5'],    label='M5: Directional Drift'),
        mpatches.Patch(facecolor=C['M1'],    label='M1: Overlap Coherence'),
        mpatches.Patch(facecolor=C['M10'],   label='M10: Reconsolidation'),
        mpatches.Patch(facecolor=C['M2'],    label='M2: Cross-Assembly LTD'),
        mpatches.Patch(facecolor=C['M7'],    label='M7: Heterosynaptic Tag'),
        plt.Line2D([0],[0], color=C['full'], lw=1.4, ls='--', label='Full model mean'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=7,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
               edgecolor='#cccccc', fontsize=9)
    footnote(fig, f'Error bars = SEM. n={N_SEEDS} seeds per condition. '
             'Significance vs full model (Welch t-test): * p<0.05, ** p<0.01.', y=0.0)
    fig.savefig(os.path.join(FIGDIR, 'fig08_summary.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGDIR, 'fig08_summary.svg'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGDIR, 'fig08_summary.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  Saved fig08_summary.[pdf|svg|png]', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — Radar / spider chart of mechanism profiles
# ══════════════════════════════════════════════════════════════════════════════
def fig09_radar():
    print('  Fig 9: Radar chart — mechanism profiles', flush=True)
    metrics_r = ['dai_core','real_schema','retention_mean','distortion']
    labels_r  = ['DAIₑₒₑₑ','REAL_\nSCHEMA','Retention','Distortion']

    # Normalise each metric to [0,1] across conditions
    data_r = {}
    for m in metrics_r:
        vals = {c: agg(c, m)['mean'] for c in COND_ORDER if c in RAW}
        mn, mx = min(vals.values()), max(vals.values())
        rng = mx - mn if mx != mn else 1.0
        data_r[m] = {c: (v - mn) / rng for c, v in vals.items()}

    N = len(metrics_r)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for cname in COND_ORDER:
        if cname not in RAW: continue
        vals = [data_r[m][cname] for m in metrics_r]
        vals += vals[:1]
        c = COND_COLORS.get(cname, '#888888')
        lw = 2.8 if cname == 'FULL' else 1.8
        alpha = 0.18 if cname == 'FULL' else 0.08
        ax.plot(angles, vals, color=c, lw=lw, label=COND_LABELS_LONG.get(cname,''),
                zorder=4 if cname=='FULL' else 3)
        ax.fill(angles, vals, color=c, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, size=12, fontweight='bold')
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines['polar'].set_color('#cccccc')
    ax.grid(color='#dddddd', linewidth=0.8)
    ax.set_title('Mechanism Profile Radar Chart\n'
                 'Normalised performance across all metrics',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              framealpha=0.9, edgecolor='#cccccc', fontsize=9)
    footnote(fig, 'Values normalised to [0,1] per metric across conditions. '
             f'Outer edge = best performance. n={N_SEEDS} seeds per condition.', y=0.0)
    fig.tight_layout()
    savefig(fig, 'fig09_radar')


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — DAI vs REAL_SCHEMA scatter (per-seed dots)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_scatter():
    print('  Fig 10: DAI vs REAL_SCHEMA scatter (mechanistic space)', flush=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    for cname in COND_ORDER:
        if cname not in RAW: continue
        dai_v = get_vals(cname, 'dai_core')
        rs_v  = get_vals(cname, 'real_schema')
        c     = COND_COLORS.get(cname, '#888888')
        lbl   = COND_LABELS_LONG.get(cname,'')
        ax.scatter(dai_v, rs_v, color=c, s=160, zorder=4,
                   edgecolors='white', linewidths=1.5,
                   label=lbl, alpha=0.9)
        # Mean crosshair
        mx, my = np.mean(dai_v), np.mean(rs_v)
        ax.scatter([mx], [my], color=c, s=320, zorder=5,
                   edgecolors='black', linewidths=1.5, marker='D')
        # 95% CI ellipse
        if len(dai_v) > 1:
            sem_x = np.std(dai_v, ddof=1) / np.sqrt(len(dai_v))
            sem_y = np.std(rs_v,  ddof=1) / np.sqrt(len(rs_v))
            ell   = mpatches.Ellipse((mx, my), 4*sem_x, 4*sem_y,
                                     edgecolor=c, facecolor='none',
                                     linewidth=1.5, linestyle='--', zorder=3, alpha=0.7)
            ax.add_patch(ell)

    # Full model centroid label
    full_dai_m = np.mean(get_vals('FULL','dai_core'))
    full_rs_m  = np.mean(get_vals('FULL','real_schema'))
    ax.annotate('Full\nModel', (full_dai_m, full_rs_m),
                xytext=(full_dai_m-0.012, full_rs_m+0.015),
                fontsize=9, color=C['full'], fontweight='bold')

    ax.set_xlabel('DAI$_{core}$ (Directional Alignment Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('REAL_SCHEMA (Weight Schema Index)',           fontsize=12, fontweight='bold')
    ax.set_title('Mechanistic Space: DAI$_{core}$ vs REAL_SCHEMA\n'
                 'Each dot = one seed. Diamond = condition mean. Ellipse = 95% CI on mean.',
                 fontsize=12, fontweight='bold', pad=8)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc',
              fontsize=9, markerscale=0.9)
    ax.grid(color=C['grid'], linewidth=0.8, zorder=0)

    footnote(fig, f'n={N_SEEDS} seeds per condition. Ellipses show 95% CI on condition mean '
             '(±1.96×SEM per axis). Diamond markers = condition means.')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, 'fig10_scatter_mechanistic')


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary():
    full_dai = get_vals('FULL','dai_core')
    full_rs  = get_vals('FULL','real_schema')
    full_ret = get_vals('FULL','retention_mean')

    print()
    print('='*95)
    print('MECHANISTIC VALIDATION — FULL RESULTS TABLE')
    print(f'Pilot n={N_SEEDS} per condition | mode={MODE} | RS fixed | natural-only')
    print('='*95)
    hdr = (f'  {"Condition":<30} {"DAI±SEM":>12} {"ΔDAI":>8} '
           f'{"RS±SEM":>12} {"ΔRS":>8} {"Ret":>7} {"d(DAI)":>8} {"p":>8} Sig')
    print(hdr)
    print('  '+'-'*93)

    dai_m = np.mean(full_dai); rs_m = np.mean(full_rs); ret_m = np.mean(full_ret)
    dai_se= np.std(full_dai,ddof=1)/np.sqrt(len(full_dai))
    rs_se = np.std(full_rs, ddof=1)/np.sqrt(len(full_rs))
    print(f'  {"Full Model":<30} {dai_m:+.4f}±{dai_se:.4f} {"—":>8} '
          f'{rs_m:.4f}±{rs_se:.4f} {"—":>8} {ret_m:7.4f} {"—":>8} {"—":>8}')

    for cname in ['ABLATE_M5','ABLATE_M1','ABLATE_M10','ABLATE_M2','ABLATE_M7']:
        if cname not in RAW: continue
        dai_v = get_vals(cname,'dai_core'); rs_v = get_vals(cname,'real_schema')
        ret_v = get_vals(cname,'retention_mean')
        m_dai = np.mean(dai_v); se_dai = np.std(dai_v,ddof=1)/np.sqrt(len(dai_v))
        m_rs  = np.mean(rs_v);  se_rs  = np.std(rs_v, ddof=1)/np.sqrt(len(rs_v))
        m_ret = np.mean(ret_v)
        d = cohens_d(full_dai, dai_v)
        t, p = welch_t(full_dai, dai_v)
        lbl = COND_LABELS_LONG.get(cname,'').replace('—','')
        print(f'  {lbl:<30} {m_dai:+.4f}±{se_dai:.4f} {m_dai-dai_m:+8.4f} '
              f'{m_rs:.4f}±{se_rs:.4f} {m_rs-rs_m:+8.4f} {m_ret:7.4f} {d:+8.3f} {p:8.4f} {sig_label(p)}')

    print()
    print('Metric definitions:')
    print('  DAI_core  : Directional Alignment Index on core neurons (cosine similarity)')
    print('  REAL_SCHEMA: (W_core-core - W_unique->core) / (W_core-core + W_unique->core)')
    print('  Retention : I_syn differential probe score (mean across memories A-D)')
    print('  Cohen d   : pooled-SD effect size (Full - Ablated); positive = mechanism matters')
    print('  p         : Welch two-sample t-test vs full model (two-tailed)')
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\nGenerating publication-grade figures -> {FIGDIR}', flush=True)
    print(f'Data: pilot_ablations.pkl  ({len(RAW)} conditions, n={N_SEEDS} seeds each)\n', flush=True)

    print_summary()

    fig01_dai()
    fig02_real_schema()
    fig03_distortion()
    fig04_retention_heatmap()
    fig05_importance()
    fig06_forest()
    fig07_strip()
    fig08_summary()
    fig09_radar()
    fig10_scatter()

    print(f'\nAll 10 figures saved to {FIGDIR}')
    print('Formats: PDF (vector) + SVG (vector) + PNG (300 dpi)')
