"""
PUBLICATION-GRADE ABLATION FIGURES
Replay Distortion / Directional Schema Abstraction
===================================================

Generates Figures 1–10 from ablation_results/ PKL files.

Usage:
  python ablation_figures.py                  # all figures from saved results
  python ablation_figures.py --fig 1 2 5      # specific figures only
  python ablation_figures.py --mode hyper     # use hyper-replay results
"""
import os, sys, pickle, argparse
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_DIR  = r'C:\Users\Admin\brain-organoid-rl\ablation_results'
FIG_DIR = os.path.join(IN_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
FONT_FAMILY = 'DejaVu Sans'
plt.rcParams.update({
    'font.family':        FONT_FAMILY,
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     1.2,
    'xtick.major.width':  1.2,
    'ytick.major.width':  1.2,
    'lines.linewidth':    2.0,
    'errorbar.capsize':   4,
})

# ── Colour palette ─────────────────────────────────────────────────────────────
C_FULL    = '#2E86AB'   # blue   — full model
C_ABLATE  = '#E84855'   # red    — ablated
C_NEUTRAL = '#8B8B8B'   # grey   — n.s.
C_GRAD    = ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']  # diverging

MECH_COLORS = {
    'M1': '#1b7837', 'M2': '#762a83', 'M3': '#e66101', 'M4': '#5e3c99',
    'M5': '#2166ac', 'M6': '#d73027', 'M7': '#1a9641', 'M8': '#c2a5cf',
    'M9': '#f4a582', 'M10': '#4d9221', 'MB': '#404040',
}

MECH_ORDER = ['FULL', 'ABLATE_M1', 'ABLATE_M2', 'ABLATE_M3', 'ABLATE_M4',
              'ABLATE_M5', 'ABLATE_M6', 'ABLATE_M7', 'ABLATE_M8',
              'ABLATE_M9', 'ABLATE_M10', 'ABLATE_MB']

MECH_XLABELS = ['Full\nModel', '−M1\nOv.Coh', '−M2\nXLTD', '−M3\nOvPri',
                '−M4\nPersBud', '−M5\nDrift', '−M6\nFatigue', '−M7\nHetTag',
                '−M8\nDecorr', '−M9\nWTA', '−M10\nReconsol', '−MB\nCoreBoost']

CUMUL_ORDER = ['MB', 'M5', 'M7', 'M2', 'M6', 'M1', 'M9', 'M3', 'M4', 'M8', 'M10']
MEMORY_LABELS = ['Memory A', 'Memory B', 'Memory C', 'Memory D']

METRICS = ['retention_A', 'retention_B', 'retention_C', 'retention_D',
           'retention_mean', 'dai_core', 'dai_unique', 'real_schema', 'distortion']


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load(name):
    p = os.path.join(IN_DIR, f'{name}.pkl')
    if not os.path.exists(p):
        print(f'  WARNING: {p} not found', flush=True)
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def _agg(seed_list, mode='natural'):
    out = {}
    for k in METRICS:
        vals = [s[mode][k] for s in seed_list if mode in s and k in s.get(mode, {})]
        if vals:
            out[k + '_mean'] = float(np.mean(vals))
            out[k + '_sem']  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[k + '_vals'] = vals
            out[k + '_n']    = len(vals)
    return out


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def _cohen_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s = np.sqrt(((len(a) - 1) * a.std(ddof=1)**2 + (len(b) - 1) * b.std(ddof=1)**2) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (s + 1e-9))


def _add_sig_bracket(ax, x1, x2, y, stars, h=0.02, color='black', fontsize=10):
    """Draw significance bracket between x1 and x2 at height y."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color=color)
    ax.text((x1 + x2) / 2, y + h + 0.005, stars, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold')


def _savefig(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path, format=ext, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
    print(f'  Saved {name}.[pdf|svg|png]', flush=True)
    plt.close(fig)


# ── Shared bar-plot engine ─────────────────────────────────────────────────────

def _single_abl_barplot(data, mode, metric_key, ylabel, title, fig_name,
                        ymin=None, ymax=None, show_seeds=True):
    """Generic single-ablation bar chart for one metric."""
    conditions = data if isinstance(data, dict) else {}
    full_seed  = conditions.get('FULL', [])
    full_agg   = _agg(full_seed, mode)
    full_vals  = full_agg.get(f'{metric_key}_vals', [])

    means, sems, colors, labels, n_list, sig_list = [], [], [], [], [], []
    for cname, xlabel in zip(MECH_ORDER, MECH_XLABELS):
        sl = conditions.get(cname, [])
        if not sl:
            means.append(np.nan); sems.append(0); colors.append(C_NEUTRAL)
            labels.append(xlabel); n_list.append(0); sig_list.append('')
            continue
        agg  = _agg(sl, mode)
        m    = agg.get(f'{metric_key}_mean', np.nan)
        s    = agg.get(f'{metric_key}_sem', 0)
        n    = agg.get(f'{metric_key}_n', 0)
        vals = agg.get(f'{metric_key}_vals', [])
        if cname == 'FULL':
            c, sig = C_FULL, ''
        else:
            if full_vals and vals:
                _, p = ttest_ind(full_vals, vals, equal_var=False)
                sig = _sig_stars(p)
                c   = C_ABLATE if p < 0.05 else C_NEUTRAL
            else:
                sig, c = '', C_NEUTRAL
        means.append(m); sems.append(s); colors.append(c)
        labels.append(xlabel); n_list.append(n); sig_list.append(sig)

    x = np.arange(len(MECH_ORDER))
    fig, ax = plt.subplots(figsize=(14, 5.5))

    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor='white',
                  linewidth=0.8, error_kw=dict(elinewidth=1.5, ecolor='#333333', capthick=1.5),
                  zorder=3)

    # Full-model reference line
    full_mean = full_agg.get(f'{metric_key}_mean', np.nan)
    if not np.isnan(full_mean):
        ax.axhline(full_mean, color=C_FULL, lw=1.5, ls='--', alpha=0.7, zorder=2)

    # Significance annotations
    ymax_data = np.nanmax([m + s for m, s in zip(means, sems)] + [full_mean])
    bracket_y = ymax_data + 0.03 * abs(ymax_data)
    for xi, (sig, m) in enumerate(zip(sig_list, means)):
        if sig and sig != 'n.s.' and not np.isnan(m):
            ax.text(xi, m + sems[xi] + 0.015 * abs(ymax_data), sig,
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#222222')

    # Seed count annotations
    if show_seeds:
        for xi, n in enumerate(n_list):
            if n > 0:
                ax.text(xi, -0.04 * abs(np.nanmax(np.abs(means)) + 0.01),
                        f'n={n}', ha='center', va='top', fontsize=8, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.35, zorder=0)

    legend_patches = [
        Patch(color=C_FULL,    label='Full model'),
        Patch(color=C_ABLATE,  label='Ablated (p<0.05)'),
        Patch(color=C_NEUTRAL, label='Ablated (n.s.)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.85,
              fontsize=9, edgecolor='#cccccc')

    ax.text(0.99, 0.01, f'mode={mode}  seeds={max(n_list) if n_list else 0}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#888888')
    fig.tight_layout()
    _savefig(fig, fig_name)


# ── Figure 1: Single ablation — DAI_core ──────────────────────────────────────

def fig01_single_dai(data, mode='natural'):
    print('  Fig 1: Single ablation — DAI_core', flush=True)
    _single_abl_barplot(
        data, mode,
        metric_key='dai_core',
        ylabel='DAI_core (Directional Alignment Index)',
        title='Figure 1: Single-Mechanism Ablations — DAI_core\n'
              'Effect of removing each mechanism on directional schema abstraction',
        fig_name='fig01_single_dai',
    )


# ── Figure 2: Single ablation — REAL_SCHEMA ───────────────────────────────────

def fig02_single_rs(data, mode='natural'):
    print('  Fig 2: Single ablation — REAL_SCHEMA', flush=True)
    _single_abl_barplot(
        data, mode,
        metric_key='real_schema',
        ylabel='REAL_SCHEMA (Weight-based Schema Index)',
        title='Figure 2: Single-Mechanism Ablations — REAL_SCHEMA\n'
              'Effect of removing each mechanism on schema weight structure',
        fig_name='fig02_single_rs',
    )


# ── Figure 3: Single ablation — Distortion ────────────────────────────────────

def fig03_single_dist(data, mode='natural'):
    print('  Fig 3: Single ablation — Distortion', flush=True)
    _single_abl_barplot(
        data, mode,
        metric_key='distortion',
        ylabel='Distortion Index (centroid displacement)',
        title='Figure 3: Single-Mechanism Ablations — Replay Distortion\n'
              'Effect of removing each mechanism on centroid movement magnitude',
        fig_name='fig03_single_dist',
    )


# ── Figure 4: Retention heatmap ───────────────────────────────────────────────

def fig04_retention_heatmap(data, mode='natural'):
    print('  Fig 4: Retention heatmap', flush=True)
    conditions = data if isinstance(data, dict) else {}
    mem_keys   = ['retention_A', 'retention_B', 'retention_C', 'retention_D']

    rows, row_labels, row_n = [], [], []
    for cname, xlabel in zip(MECH_ORDER, MECH_XLABELS):
        sl = conditions.get(cname, [])
        if not sl:
            rows.append([np.nan] * 4)
            row_labels.append(xlabel.replace('\n', ' '))
            row_n.append(0)
            continue
        agg = _agg(sl, mode)
        rows.append([agg.get(f'{k}_mean', np.nan) for k in mem_keys])
        row_labels.append(xlabel.replace('\n', ' '))
        row_n.append(agg.get('retention_A_n', 0))

    mat   = np.array(rows, dtype=float)
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)

    cmap = LinearSegmentedColormap.from_list('ret',
        ['#d73027', '#fee090', '#ffffbf', '#a6d96a', '#1a9641'])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                txt_col = 'white' if (v < vmin + 0.25*(vmax-vmin) or v > vmax - 0.25*(vmax-vmin)) else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=9, color=txt_col, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(MEMORY_LABELS, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f'{l}  (n={n})' for l, n in zip(row_labels, row_n)], fontsize=9)
    ax.set_title(
        'Figure 4: Retention Heatmap\nRows = ablation conditions; Columns = individual memories',
        fontsize=13, fontweight='bold', pad=12,
    )
    plt.colorbar(im, ax=ax, label='Retention Score (I_syn)', shrink=0.8, pad=0.02)
    fig.text(0.5, 0.01, f'mode={mode}', ha='center', fontsize=9, color='#888888')
    fig.tight_layout()
    _savefig(fig, 'fig04_retention_heatmap')


# ── Figure 5: Mechanism importance ranking ────────────────────────────────────

def fig05_importance_ranking(imp_data, mode='natural'):
    print('  Fig 5: Mechanism importance ranking', flush=True)
    if not imp_data:
        print('    No importance data available.', flush=True)
        return

    ranked = sorted(imp_data.items(),
                    key=lambda x: abs(x[1]['delta_dai']) + abs(x[1]['delta_rs']),
                    reverse=True)
    labels   = [mid for mid, _ in ranked]
    d_dai    = [v['delta_dai']    for _, v in ranked]
    d_rs     = [v['delta_rs']     for _, v in ranked]
    d_ret    = [v['delta_ret']    for _, v in ranked]
    p_vals   = [v['p_dai']        for _, v in ranked]
    n_vals   = [v['n_seeds']      for _, v in ranked]

    x     = np.arange(len(labels))
    width = 0.26
    bar_colors_dai = [C_ABLATE if p < 0.05 else C_NEUTRAL for p in p_vals]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    b1 = ax.bar(x - width, d_dai, width, label='ΔDAI_core',    color='#2166ac', alpha=0.9)
    b2 = ax.bar(x,          d_rs,  width, label='ΔREAL_SCHEMA', color='#1a9641', alpha=0.9)
    b3 = ax.bar(x + width,  d_ret, width, label='ΔRetention',   color='#d73027', alpha=0.9)

    ax.axhline(0, color='black', lw=1.0, ls='-')
    for xi, (p, n) in enumerate(zip(p_vals, n_vals)):
        sig = _sig_stars(p)
        if sig != 'n.s.':
            yval = d_dai[xi]
            ax.text(xi - width, yval + (0.002 if yval >= 0 else -0.006), sig,
                    ha='center', va='bottom' if yval >= 0 else 'top',
                    fontsize=10, fontweight='bold', color='#1a1a1a')
        ax.text(xi, -0.035, f'n={n}', ha='center', va='top', fontsize=7.5, color='#666666')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Ablate {lbl}' for lbl in labels], fontsize=10)
    ax.set_ylabel('Δ from Full Model (Full − Ablated)', fontsize=12)
    ax.set_title(
        'Figure 5: Mechanism Importance Ranking\n'
        'Higher bar = larger drop when mechanism is removed = more important',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.legend(loc='upper right', framealpha=0.85, fontsize=10)
    ax.grid(axis='y', alpha=0.35, zorder=0)
    # Rank annotations
    for xi in range(len(labels)):
        ax.text(xi, ax.get_ylim()[1] * 0.97, f'#{xi+1}',
                ha='center', va='top', fontsize=8, color='#555555')
    fig.tight_layout()
    _savefig(fig, 'fig05_importance_ranking')


# ── Figure 6: Cumulative emergence — DAI_core ────────────────────────────────

def fig06_cumulative_dai(cum_data, mode='natural'):
    print('  Fig 6: Cumulative emergence — DAI_core', flush=True)
    _cumulative_curve(cum_data, mode, 'dai_core',
                      'DAI_core',
                      'Figure 6: Cumulative Emergence — DAI_core\n'
                      'DAI_core as mechanisms are progressively ablated',
                      'fig06_cumulative_dai')


# ── Figure 7: Cumulative emergence — REAL_SCHEMA ─────────────────────────────

def fig07_cumulative_rs(cum_data, mode='natural'):
    print('  Fig 7: Cumulative emergence — REAL_SCHEMA', flush=True)
    _cumulative_curve(cum_data, mode, 'real_schema',
                      'REAL_SCHEMA',
                      'Figure 7: Cumulative Emergence — REAL_SCHEMA\n'
                      'Schema weight index as mechanisms are progressively ablated',
                      'fig07_cumulative_rs')


def _cumulative_curve(cum_data, mode, metric, ylabel, title, fig_name):
    """Line plot across cumulative ablation steps."""
    if not cum_data:
        print('    No cumulative data available.', flush=True)
        return
    steps, means, sems, x_labels = [], [], [], []
    for i, mid in enumerate(CUMUL_ORDER):
        ckey = f'CUM_{i + 1}_{mid}'
        sl   = cum_data.get(ckey, [])
        if not sl:
            continue
        agg  = _agg(sl, mode)
        m    = agg.get(f'{metric}_mean', np.nan)
        s    = agg.get(f'{metric}_sem', 0)
        n    = agg.get(f'{metric}_n', 0)
        steps.append(i + 1)
        means.append(m)
        sems.append(s)
        x_labels.append(f'+abl {mid}\n(n={n})')

    if not steps:
        print('    No cumulative conditions found.', flush=True)
        return

    means, sems = np.array(means), np.array(sems)
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, means - sems, means + sems, alpha=0.25, color=C_ABLATE)
    ax.plot(x, means, 'o-', color=C_ABLATE, lw=2, ms=7, zorder=4)

    # Label each step with the mechanism being ablated
    for xi, (mid, m) in enumerate(zip(CUMUL_ORDER[:len(steps)], means)):
        c = MECH_COLORS.get(mid, '#888888')
        ax.scatter(xi, m, color=c, s=80, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Mechanisms ablated (cumulative)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.grid(alpha=0.35, zorder=0)

    legend_patches = [Patch(color=MECH_COLORS.get(mid, '#888888'), label=f'{mid}')
                      for mid in CUMUL_ORDER[:len(steps)]]
    ax.legend(handles=legend_patches, loc='upper right', ncol=2,
              fontsize=8, framealpha=0.85, title='Mechanism ablated')
    fig.tight_layout()
    _savefig(fig, fig_name)


# ── Figure 8: Synergy interaction matrix ─────────────────────────────────────

def fig08_synergy_matrix(single_data, interact_data, mode='natural'):
    print('  Fig 8: Synergy interaction matrix', flush=True)
    PAIRS = [('M2', 'M7'), ('M2', 'M10'), ('M5', 'M8'), ('M6', 'M7'), ('M7', 'M10')]
    mechs = list({m for pair in PAIRS for m in pair})
    mechs_sorted = sorted(set(mechs))
    n = len(mechs_sorted)
    idx = {m: i for i, m in enumerate(mechs_sorted)}

    full_agg = _agg(single_data.get('FULL', []), mode) if single_data else {}
    full_dai = full_agg.get('dai_core_mean', 0)

    # Matrix: rows/cols = individual mechanisms, value = synergy index
    # Synergy = (ABL_A+B) - (ABL_A + ABL_B - FULL)
    # Positive synergy → mechanisms amplify each other
    mat     = np.full((n, n), np.nan)
    mat_sig = np.full((n, n), '', dtype=object)

    for mid_a, mid_b in PAIRS:
        ia, ib = idx[mid_a], idx[mid_b]
        lbl    = f'{mid_a}+{mid_b}'
        sl_ab  = interact_data.get(lbl, []) if interact_data else []
        sl_a   = single_data.get(f'ABLATE_{mid_a}', []) if single_data else []
        sl_b   = single_data.get(f'ABLATE_{mid_b}', []) if single_data else []

        if sl_ab and sl_a and sl_b:
            dai_ab = _agg(sl_ab, mode).get('dai_core_mean', np.nan)
            dai_a  = _agg(sl_a,  mode).get('dai_core_mean', np.nan)
            dai_b  = _agg(sl_b,  mode).get('dai_core_mean', np.nan)
            # Expected additive: full − (full − A) − (full − B) = A + B − full
            additive = dai_a + dai_b - full_dai
            synergy  = additive - dai_ab  # >0: joint ablation is WORSE than expected
            mat[ia, ib] = synergy
            mat[ib, ia] = synergy
            # Significance: compare joint vs additive expectation
            ab_vals = [s[mode]['dai_core'] for s in sl_ab if mode in s]
            a_vals  = [s[mode]['dai_core'] for s in sl_a  if mode in s]
            b_vals  = [s[mode]['dai_core'] for s in sl_b  if mode in s]
            if a_vals and b_vals and ab_vals:
                additive_vals = [a + b - full_dai for a, b in zip(a_vals, b_vals[:len(a_vals)])]
                if len(additive_vals) > 1 and len(ab_vals) > 1:
                    _, p = ttest_ind(ab_vals, additive_vals, equal_var=False)
                    sig  = _sig_stars(p)
                    mat_sig[ia, ib] = sig
                    mat_sig[ib, ia] = sig

    cmap = LinearSegmentedColormap.from_list('synergy', ['#d73027', '#ffffbf', '#1a9641'])
    vabs = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect='auto')

    for i in range(n):
        for j in range(n):
            v   = mat[i, j]
            sig = mat_sig[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > 0.6 * vabs else 'black'
                ax.text(j, i, f'{v:+.3f}\n{sig}', ha='center', va='center',
                        fontsize=9, color=c, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=12, color='#aaaaaa')

    ax.set_xticks(range(n))
    ax.set_xticklabels(mechs_sorted, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(mechs_sorted, fontsize=11)
    ax.set_title(
        'Figure 8: Synergy Interaction Matrix\n'
        'Positive = synergistic (joint > additive); Negative = redundant',
        fontsize=13, fontweight='bold', pad=12,
    )
    plt.colorbar(im, ax=ax, label='Synergy Index (DAI_core)', shrink=0.85)
    fig.tight_layout()
    _savefig(fig, 'fig08_synergy_matrix')


# ── Figure 9: Effect-size forest plot ─────────────────────────────────────────

def fig09_forest_plot(single_data, mode='natural'):
    print('  Fig 9: Effect-size forest plot', flush=True)
    if not single_data:
        print('    No single-ablation data available.', flush=True)
        return

    full_agg  = _agg(single_data.get('FULL', []), mode)
    full_vals = full_agg.get('dai_core_vals', [])
    full_n    = len(full_vals)

    rows = []
    for mid in ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','MB']:
        sl   = single_data.get(f'ABLATE_{mid}', [])
        if not sl:
            continue
        agg  = _agg(sl, mode)
        vals = agg.get('dai_core_vals', [])
        n    = len(vals)
        if n < 2 or full_n < 2:
            continue
        d    = _cohen_d(full_vals, vals)
        _, p = ttest_ind(full_vals, vals, equal_var=False)
        # 95% CI on Cohen's d (Hedges correction approximation)
        se_d = np.sqrt((full_n + n) / (full_n * n) + d**2 / (2 * (full_n + n - 2)))
        ci_lo, ci_hi = d - 1.96 * se_d, d + 1.96 * se_d
        rows.append({'mid': mid, 'd': d, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                     'p': p, 'n': n, 'sig': _sig_stars(p)})

    rows.sort(key=lambda r: r['d'], reverse=True)

    y   = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.55 + 1.5)))

    for yi, row in enumerate(rows):
        c = C_ABLATE if row['p'] < 0.05 else C_NEUTRAL
        ax.plot([row['ci_lo'], row['ci_hi']], [yi, yi], color=c, lw=2.5, zorder=3)
        ax.scatter(row['d'], yi, s=80, color=c, zorder=4)
        ax.text(max(row['ci_hi'], 0) + 0.05, yi,
                f"d={row['d']:+.2f}  {row['sig']}  n={row['n']}",
                va='center', fontsize=9.5)

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f'Ablate {r["mid"]}' for r in rows], fontsize=11)
    ax.set_xlabel("Cohen's d  (Full − Ablated, DAI_core)", fontsize=12)
    ax.set_title(
        "Figure 9: Effect-Size Forest Plot\n"
        "Cohen's d for each single-mechanism ablation (DAI_core)",
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.grid(axis='x', alpha=0.35)
    # Magnitude guide
    for xv, lbl in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axvline(xv,  color='#cccccc', lw=0.8, ls=':')
        ax.axvline(-xv, color='#cccccc', lw=0.8, ls=':')

    legend_patches = [
        Patch(color=C_ABLATE, label='p<0.05'),
        Patch(color=C_NEUTRAL, label='p≥0.05'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', framealpha=0.85)
    fig.tight_layout()
    _savefig(fig, 'fig09_forest_plot')


# ── Figure 10: Summary 4-panel ────────────────────────────────────────────────

def fig10_summary(single_data, mode='natural'):
    print('  Fig 10: Summary 4-panel', flush=True)
    if not single_data:
        print('    No single-ablation data available.', flush=True)
        return

    full_agg  = _agg(single_data.get('FULL', []), mode)
    full_dai  = full_agg.get('dai_core_vals', [])
    full_rs   = full_agg.get('real_schema_vals', [])
    full_di   = full_agg.get('distortion_vals', [])
    full_ret  = full_agg.get('retention_mean_vals', [])

    metrics   = [
        ('dai_core',       'DAI_core',    full_dai,  '#2166ac'),
        ('real_schema',    'REAL_SCHEMA', full_rs,   '#1a9641'),
        ('distortion',     'Distortion',  full_di,   '#d73027'),
        ('retention_mean', 'Retention\n(mean)',  full_ret,  '#762a83'),
    ]

    x      = np.arange(len(MECH_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes   = axes.ravel()

    for ax_i, (mkey, mlabel, full_vals, color) in enumerate(metrics):
        ax     = axes[ax_i]
        means, sems, colors = [], [], []
        for cname in MECH_ORDER:
            sl  = single_data.get(cname, [])
            if not sl:
                means.append(np.nan); sems.append(0); colors.append(C_NEUTRAL)
                continue
            agg = _agg(sl, mode)
            m   = agg.get(f'{mkey}_mean', np.nan)
            s   = agg.get(f'{mkey}_sem', 0)
            v   = agg.get(f'{mkey}_vals', [])
            if cname == 'FULL':
                c = color
            elif full_vals and v:
                _, p = ttest_ind(full_vals, v, equal_var=False)
                c = C_ABLATE if p < 0.05 else C_NEUTRAL
            else:
                c = C_NEUTRAL
            means.append(m); sems.append(s); colors.append(c)

        ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor='white',
               linewidth=0.7, error_kw=dict(elinewidth=1.5, ecolor='#333333'),
               zorder=3)
        full_m = full_agg.get(f'{mkey}_mean', np.nan)
        if not np.isnan(full_m):
            ax.axhline(full_m, color=color, lw=1.5, ls='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(MECH_XLABELS, fontsize=7.5, ha='center')
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(f'{mlabel}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Figure 10: Ablation Study Summary\nAll metrics — single-mechanism ablations',
        fontsize=14, fontweight='bold', y=1.01,
    )
    n_seeds = max(
        len(single_data.get('FULL', [])), 0
    )
    fig.text(0.5, -0.01, f'mode={mode}  n_seeds={n_seeds}  95% CI error bars',
             ha='center', fontsize=10, color='#666666')
    fig.tight_layout()
    _savefig(fig, 'fig10_summary')


# ── Master generator ──────────────────────────────────────────────────────────

def generate_all_figures(mode='natural', fig_list=None):
    print(f'\nGenerating figures (mode={mode}) -> {FIG_DIR}', flush=True)

    single_data   = _load('single_ablations')
    cum_data      = _load('cumulative_ablations')
    interact_data = _load('interaction_ablations')
    imp_pkl       = _load('importance_analysis')
    imp_data      = imp_pkl.get('importance', {}) if imp_pkl else None

    # Interaction data keyed by pair label
    interact_dict = {}
    if interact_data:
        for k, v in interact_data.items():
            # Keys like 'M2+M7', stored in interact_data directly
            interact_dict[k] = v

    fig_funcs = {
        1:  lambda: fig01_single_dai(single_data, mode),
        2:  lambda: fig02_single_rs(single_data, mode),
        3:  lambda: fig03_single_dist(single_data, mode),
        4:  lambda: fig04_retention_heatmap(single_data, mode),
        5:  lambda: fig05_importance_ranking(imp_data, mode),
        6:  lambda: fig06_cumulative_dai(cum_data, mode),
        7:  lambda: fig07_cumulative_rs(cum_data, mode),
        8:  lambda: fig08_synergy_matrix(single_data, interact_dict, mode),
        9:  lambda: fig09_forest_plot(single_data, mode),
        10: lambda: fig10_summary(single_data, mode),
    }

    to_run = fig_list if fig_list else list(range(1, 11))
    for n in to_run:
        fn = fig_funcs.get(n)
        if fn:
            try:
                fn()
            except Exception as e:
                print(f'  Fig {n} ERROR: {e}', flush=True)
                import traceback; traceback.print_exc()

    print(f'\nAll figures saved in {FIG_DIR}', flush=True)


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ablation figures')
    parser.add_argument('--mode', default='natural', choices=['natural', 'hyper'])
    parser.add_argument('--fig', nargs='*', type=int, help='Specific figure numbers to generate')
    args = parser.parse_args()
    generate_all_figures(mode=args.mode, fig_list=args.fig)
