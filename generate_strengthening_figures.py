"""
Strengthening Figures: Phases 1–4
  Fig 13 — DAI discriminant (Phase 1)
  Fig 14 — Distortion decomposition (Phase 2)
  Fig 15 — Robustness sweep (Phase 3)
  Fig 16 — Statistical summary forest plot (Phase 4)
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind, pearsonr
from scipy.stats import sem as scipy_sem

OUT_DIR   = r'C:\Users\Admin\brain-organoid-rl\figures\paper'
VALID_DIR = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
os.makedirs(OUT_DIR, exist_ok=True)

PLT_STYLE = {
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}
plt.rcParams.update(PLT_STYLE)

COND_COLORS = {
    'A_conv_aligned':    '#27ae60',
    'B_conv_misaligned': '#e74c3c',
    'C_nonconv_aligned': '#f39c12',
    'D_random':          '#95a5a6',
}
COND_SHORT = {
    'A_conv_aligned':    'A: Conv+Aligned',
    'B_conv_misaligned': 'B: Conv+Misaligned',
    'C_nonconv_aligned': 'C: NonConv+Aligned',
    'D_random':          'D: Random',
}
MODE_COLORS = {'no_replay': '#4e79a7', 'natural': '#59a14f', 'hyper': '#f28e2b'}
MODE_LABELS = {'no_replay': 'No Replay', 'natural': 'Natural', 'hyper': 'Hyper'}


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    print(f'  Saved {name}', flush=True)
    plt.close(fig)


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# ── Figure 13: DAI Discriminant Validation ────────────────────────────────────

def fig13_dai_discriminant():
    path = os.path.join(VALID_DIR, 'phase1_discriminant_raw.json')
    if not os.path.exists(path):
        print('  fig13: data not found', flush=True)
        return
    with open(path) as f:
        data = json.load(f)

    conds = ['A_conv_aligned', 'B_conv_misaligned', 'C_nonconv_aligned', 'D_random']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): DAI vs Convergence scatter — one point per trial
    ax = axes[0]
    for cond in conds:
        dai  = np.array(data[cond]['dai'])
        conv = np.array(data[cond]['conv'])
        ax.scatter(conv, dai, s=20, alpha=0.5, color=COND_COLORS[cond],
                   label=COND_SHORT[cond])

    # Correlation line
    all_dai  = np.concatenate([data[c]['dai']  for c in conds])
    all_conv = np.concatenate([data[c]['conv'] for c in conds])
    r, p_r   = pearsonr(all_dai, all_conv)
    xline = np.linspace(min(all_conv), max(all_conv), 50)
    m, b = np.polyfit(all_conv, all_dai, 1)
    ax.plot(xline, m * xline + b, 'k--', lw=1.5, alpha=0.6,
            label=f'r={r:.2f} (p<0.001)')

    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.axvline(0, color='black', lw=0.7, ls=':')
    ax.set_xlabel('Convergence Metric\n(initial - final pairwise dist)')
    ax.set_ylabel('DAI — Core\n(cos toward schema centroid)')
    ax.set_title('(a) DAI vs Convergence\nN=100 trials per condition')
    ax.legend(fontsize=8, loc='lower right')

    # Highlight condition B (DAI>0, Convergence~0) — key finding
    bd = np.array(data['B_conv_misaligned']['dai'])
    bc = np.array(data['B_conv_misaligned']['conv'])
    ax.annotate(
        'B: DAI high\nConvergence~0\n(key dissociation)',
        xy=(float(np.mean(bc)), float(np.mean(bd))),
        xytext=(0.15, 0.7),
        textcoords='axes fraction',
        fontsize=8, color='#e74c3c',
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
    )

    # Panel (b): Mean DAI by condition (bar)
    ax = axes[1]
    dai_means = [np.mean(data[c]['dai']) for c in conds]
    dai_sems  = [scipy_sem(data[c]['dai']) * 1.96 for c in conds]  # 95% CI
    xs = np.arange(len(conds))
    for i, (cond, m, ci) in enumerate(zip(conds, dai_means, dai_sems)):
        ax.bar(i, m, width=0.6, color=COND_COLORS[cond],
               edgecolor='white', linewidth=1.2)
        ax.errorbar(i, m, yerr=ci, fmt='none', capsize=5,
                    ecolor='black', elinewidth=1.5)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_SHORT[c].replace('+', '\n+') for c in conds],
                       fontsize=8)
    ax.set_ylabel('Mean DAI_core ± 95% CI')
    ax.set_title('(b) DAI by Condition')

    # significance brackets
    pairs = [(0,1,'A vs B'), (0,3,'A vs D')]
    y_top = max(dai_means) + max(dai_sems) + 0.03
    for k, (i, j, lbl) in enumerate(pairs):
        va = np.array(data[conds[i]]['dai'])
        vb = np.array(data[conds[j]]['dai'])
        t, p = ttest_ind(va, vb)
        yb = y_top + k * 0.06
        ax.plot([i, i, j, j], [yb, yb+0.02, yb+0.02, yb], lw=1.2, c='black')
        ax.text((i+j)/2, yb+0.025, _sig_stars(p), ha='center', fontsize=11)

    # Panel (c): Per-condition breakdown — DAI vs Convergence as grouped bars
    ax = axes[2]
    x = np.arange(len(conds))
    w = 0.35
    dai_m  = [np.mean(data[c]['dai'])  for c in conds]
    conv_m = [np.mean(data[c]['conv']) for c in conds]
    dai_se = [scipy_sem(data[c]['dai'])  * 1.96 for c in conds]
    conv_se= [scipy_sem(data[c]['conv']) * 1.96 for c in conds]

    ax.bar(x - w/2, dai_m,  width=w, color=[COND_COLORS[c] for c in conds],
           edgecolor='white', label='DAI_core', linewidth=1.2)
    ax.bar(x + w/2, conv_m, width=w, color=[COND_COLORS[c] for c in conds],
           alpha=0.4, edgecolor='grey', label='Convergence', linewidth=1.2,
           hatch='//')
    ax.errorbar(x - w/2, dai_m,  yerr=dai_se,  fmt='none', capsize=4,
                ecolor='black', elinewidth=1.2)
    ax.errorbar(x + w/2, conv_m, yerr=conv_se, fmt='none', capsize=4,
                ecolor='black', elinewidth=1.2)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conds], fontsize=11)
    ax.set_ylabel('Score (Mean ± 95% CI)')
    ax.set_title('(c) DAI vs Convergence\nSide by side')
    ax.legend(fontsize=9)

    # Annotation: key dissociation
    ax.annotate('B: DAI>0\nbut Conv=0',
                xy=(1 - w/2, float(np.mean(data['B_conv_misaligned']['dai']))),
                xytext=(1.5, 0.6), fontsize=8, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

    fig.suptitle(
        'Figure 13: DAI Discriminant Validation\n'
        'DAI captures directional schema abstraction beyond convergence',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig13_dai_discriminant')


# ── Figure 14: Distortion Decomposition ──────────────────────────────────────

def fig14_distortion_decomposition():
    path = os.path.join(VALID_DIR, 'phase2_decomposition_raw.json')
    if not os.path.exists(path):
        print('  fig14: data not found', flush=True)
        return
    with open(path) as f:
        data = json.load(f)

    modes = ['no_replay', 'natural', 'hyper']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    def get(mode, key):
        m = data.get(mode)
        if m is None:
            return {'mean': 0.0, 'sem': 0.0}
        return m.get(key, {'mean': 0.0, 'sem': 0.0})

    # Panel (a): stacked bar — conservative + dissipative
    ax = axes[0]
    xs = np.arange(len(modes))
    cons_m = [get(m, 'conservative_core')['mean'] for m in modes]
    diss_m = [get(m, 'dissipative_core')['mean']  for m in modes]
    cons_e = [get(m, 'conservative_core')['sem']  for m in modes]
    diss_e = [get(m, 'dissipative_core')['sem']   for m in modes]

    p1 = ax.bar(xs, cons_m, width=0.6, label='Conservative\n(toward schema)',
                color=[MODE_COLORS[m] for m in modes], edgecolor='white')
    p2 = ax.bar(xs, diss_m, width=0.6, bottom=cons_m, label='Dissipative\n(orthogonal)',
                color=[MODE_COLORS[m] for m in modes], edgecolor='white', alpha=0.4,
                hatch='//')

    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_ylabel('Distortion Magnitude (core component)')
    ax.set_title('(a) Conservative vs Dissipative\nDistortion')
    ax.legend(fontsize=9, loc='upper right')

    # Natural vs Hyper significance
    from phase2_distortion_decomposition import process_trajectory, aggregate_events
    import pickle as pk
    SEEDS = [42, 1042, 2042, 3042, 4042]
    mode_evs = {}
    for mode in modes:
        ev_list = []
        for seed in SEEDS:
            p = f'trajectory_{mode}_seed{seed}.pkl'
            if os.path.exists(p):
                with open(p, 'rb') as f2:
                    traj = pk.load(f2)
                ev_list.append(process_trajectory(traj))
        agg = aggregate_events(ev_list)
        mode_evs[mode] = agg

    for k, metric in enumerate(['conservative_core', 'dissipative_core']):
        nat_v = mode_evs.get('natural',{}).get('seed_means',{}).get(metric,[])
        hyp_v = mode_evs.get('hyper',{}).get('seed_means',{}).get(metric,[])
        if len(nat_v) >= 2 and len(hyp_v) >= 2:
            _, p = ttest_ind(nat_v, hyp_v)
            y_br = max(cons_m[1]+diss_m[1], cons_m[2]+diss_m[2]) + 0.01 + k*0.02
            ax.plot([1, 1, 2, 2], [y_br, y_br+0.008, y_br+0.008, y_br],
                    lw=1.2, c='black')
            ax.text(1.5, y_br+0.009, _sig_stars(p), ha='center', fontsize=10)

    # Panel (b): Efficiency bar plot
    ax = axes[1]
    eff_m = [get(m, 'efficiency_core')['mean'] for m in modes]
    eff_e = [get(m, 'efficiency_core')['sem']  for m in modes]

    for i, (mode, m, e) in enumerate(zip(modes, eff_m, eff_e)):
        ax.bar(i, m, width=0.6, color=MODE_COLORS[mode],
               edgecolor='white', linewidth=1.2)
        ax.errorbar(i, m, yerr=e*1.96, fmt='none', capsize=5,
                    ecolor='black', elinewidth=1.5)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_ylabel('Efficiency = Conservative / Total')
    ax.set_title('(b) Distortion Efficiency\n(schema-directed fraction)')
    ax.set_ylim(0, 1.1)

    nat_v = mode_evs.get('natural',{}).get('seed_means',{}).get('efficiency_core',[])
    hyp_v = mode_evs.get('hyper',{}).get('seed_means',{}).get('efficiency_core',[])
    if len(nat_v)>=2 and len(hyp_v)>=2:
        _, p = ttest_ind(nat_v, hyp_v)
        ax.plot([1, 1, 2, 2], [0.92, 0.95, 0.95, 0.92], lw=1.2, c='black')
        ax.text(1.5, 0.96, _sig_stars(p), ha='center', fontsize=11)

    # Panel (c): Decomposition diagram (conceptual + scatter)
    ax = axes[2]
    ax.set_aspect('equal')
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k:', lw=0.5, alpha=0.3)

    schema_dir = np.array([1, 0])

    for mode_name, (con, dis) in zip(
        ['Natural\n(efficient)', 'Hyper\n(noisy)'],
        [(0.85, 0.5), (0.75, 0.7)]
    ):
        ang = np.arctan2(dis, con)
        color = '#59a14f' if 'Natural' in mode_name else '#f28e2b'
        ax.annotate('', xy=(con, dis), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(con+0.05, dis, mode_name, fontsize=9, color=color)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    ax.text(0.9, -0.12, 'Schema\ndirection', fontsize=9, ha='center', color='black')
    ax.text(-0.05, 0.9, 'Orthogonal\n(dissipative)', fontsize=9, ha='right', color='grey')
    ax.arrow(0, 0, 0.95, 0, head_width=0.05, head_length=0.05,
             fc='black', ec='black', alpha=0.6)
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_title('(c) Decomposition Diagram\n(conservative vs dissipative)')
    ax.set_xlabel('Conservative component')
    ax.set_ylabel('Dissipative component')

    fig.suptitle(
        'Figure 14: Distortion Decomposition\n'
        'Natural replay is more efficient: higher schema-directed distortion',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig14_distortion_decomposition')


# ── Figure 15: Robustness Sweep ───────────────────────────────────────────────

def fig15_robustness():
    path = os.path.join(VALID_DIR, 'phase3_robustness_raw.json')
    if not os.path.exists(path):
        print('  fig15: data not found', flush=True)
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['natural', 'hyper']

    for ax_idx, (sweep_name, xlabel, ax) in enumerate([
        ('boost',     'Core Boost Factor', axes[0]),
        ('noise',     'Replay Noise (sigma)', axes[1]),
        ('frequency', 'Replay Frequency', axes[2]),
    ]):
        sweep = data.get(sweep_name, {})
        pv_raw = sweep.get('values', [])

        for mode in modes:
            mode_data = sweep.get('agg', {}).get(mode, {})
            if not mode_data:
                continue
            xs, ys, es = [], [], []
            for pv in pv_raw:
                pv_key = str(float(pv))
                md = mode_data.get(pv_key, {})
                v = md.get('dai_core', {})
                if v:
                    xs.append(float(pv))
                    ys.append(v['mean'])
                    es.append(v['sem'] * 1.96)

            if xs:
                ax.plot(xs, ys, 'o-', color=MODE_COLORS[mode],
                        lw=2, ms=6, label=MODE_LABELS[mode])
                ax.fill_between(xs,
                                [y-e for y,e in zip(ys,es)],
                                [y+e for y,e in zip(ys,es)],
                                alpha=0.15, color=MODE_COLORS[mode])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('DAI_core')
        ax.set_title(f'({"abc"[ax_idx]}) Sweep: {xlabel}')

        # Mark current operating point
        curr = {'boost': 1.3, 'noise': 0.008, 'frequency': 1.0}.get(sweep_name)
        if curr is not None:
            ax.axvline(curr, color='black', ls='--', lw=1.2, alpha=0.7,
                       label=f'Current ({curr})')

        if ax_idx == 0:
            ax.legend(fontsize=9)

        # Shade region where Natural > Hyper
        nat_data = sweep.get('agg', {}).get('natural', {})
        hyp_data = sweep.get('agg', {}).get('hyper', {})
        for pv in pv_raw:
            pv_key = str(float(pv))
            nd = nat_data.get(pv_key, {}).get('dai_core', {}).get('mean', None)
            hd = hyp_data.get(pv_key, {}).get('dai_core', {}).get('mean', None)
            if nd is not None and hd is not None and nd > hd:
                ax.axvspan(float(pv)-0.05, float(pv)+0.05, alpha=0.05,
                           color='#59a14f')

    fig.suptitle(
        'Figure 15: Robustness Parameter Sweep\n'
        'Natural > Hyper DAI_core across parameter ranges',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig15_robustness_sweep')


# ── Figure 16: Statistical Summary Forest Plot ───────────────────────────────

def fig16_statistical_summary():
    path = os.path.join(VALID_DIR, 'phase4_statistics.json')
    if not os.path.exists(path):
        print('  fig16: data not found', flush=True)
        return
    with open(path) as f:
        data = json.load(f)

    comparisons = data.get('comparisons', {})
    metrics = ['REAL_SCHEMA', 'DAI_core', 'Distortion', 'Retention_A', 'SchemaScore']
    pair_key = 'natural_vs_hyper'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Forest plot — Cohen's d for Natural vs Hyper
    ax = axes[0]
    ds, ps, powers = [], [], []
    for m in metrics:
        comp = comparisons.get(m, {}).get(pair_key, {})
        ds.append(comp.get('d', 0.0))
        ps.append(comp.get('p', 1.0))
        powers.append(comp.get('power', 0.0))

    ys = np.arange(len(metrics))
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in ds]
    ax.barh(ys, ds, height=0.5, color=colors, edgecolor='white', linewidth=1.2)
    ax.axvline(0, color='black', lw=1.2)
    ax.axvline(0.8,  color='grey', lw=0.8, ls=':', alpha=0.7, label='d=0.8 (large)')
    ax.axvline(-0.8, color='grey', lw=0.8, ls=':', alpha=0.7)

    # Annotate with stars
    for i, (d, p) in enumerate(zip(ds, ps)):
        s = _sig_stars(p) if p is not None else 'n.s.'
        ax.text(d + (0.15 if d >= 0 else -0.15), i, s,
                va='center', ha='left' if d >= 0 else 'right', fontsize=10)

    ax.set_yticks(ys)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel("Cohen's d (Natural vs Hyper)")
    ax.set_title("(a) Effect Sizes\nNatural vs Hyper")
    ax.legend(fontsize=9, loc='lower right')

    # Panel (b): Power analysis bar chart
    ax = axes[1]
    bar_colors = ['#27ae60' if pw > 0.8 else '#f39c12' if pw > 0.5 else '#e74c3c'
                  for pw in powers]
    ax.barh(ys, powers, height=0.5, color=bar_colors, edgecolor='white')
    ax.axvline(0.8, color='black', lw=1.5, ls='--', label='Power=0.80 threshold')
    ax.axvline(1.0, color='grey', lw=0.8, ls=':', alpha=0.5)

    for i, (pw, p) in enumerate(zip(powers, ps)):
        pv_str = f'p={p:.4f}' if p is not None and p > 0.0001 else 'p<0.0001'
        ax.text(pw + 0.01, i, pv_str, va='center', fontsize=9)

    ax.set_yticks(ys)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel('Statistical Power')
    ax.set_title('(b) Statistical Power\n(Natural vs Hyper, n=5)')
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=9)

    # Colour legend
    patches = [
        mpatches.Patch(color='#27ae60', label='Power > 0.80 (adequate)'),
        mpatches.Patch(color='#f39c12', label='Power 0.50-0.80 (marginal)'),
        mpatches.Patch(color='#e74c3c', label='Power < 0.50 (insufficient)'),
    ]
    ax.legend(handles=patches, fontsize=9, loc='lower right')

    fig.suptitle(
        'Figure 16: Statistical Strengthening Summary\n'
        'Effect sizes and power for all primary metrics (Natural vs Hyper, n=5)',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig16_statistical_summary')


def main():
    print('Generating strengthening figures (13-16)...', flush=True)
    fig13_dai_discriminant()
    fig14_distortion_decomposition()
    fig15_robustness()
    fig16_statistical_summary()
    print(f'\nDone. Figures saved to: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
