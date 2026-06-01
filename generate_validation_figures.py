"""
Validation Figures 10–12:
  Fig 10 — DAI synthetic validation (Phase A)
  Fig 11 — REAL_SCHEMA validation (Phase B)
  Fig 12 — Seed 42 outlier analysis (Phase C)

Run AFTER validate_dai.py, validate_real_schema.py, audit_seed42.py.
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\Admin\brain-organoid-rl')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp

VALID_DIR = r'C:\Users\Admin\brain-organoid-rl\figures\validation'
OUT_DIR   = r'C:\Users\Admin\brain-organoid-rl\figures\paper'
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
    'aligned':     '#27ae60',
    'random':      '#95a5a6',
    'anti_aligned':'#e74c3c',
}
COND_LABELS = {
    'aligned':     'Aligned\n(toward schema)',
    'random':      'Random\n(null)',
    'anti_aligned':'Anti-aligned\n(away from schema)',
}

CASE_COLORS = {
    'strong_core':           '#27ae60',
    'random':                '#95a5a6',
    'core_unique_dominant':  '#e74c3c',
}
CASE_LABELS = {
    'strong_core':          'Strong Core\n(expected HIGH)',
    'random':               'Random\n(expected ≈ 0)',
    'core_unique_dominant': 'Unique Dominant\n(expected LOW)',
}


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    print(f'  Saved {name}.pdf/.png', flush=True)
    plt.close(fig)


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# ── Figure 10: DAI Validation ────────────────────────────────────────────────

def fig10_dai_validation():
    path = os.path.join(VALID_DIR, 'dai_validation_raw.json')
    if not os.path.exists(path):
        print('  fig10: data not found, skipping', flush=True)
        return

    with open(path) as f:
        data = json.load(f)

    conds = ['aligned', 'random', 'anti_aligned']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel (a): violin / box plot of DAI_core distribution
    ax = axes[0]
    parts = ax.violinplot(
        [data[c]['mean_core'] for c in conds],
        positions=range(len(conds)),
        showmedians=True, showextrema=True,
    )
    for k, cond in enumerate(conds):
        parts['bodies'][k].set_facecolor(COND_COLORS[cond])
        parts['bodies'][k].set_alpha(0.7)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=9)
    ax.set_ylabel('DAI — Core Component\ncos(Δcentroid, toward schema)')
    ax.set_title('(a) DAI Distribution by Condition')

    # Significance brackets
    vals = [np.array(data[c]['mean_core']) for c in conds]
    y_top = max(np.max(v) for v in vals) + 0.05
    for k, (i, j) in enumerate([(0,1),(1,2),(0,2)]):
        t, p = ttest_ind(vals[i], vals[j])
        dy = 0.06 * (k+1)
        ax.plot([i, i, j, j], [y_top+dy, y_top+dy+0.02, y_top+dy+0.02, y_top+dy],
                lw=1.2, c='black')
        ax.text((i+j)/2, y_top+dy+0.025, _sig_stars(p),
                ha='center', va='bottom', fontsize=10)

    # Panel (b): histograms overlaid
    ax = axes[1]
    bins = np.linspace(-1.1, 1.1, 40)
    for cond in conds:
        ax.hist(data[cond]['mean_core'], bins=bins, alpha=0.55,
                color=COND_COLORS[cond], label=cond.replace('_', ' '),
                edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('DAI — Core Component')
    ax.set_ylabel('Count (N=100 trials)')
    ax.set_title('(b) DAI Histogram')
    ax.legend(fontsize=9)

    # Panel (c): mean ± 95% CI
    ax = axes[2]
    means  = [np.mean(data[c]['mean_core']) for c in conds]
    sems   = [np.std(data[c]['mean_core']) / np.sqrt(len(data[c]['mean_core']))
              for c in conds]
    ci95   = [1.96 * s for s in sems]
    xs     = np.arange(len(conds))

    for i, (cond, m, ci) in enumerate(zip(conds, means, ci95)):
        ax.bar(i, m, width=0.6, color=COND_COLORS[cond],
               edgecolor='white', linewidth=1.2)
        ax.errorbar(i, m, yerr=ci, fmt='none', capsize=6,
                    ecolor='black', elinewidth=1.5)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axhline(+1, color=COND_COLORS['aligned'],     lw=0.8, ls=':', alpha=0.6,
               label='Expected +1')
    ax.axhline(-1, color=COND_COLORS['anti_aligned'], lw=0.8, ls=':', alpha=0.6,
               label='Expected −1')
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=9)
    ax.set_ylabel('Mean DAI ± 95% CI')
    ax.set_title('(c) Mean ± 95% CI')
    ax.legend(fontsize=8)

    fig.suptitle(
        'Figure 10: DAI Synthetic Validation\n'
        'N=100 synthetic trajectories per condition',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig10_dai_validation')


# ── Figure 11: REAL_SCHEMA Validation ───────────────────────────────────────

def fig11_real_schema_validation():
    path = os.path.join(VALID_DIR, 'real_schema_validation_raw.json')
    if not os.path.exists(path):
        print('  fig11: data not found, skipping', flush=True)
        return

    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel (a): per-case box plots
    ax = axes[0]
    cases = ['strong_core', 'random', 'core_unique_dominant']
    case_data = [data['per_case'][c] for c in cases]
    bp = ax.boxplot(case_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, case in zip(bp['boxes'], cases):
        patch.set_facecolor(CASE_COLORS[case])
        patch.set_alpha(0.7)

    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticks(range(1, len(cases)+1))
    ax.set_xticklabels([CASE_LABELS[c] for c in cases], fontsize=9)
    ax.set_ylabel('REAL_SCHEMA Index')
    ax.set_title('(a) REAL_SCHEMA by Construction')

    # Significance
    vals = [np.array(case_data[i]) for i in range(len(cases))]
    y_top = max(np.max(v) for v in vals) + 0.05
    for k, (i, j) in enumerate([(0,1),(1,2),(0,2)]):
        t, p = ttest_ind(vals[i], vals[j])
        dy = 0.07 * (k+1)
        x1, x2 = i+1, j+1
        ax.plot([x1, x1, x2, x2],
                [y_top+dy, y_top+dy+0.03, y_top+dy+0.03, y_top+dy],
                lw=1.2, c='black')
        ax.text((x1+x2)/2, y_top+dy+0.035, _sig_stars(p),
                ha='center', va='bottom', fontsize=10)

    # Panel (b): scaling curve
    ax = axes[1]
    scaling = data['scaling']
    strengths = sorted(float(k) for k in scaling.keys())
    rs_vals   = [scaling[str(s)] for s in strengths]

    ax.plot(strengths, rs_vals, 'o-', color='#2980b9', lw=2, ms=5)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(strengths, 0, rs_vals,
                    where=[r > 0 for r in rs_vals],
                    alpha=0.15, color='#27ae60', label='RS > 0')
    ax.fill_between(strengths, rs_vals, 0,
                    where=[r < 0 for r in rs_vals],
                    alpha=0.15, color='#e74c3c', label='RS < 0')
    ax.set_xlabel('Core–Core Weight Strength')
    ax.set_ylabel('REAL_SCHEMA Index')
    ax.set_title('(b) Scaling Curve')
    ax.legend(fontsize=9)

    # Check monotonicity
    mono = all(rs_vals[i] <= rs_vals[i+1] for i in range(len(rs_vals)-1))
    ax.text(0.05, 0.95, f'Monotonic: {"YES ✓" if mono else "NO ✗"}',
            transform=ax.transAxes, fontsize=10,
            color='#27ae60' if mono else '#e74c3c', va='top')

    # Panel (c): noise robustness
    ax = axes[2]
    noise = data['noise_robust']
    sigmas = sorted(float(k) for k in noise.keys())
    means_ = [noise[str(s)]['mean'] for s in sigmas]
    stds_  = [noise[str(s)]['std']  for s in sigmas]
    errs_  = [1.96 * noise[str(s)]['std'] for s in sigmas]

    ax.plot(sigmas, means_, 'o-', color='#27ae60', lw=2, ms=6, label='Strong core')
    ax.fill_between(sigmas,
                    [m-e for m, e in zip(means_, errs_)],
                    [m+e for m, e in zip(means_, errs_)],
                    alpha=0.25, color='#27ae60')
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5, label='RS=0')
    ax.set_xlabel('Noise Magnitude (σ)')
    ax.set_ylabel('REAL_SCHEMA Index')
    ax.set_title('(c) Noise Robustness')
    ax.legend(fontsize=9)

    fig.suptitle(
        'Figure 11: REAL_SCHEMA Synthetic Validation\n'
        'N=100 synthetic networks per case',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig11_real_schema_validation')


# ── Figure 12: Seed 42 Outlier Analysis ─────────────────────────────────────

def fig12_seed42_outlier():
    path = os.path.join(VALID_DIR, 'seed42_audit_raw.json')
    if not os.path.exists(path):
        print('  fig12: data not found, skipping', flush=True)
        return

    with open(path) as f:
        data = json.load(f)

    anomaly = data['anomaly_trajectory']
    refs    = data['reference_trajectories']
    reruns  = data['rerun_results']

    # Weight-based REAL_SCHEMA values from _distortion_paper.py run
    # (not in trajectory pkl — taken from known 5-seed run output)
    wb_rs = {42: 0.0473, 1042: 0.4725, 2042: 0.4939, 3042: 0.4841, 4042: 0.4933}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel (a): retention vs weight-based REAL_SCHEMA
    ax = axes[0]
    ref_seeds = [int(r['label'].replace('seed', '').replace('_hyper', '').split('_')[0][-4:])
                 for r in refs]
    ref_ret = [r['retention_A'] for r in refs]
    ref_rs_wb = [wb_rs.get(s, np.nan) for s in [1042, 2042, 3042, 4042]]
    ax.scatter(ref_ret, ref_rs_wb, s=80, color='#f28e2b',
               label='Hyper seeds 1042-4042', zorder=5)
    ax.scatter([anomaly['retention_A']], [wb_rs[42]],
               s=250, color='#e74c3c', marker='*', zorder=6, label='Seed 42 (anomaly)')

    if reruns:
        re_ret = [r['ret_A']       for r in reruns]
        re_rs  = [r['REAL_SCHEMA'] for r in reruns]
        ax.scatter(re_ret, re_rs, s=40, color='#95a5a6', alpha=0.7,
                   marker='x', label='Seed 42 re-runs (x10)')

    ax.set_xlabel('Retention (Memory A)')
    ax.set_ylabel('REAL_SCHEMA Index\n(weight-based)')
    ax.set_title('(a) Retention vs Schema Formation')
    ax.legend(fontsize=8)
    ax.axvline(0.45, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.text(anomaly['retention_A']+0.01, wb_rs[42]+0.02,
            f'RS={wb_rs[42]:.3f}\n(collapsed!)',
            fontsize=8, color='#e74c3c', fontweight='bold')

    # Panel (b): baseline scores comparison
    ax = axes[1]
    labels_seeds = ['42\n(anomaly)'] + [r['label'].split('seed')[-1] for r in refs]
    baseline_D   = [anomaly['baseline_scores'][3] if len(anomaly.get('baseline_scores',[])) > 3 else 0.0]
    for r in refs:
        bs = r.get('baseline_scores', [0]*4)
        baseline_D.append(bs[3] if len(bs) > 3 else 0.0)

    colors = ['#e74c3c'] + ['#f28e2b'] * len(refs)
    bars = ax.bar(range(len(labels_seeds)), baseline_D, color=colors, width=0.6,
                  edgecolor='white')
    ax.set_xticks(range(len(labels_seeds)))
    ax.set_xticklabels(labels_seeds, fontsize=10)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Baseline Score (Memory D)')
    ax.set_title('(b) Memory D Baseline Encoding')
    ax.axhline(0.05, color='red', ls=':', lw=1.5, alpha=0.7,
               label='Encoding threshold')
    ax.legend(fontsize=9)

    # Annotation
    ax.text(0, baseline_D[0] + 0.005,
            f'{baseline_D[0]:.3f}\n(not encoded!)',
            ha='center', fontsize=8, color='#e74c3c', fontweight='bold')

    # Panel (c): re-run distributions
    ax = axes[2]
    if reruns:
        re_ret = [r['ret_A']       for r in reruns]
        re_rs  = [r['REAL_SCHEMA'] for r in reruns]

        rep_ids = [r['rep'] for r in reruns]
        ax.plot(rep_ids, re_ret, 'o-', color='#4e79a7', lw=1.5, ms=6,
                label='Retention A')
        ax.plot(rep_ids, re_rs,  's-', color='#59a14f', lw=1.5, ms=6,
                label='REAL_SCHEMA')

        # Reference lines from original other seeds
        if refs:
            ax.axhline(np.mean([r['retention_A']       for r in refs]),
                       color='#4e79a7', ls='--', lw=1, alpha=0.6,
                       label=f'Ref ret_A mean ({np.mean([r["retention_A"] for r in refs]):.3f})')
            ax.axhline(np.mean([r['schema_ratio_final'] for r in refs]),
                       color='#59a14f', ls='--', lw=1, alpha=0.6,
                       label=f'Ref RS mean ({np.mean([r["schema_ratio_final"] for r in refs]):.3f})')

        ax.set_xlabel('Re-run Index')
        ax.set_ylabel('Score')
        ax.set_title('(c) Seed 42 Hyper: 10 Re-runs')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Re-run data\nnot available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('(c) Re-run data')

    fig.suptitle(
        'Figure 12: Seed 42 Hyper Replay Anomaly Analysis\n'
        'High retention + collapsed schema = runaway potentiation failure mode',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'fig12_seed42_outlier')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('Generating validation figures...', flush=True)
    fig10_dai_validation()
    fig11_real_schema_validation()
    fig12_seed42_outlier()
    print(f'\nAll validation figures saved to: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
